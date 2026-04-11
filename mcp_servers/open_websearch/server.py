"""
Open Web Search MCP Server — Port 8004

A Python-based web search server that aggregates results from multiple free sources:
  1. Bing News RSS (no API key, reliable) for news queries
  2. DuckDuckGo Instant Answer API (lightweight JSON endpoint) for general queries
  3. httpx fetch for page retrieval

Sources match the open-webSearch project design (Aas-ee/open-webSearch) but run
entirely in Python without Node.js.

Tools:
    search      — search the web (Bing News RSS + DuckDuckGo Instant Answer)
    fetch_web   — fetch and extract plain text from a URL
"""

import logging
import os
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [open_websearch] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Open Web Search MCP Server", version="1.0.0")

PORT = int(os.environ.get("OPEN_WEB_SEARCH_MCP_PORT", 8004))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search",
        "description": (
            "Search the web for current news and information. Uses Bing News RSS. "
            "No API key required. Use for recent news, current events, qualitative "
            "commentary, analyst opinions, or any topic not in structured finance APIs."
        ),
        "parameters": {
            "query": {"type": "string", "description": "Web search query"},
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 10, max 30)",
            },
        },
        "required": ["query"],
    },
    {
        "name": "fetch_web",
        "description": (
            "Fetch the text content of a web page by URL. "
            "Useful for reading a full article after finding it via search."
        ),
        "parameters": {
            "url": {"type": "string", "description": "Full URL (https://...)"},
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return (default 8000)",
            },
        },
        "required": ["url"],
    },
]


# ── Request schema ──────────────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    tool: str
    arguments: dict = {}


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "server": "open_websearch", "port": PORT}


@app.get("/tools")
def list_tools():
    return {"tools": TOOLS}


@app.post("/call")
async def call_tool(req: ToolCallRequest):
    tool = req.tool
    args = req.arguments
    logger.info(f"Tool call: {tool} args={args}")

    try:
        if tool == "search":
            return await _search(args)
        elif tool == "fetch_web":
            return await _fetch_web(args)
        else:
            return {"result": None, "error": f"Unknown tool: {tool}"}
    except Exception as e:
        logger.exception(f"Tool call failed: {tool}")
        return {"result": None, "error": str(e)}


# ── Tool implementations ────────────────────────────────────────────────────────

async def _search(args: dict) -> dict:
    query = args.get("query", "").strip()
    limit = min(int(args.get("limit", 10)), 30)

    if not query:
        return {"result": None, "error": "query is required"}

    results = []

    # Strategy 1: Bing News RSS (most reliable for financial news)
    try:
        bing_results = await _bing_news_rss(query, limit)
        results.extend(bing_results)
        logger.info(f"Bing News returned {len(bing_results)} results for: {query!r}")
    except Exception as e:
        logger.warning(f"Bing News RSS failed: {e}")

    # Strategy 2: DuckDuckGo Instant Answer API (lightweight, no JS)
    if len(results) < limit:
        try:
            ddg_results = await _ddg_instant(query)
            results.extend(ddg_results)
        except Exception as e:
            logger.warning(f"DuckDuckGo Instant Answer failed: {e}")

    # Deduplicate by URL
    seen_urls = set()
    deduped = []
    for r in results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped.append(r)

    deduped = deduped[:limit]
    return {
        "result": {"query": query, "count": len(deduped), "results": deduped},
        "error": None,
    }


async def _bing_news_rss(query: str, limit: int) -> list:
    """Fetch Bing News RSS feed — free, no API key, good for financial news."""
    encoded = quote_plus(query)
    url = f"https://www.bing.com/news/search?q={encoded}&format=RSS&count={limit}"
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    root = ET.fromstring(resp.text)
    items = root.findall(".//item")
    results = []
    for item in items[:limit]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        description = item.findtext("description") or ""
        pub_date = item.findtext("pubDate") or ""
        # Strip HTML from description
        description = _strip_html(description)
        results.append({
            "title": title.strip(),
            "url": link.strip(),
            "snippet": description.strip()[:400],
            "published": pub_date.strip(),
            "source": "bing_news",
        })
    return results


async def _ddg_instant(query: str) -> list:
    """DuckDuckGo Instant Answer API — lightweight JSON, no search results but useful for
    abstract/definition type answers."""
    encoded = quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1&no_html=1"
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    results = []
    # RelatedTopics contain web results
    for topic in data.get("RelatedTopics", [])[:5]:
        if isinstance(topic, dict) and "FirstURL" in topic:
            results.append({
                "title": topic.get("Text", "")[:100],
                "url": topic.get("FirstURL", ""),
                "snippet": topic.get("Text", "")[:400],
                "published": "",
                "source": "duckduckgo",
            })
    return results


async def _fetch_web(args: dict) -> dict:
    url = args.get("url", "").strip()
    max_chars = min(int(args.get("max_chars", 8000)), 50000)

    if not url:
        return {"result": None, "error": "url is required"}
    if not url.startswith(("http://", "https://")):
        return {"result": None, "error": "url must start with http:// or https://"}

    try:
        async with httpx.AsyncClient(
            headers=HEADERS, follow_redirects=True, timeout=15.0
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            text = _strip_html(resp.text) if "html" in content_type else resp.text
            text = text[:max_chars]
        return {
            "result": {"url": url, "chars": len(text), "content": text},
            "error": None,
        }
    except httpx.HTTPStatusError as e:
        return {"result": None, "error": f"HTTP {e.response.status_code} fetching {url}"}
    except Exception as e:
        return {"result": None, "error": f"Fetch failed: {str(e)}"}


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"&[a-zA-Z]+;", " ", html)  # HTML entities
    html = re.sub(r"\s+", " ", html).strip()
    return html


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Open Web Search MCP server on port {PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
