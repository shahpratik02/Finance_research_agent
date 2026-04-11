"""
Yahoo Finance MCP Server — Port 8001

Exposes Yahoo Finance data as a locally hosted HTTP tool server.
The main app's mcp_client.py calls this via POST /call.

Tools:
    get_ticker_info      — company info, price, fundamentals, valuation metrics
    get_ticker_news      — recent news articles for a ticker
    get_price_history    — historical OHLCV price data
    search               — search Yahoo Finance for stocks/ETFs/news
"""

import logging
import os
from typing import Any

import yfinance as yf
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [yahoo_finance] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Yahoo Finance MCP Server", version="1.0.0")

PORT = int(os.environ.get("YAHOO_FINANCE_MCP_PORT", 8001))

# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_ticker_info",
        "description": (
            "Get comprehensive stock data for a ticker: company info, current price, "
            "market cap, valuation metrics (P/E, EV/EBITDA), dividend yield, beta, "
            "52-week range, revenue, net income, and analyst targets."
        ),
        "parameters": {
            "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL, MSFT, NVDA"},
        },
        "required": ["symbol"],
    },
    {
        "name": "get_ticker_news",
        "description": "Fetch recent news articles and press releases for a stock ticker.",
        "parameters": {
            "symbol": {"type": "string", "description": "Ticker symbol"},
            "count": {"type": "integer", "description": "Max articles to return (default 10, max 50)"},
        },
        "required": ["symbol"],
    },
    {
        "name": "get_price_history",
        "description": (
            "Get historical OHLCV price data for a ticker. "
            "period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. "
            "interval: 1m, 5m, 15m, 30m, 1h, 1d, 5d, 1wk, 1mo."
        ),
        "parameters": {
            "symbol": {"type": "string"},
            "period": {"type": "string", "description": "Time period (default: 1mo)"},
            "interval": {"type": "string", "description": "Data interval (default: 1d)"},
        },
        "required": ["symbol"],
    },
    {
        "name": "search",
        "description": "Search Yahoo Finance for stocks, ETFs, and news articles.",
        "parameters": {
            "query": {"type": "string", "description": "Search query"},
            "search_type": {
                "type": "string",
                "description": "Filter results: 'all' | 'quotes' | 'news' (default: all)",
            },
        },
        "required": ["query"],
    },
]


# ── Request schema ──────────────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    tool: str
    arguments: dict = {}


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "server": "yahoo_finance", "port": PORT}


@app.get("/tools")
def list_tools():
    return {"tools": TOOLS}


@app.post("/call")
def call_tool(req: ToolCallRequest):
    tool = req.tool
    args = req.arguments
    logger.info(f"Tool call: {tool} args={args}")

    try:
        if tool == "get_ticker_info":
            return _get_ticker_info(args)
        elif tool == "get_ticker_news":
            return _get_ticker_news(args)
        elif tool == "get_price_history":
            return _get_price_history(args)
        elif tool == "search":
            return _search(args)
        else:
            return {"result": None, "error": f"Unknown tool: {tool}"}
    except Exception as e:
        logger.exception(f"Tool call failed: {tool}")
        return {"result": None, "error": str(e)}


# ── Tool implementations ────────────────────────────────────────────────────────

def _get_ticker_info(args: dict) -> dict:
    symbol = args.get("symbol", "").upper().strip()
    if not symbol:
        return {"result": None, "error": "symbol is required"}
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    # Return full info dict — mcp_client/source_normalizer will extract what it needs
    return {"result": info, "error": None}


def _get_ticker_news(args: dict) -> dict:
    symbol = args.get("symbol", "").upper().strip()
    count = min(int(args.get("count", 10)), 50)
    if not symbol:
        return {"result": None, "error": "symbol is required"}
    ticker = yf.Ticker(symbol)
    news = ticker.news or []
    return {"result": news[:count], "error": None}


def _get_price_history(args: dict) -> dict:
    symbol = args.get("symbol", "").upper().strip()
    period = args.get("period", "1mo")
    interval = args.get("interval", "1d")
    if not symbol:
        return {"result": None, "error": "symbol is required"}
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist.empty:
        return {"result": [], "error": None}
    data = []
    for date, row in hist.iterrows():
        data.append({
            "date": str(date.date()) if hasattr(date, "date") else str(date),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        })
    return {
        "result": {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "rows": len(data),
            "data": data,
        },
        "error": None,
    }


def _search(args: dict) -> dict:
    query = args.get("query", "").strip()
    search_type = args.get("search_type", "all")
    if not query:
        return {"result": None, "error": "query is required"}
    results = yf.Search(query)
    output: dict[str, Any] = {}
    if search_type in ("all", "quotes"):
        output["quotes"] = list(results.quotes or [])
    if search_type in ("all", "news"):
        output["news"] = list(results.news or [])
    return {"result": output, "error": None}


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Yahoo Finance MCP server on port {PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
