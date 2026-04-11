"""
MCP tool client.

Responsibilities:
- Provide a unified call interface for all four MCP servers:
      yahoo_finance      — stock prices, market cap, basic fundamentals
      fred               — macro time series (rates, inflation, unemployment)
      financial_datasets — financial statements, richer company data
      open_web_search    — recent news, general discovery
- Route tool calls to the correct MCP server based on tool name.
- Enforce the per-agent tool budget (tracked externally by the agent module).
- Return raw tool responses as dicts; normalization is done in source_normalizer.py.
- Retry once on transient network errors. Log and continue on persistent failures.
"""

import logging
import time
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ── Tool → provider routing table ─────────────────────────────────────────────
# Every tool exposed by any MCP server must have an entry here.
# Format: "tool_name": "provider_key"
# Provider key must match one of the keys in settings.mcp_url_for().

_TOOL_ROUTING: dict[str, str] = {
    # Yahoo Finance tools
    "get_ticker_info":     "yahoo_finance",
    "get_ticker_news":     "yahoo_finance",
    "get_price_history":   "yahoo_finance",
    "search":              "yahoo_finance",   # general Yahoo Finance search

    # FRED tools
    "get_series":          "fred",
    "search_series":       "fred",
    "get_series_info":     "fred",

    # Financial Datasets tools
    "get_income_statement":            "financial_datasets",
    "get_balance_sheet":               "financial_datasets",
    "get_cash_flow_statement":         "financial_datasets",
    "get_financial_metrics":           "financial_datasets",
    "get_financial_metrics_snapshot":  "financial_datasets",
    "get_stock_prices":                "financial_datasets",
    "get_stock_price_snapshot":        "financial_datasets",
    "get_company_facts":               "financial_datasets",
    "get_insider_trades":              "financial_datasets",
    "get_news":                        "financial_datasets",
    "get_analyst_estimates":           "financial_datasets",
    "get_filings":                     "financial_datasets",
    "get_filing_items":                "financial_datasets",
    "screen_stocks":                   "financial_datasets",

    # Open Web Search tools
    # Note: 'web_search' is the alias the LLM uses; the server exposes it as 'search'.
    # The _post_with_retry call translates it automatically (see _resolve_tool_name).
    "web_search":   "open_web_search",
    "search_web":   "open_web_search",   # secondary alias
    "fetch_web":    "open_web_search",
}

# ── Tool name aliases ──────────────────────────────────────────────────────────
# Maps client-facing tool names (what the LLM calls) → server-side tool names.
# Only entries that differ need to be listed.

_TOOL_ALIAS: dict[str, str] = {
    "web_search":  "search",   # LLM says web_search → server exposes 'search'
    "search_web":  "search",   # secondary alias
}

# ── Constants ──────────────────────────────────────────────────────────────────

_CALL_TIMEOUT_S = 30.0   # seconds per individual HTTP call
_RETRY_DELAY_S = 1.0     # wait before the one retry attempt
_RETRYABLE_STATUS = {500, 502, 503, 504}


# ── Public API ─────────────────────────────────────────────────────────────────

class ToolCallError(Exception):
    """Raised when a tool call fails after retrying and cannot be recovered."""


def call_tool(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Call a single MCP tool by name and return the raw result dict.

    The MCP server always returns:
        { "result": <data> | null, "error": <str> | null }

    This function:
    - routes to the correct server, 
    - retries once on transient HTTP errors,
    - raises ToolCallError only for persistent failures,
    - returns the full {"result": ..., "error": ...} dict on success
      so the caller (source_normalizer) can inspect both fields.

    Args:
        tool:      Name of the tool, must be in _TOOL_ROUTING.
        arguments: Dict of keyword arguments to pass to the tool.

    Returns:
        The full server response dict: {"result": ..., "error": ...}

    Raises:
        ToolCallError: On unknown tool, persistent network error, or non-2xx
                       response that doesn't recover after one retry.
    """
    provider = _resolve_provider(tool)
    base_url = settings.mcp_url_for(provider)
    url = f"{base_url}/call"
    payload = {"tool": tool, "arguments": arguments}

    logger.info(f"[mcp] {tool}({_fmt_args(arguments)}) → {provider}")

    # Some tool names are client-side aliases. Translate to the server's actual name.
    server_tool = _TOOL_ALIAS.get(tool, tool)
    payload = {"tool": server_tool, "arguments": arguments}

    return _post_with_retry(url, payload, tool=tool)


def list_tools(provider: str) -> list[dict]:
    """
    Return the tool manifest from a running MCP server.

    Args:
        provider: One of 'yahoo_finance', 'fred', 'financial_datasets',
                  'open_web_search'.

    Returns:
        List of tool descriptor dicts, each with 'name', 'description',
        'parameters', and 'required' keys.
    """
    base_url = settings.mcp_url_for(provider)
    url = f"{base_url}/tools"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json().get("tools", [])
    except Exception as e:
        logger.warning(f"[mcp] list_tools({provider}) failed: {e}")
        return []


def health_check(provider: str) -> bool:
    """
    Check whether a given MCP server is reachable and healthy.

    Returns True if the server replies with status='ok', False otherwise.
    """
    base_url = settings.mcp_url_for(provider)
    url = f"{base_url}/health"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json().get("status") == "ok"
    except Exception as e:
        logger.warning(f"[mcp] health_check({provider}) failed: {e}")
        return False


def all_providers() -> list[str]:
    """Return the list of configured provider names."""
    return ["yahoo_finance", "fred", "financial_datasets", "open_web_search"]


def provider_for_tool(tool: str) -> str:
    """
    Return the provider name for a given tool.

    Raises:
        ToolCallError: If the tool is not in the routing table.
    """
    return _resolve_provider(tool)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_provider(tool: str) -> str:
    """Map a tool name to its MCP provider. Raises ToolCallError if unknown."""
    # Handle 'search' ambiguity: if model says 'search', route to yahoo_finance.
    # For web/news intent, the model should use 'web_search' or 'fetch_web'.
    provider = _TOOL_ROUTING.get(tool)
    if provider is None:
        raise ToolCallError(
            f"Unknown tool {tool!r}. "
            f"Available tools: {sorted(_TOOL_ROUTING.keys())}"
        )
    return provider


def _post_with_retry(
    url: str,
    payload: dict,
    tool: str,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """
    POST to an MCP server endpoint with one retry on transient errors.

    Returns the parsed JSON response dict on success.
    Raises ToolCallError on persistent failure.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=_CALL_TIMEOUT_S) as client:
                resp = client.post(url, json=payload)

            if resp.status_code in _RETRYABLE_STATUS and attempt < max_attempts:
                logger.warning(
                    f"[mcp] {tool} attempt {attempt} → HTTP {resp.status_code}, retrying..."
                )
                time.sleep(_RETRY_DELAY_S)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Log server-side errors without raising (caller decides how to handle)
            if data.get("error"):
                logger.warning(f"[mcp] {tool} server error: {data['error']}")

            return data

        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(f"[mcp] {tool} attempt {attempt} timed out after {_CALL_TIMEOUT_S}s")
            if attempt < max_attempts:
                time.sleep(_RETRY_DELAY_S)

        except httpx.HTTPStatusError as e:
            last_error = e
            logger.error(
                f"[mcp] {tool} attempt {attempt} HTTP error "
                f"{e.response.status_code}: {e.response.text[:200]}"
            )
            # Non-retryable HTTP errors (e.g. 400, 404) → don't retry
            if e.response.status_code not in _RETRYABLE_STATUS:
                break

        except Exception as e:
            last_error = e
            logger.error(f"[mcp] {tool} attempt {attempt} unexpected error: {e}")
            if attempt < max_attempts:
                time.sleep(_RETRY_DELAY_S)

    raise ToolCallError(
        f"Tool {tool!r} failed after {max_attempts} attempt(s). "
        f"Last error: {last_error}"
    )


def _fmt_args(arguments: dict) -> str:
    """Format tool arguments for a compact log line."""
    parts = []
    for k, v in arguments.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "..."
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)
