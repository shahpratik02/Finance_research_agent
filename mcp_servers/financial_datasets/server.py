"""
Financial Datasets Proxy MCP Server — Port 8003

Proxies tool calls to the official Financial Datasets REST API at
https://api.financialdatasets.ai/ using the FINANCIAL_DATASETS_API_KEY.

This lets the main app's mcp_client.py call http://127.0.0.1:8003 uniformly
while the actual data comes from the remote API.

Tools:
    get_income_statement       — income statement (revenue, net income, EPS)
    get_balance_sheet          — balance sheet (assets, liabilities, equity)
    get_cash_flow_statement    — cash flow statement
    get_financial_metrics      — historical P/E, EV/EBITDA, margins, etc.
    get_financial_metrics_snapshot — latest financial metrics snapshot
    get_stock_prices           — historical OHLCV stock prices
    get_stock_price_snapshot   — latest price snapshot
    get_company_facts          — company info: sector, employees, market cap
    get_insider_trades         — insider buying/selling transactions
    get_news                   — recent financial news for a company or market
    get_analyst_estimates      — analyst consensus revenue/EPS estimates
    get_filings                — SEC filings list (10-K, 10-Q, 8-K)
    get_filing_items           — extract sections from SEC filings

Ref: https://docs.financialdatasets.ai/quickstart
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# Load .env from the project root (two levels up from this file)
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=True)
    except ImportError:
        pass  # python-dotenv not installed; rely on environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [financial_datasets] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Datasets Proxy MCP Server", version="1.0.0")

PORT = int(os.environ.get("FINANCIAL_DATASETS_MCP_PORT", 8003))
UPSTREAM = "https://api.financialdatasets.ai"

# ── Tool→endpoint map ──────────────────────────────────────────────────────────
# Format: (HTTP method, upstream path, query param names, body param names)
# query params → sent as ?param=value
# body params  → sent as JSON body (POST only)

TOOL_MAP: dict[str, dict] = {
    "get_income_statement": {
        "method": "GET",
        "path": "/financials/income-statements/",
        "query_params": ["ticker", "period", "limit"],
    },
    "get_balance_sheet": {
        "method": "GET",
        "path": "/financials/balance-sheets/",
        "query_params": ["ticker", "period", "limit"],
    },
    "get_cash_flow_statement": {
        "method": "GET",
        "path": "/financials/cash-flow-statements/",
        "query_params": ["ticker", "period", "limit"],
    },
    "get_financial_metrics": {
        "method": "GET",
        "path": "/financial-metrics/",
        "query_params": ["ticker", "period", "limit"],
    },
    "get_financial_metrics_snapshot": {
        "method": "GET",
        "path": "/financial-metrics/snapshot/",
        "query_params": ["ticker"],
    },
    "get_stock_prices": {
        "method": "GET",
        "path": "/prices/",
        "query_params": ["ticker", "start_date", "end_date", "interval", "limit"],
    },
    "get_stock_price_snapshot": {
        "method": "GET",
        "path": "/prices/snapshot/",
        "query_params": ["ticker"],
    },
    "get_company_facts": {
        "method": "GET",
        "path": "/company/facts/",
        "query_params": ["ticker"],
    },
    "get_insider_trades": {
        "method": "GET",
        "path": "/insider-trades/",
        "query_params": ["ticker", "limit"],
    },
    "get_news": {
        "method": "GET",
        "path": "/news/",
        "query_params": ["ticker", "limit"],
    },
    "get_analyst_estimates": {
        "method": "GET",
        "path": "/analyst-estimates/",
        "query_params": ["ticker", "period", "limit"],
    },
    "get_filings": {
        "method": "GET",
        "path": "/sec-filings/",
        "query_params": ["ticker", "form_type", "limit"],
    },
    "get_filing_items": {
        "method": "GET",
        "path": "/sec-filings/items/",
        "query_params": ["ticker", "form_type", "item", "limit"],
    },
    "screen_stocks": {
        "method": "POST",
        "path": "/financials/search/screener",
        "body_key": "filters",  # special: pass body as-is
    },
}

TOOLS = [
    {
        "name": "get_income_statement",
        "description": "Get income statement data (revenue, gross profit, net income, EPS). period: annual | quarterly | ttm.",
        "parameters": {
            "ticker": {"type": "string"},
            "period": {"type": "string", "description": "annual | quarterly | ttm (default: annual)"},
            "limit": {"type": "integer", "description": "Number of periods (default: 4)"},
        },
        "required": ["ticker"],
    },
    {
        "name": "get_balance_sheet",
        "description": "Get balance sheet data (assets, liabilities, equity, cash, debt). period: annual | quarterly.",
        "parameters": {
            "ticker": {"type": "string"},
            "period": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["ticker"],
    },
    {
        "name": "get_cash_flow_statement",
        "description": "Get cash flow statement (operating, investing, financing activities, free cash flow).",
        "parameters": {
            "ticker": {"type": "string"},
            "period": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["ticker"],
    },
    {
        "name": "get_financial_metrics",
        "description": "Get historical financial metrics: P/E ratio, EV/EBITDA, gross margin, ROE, debt/equity.",
        "parameters": {
            "ticker": {"type": "string"},
            "period": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["ticker"],
    },
    {
        "name": "get_financial_metrics_snapshot",
        "description": "Get latest financial metrics snapshot: market cap, P/E, P/B, EV/EBITDA, dividend yield.",
        "parameters": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    {
        "name": "get_stock_prices",
        "description": "Get historical OHLCV stock prices over a date range.",
        "parameters": {
            "ticker": {"type": "string"},
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            "interval": {"type": "string", "description": "day | week | month"},
            "limit": {"type": "integer"},
        },
        "required": ["ticker"],
    },
    {
        "name": "get_stock_price_snapshot",
        "description": "Get the latest stock price snapshot (current price, OHLC, volume).",
        "parameters": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    {
        "name": "get_company_facts",
        "description": "Get company details: name, sector, industry, employees, exchange, market cap, description.",
        "parameters": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    {
        "name": "get_insider_trades",
        "description": "Get recent insider trading transactions (purchases/sales by officers and directors).",
        "parameters": {
            "ticker": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["ticker"],
    },
    {
        "name": "get_news",
        "description": "Get recent news articles. Pass a ticker for company-specific news, or omit for broad market news.",
        "parameters": {
            "ticker": {"type": "string", "description": "Optional ticker for company-specific news"},
            "limit": {"type": "integer"},
        },
        "required": [],
    },
    {
        "name": "get_analyst_estimates",
        "description": "Get analyst consensus estimates for revenue and earnings (EPS) for upcoming periods.",
        "parameters": {
            "ticker": {"type": "string"},
            "period": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["ticker"],
    },
    {
        "name": "get_filings",
        "description": "Get a list of SEC filings for a company (10-K, 10-Q, 8-K).",
        "parameters": {
            "ticker": {"type": "string"},
            "form_type": {"type": "string", "description": "10-K | 10-Q | 8-K"},
            "limit": {"type": "integer"},
        },
        "required": ["ticker"],
    },
    {
        "name": "get_filing_items",
        "description": "Extract specific sections from SEC filings, e.g. Risk Factors (Item 1A) or MD&A (Item 7) from a 10-K.",
        "parameters": {
            "ticker": {"type": "string"},
            "form_type": {"type": "string"},
            "item": {"type": "string", "description": "e.g. '1A' for Risk Factors"},
            "limit": {"type": "integer"},
        },
        "required": ["ticker", "form_type", "item"],
    },
    {
        "name": "screen_stocks",
        "description": "Screen and filter stocks by financial metrics. Pass filters as a list of {field, operator, value} dicts.",
        "parameters": {
            "filters": {"type": "array", "description": "List of filter conditions"},
            "limit": {"type": "integer"},
        },
        "required": ["filters"],
    },
]


# ── Request schema ──────────────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    tool: str
    arguments: dict = {}


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "server": "financial_datasets",
        "port": PORT,
        "api_key_set": bool(os.environ.get("FINANCIAL_DATASETS_API_KEY", "")),
    }


@app.get("/tools")
def list_tools():
    return {"tools": TOOLS}


@app.post("/call")
async def call_tool(req: ToolCallRequest):
    tool = req.tool
    args = req.arguments
    logger.info(f"Tool call: {tool} args={list(args.keys())}")

    if tool not in TOOL_MAP:
        return {"result": None, "error": f"Unknown tool: {tool}"}

    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
    if not api_key:
        return {
            "result": None,
            "error": "FINANCIAL_DATASETS_API_KEY is not set. Set it in the .env file.",
        }

    spec = TOOL_MAP[tool]
    method = spec["method"]
    path = spec["path"]
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = UPSTREAM + path

            if method == "GET":
                # Only pass params that are actually provided (skip None/empty)
                query_params = {
                    k: v for k, v in args.items()
                    if k in spec.get("query_params", []) and v is not None
                }
                resp = await client.get(url, params=query_params, headers=headers)
            else:  # POST
                resp = await client.post(url, json=args, headers=headers)

            resp.raise_for_status()
            return {"result": resp.json(), "error": None}

    except httpx.HTTPStatusError as e:
        body = e.response.text[:500]
        logger.error(f"Upstream error {e.response.status_code}: {body}")
        return {"result": None, "error": f"Upstream HTTP {e.response.status_code}: {body}"}
    except Exception as e:
        logger.exception(f"Tool call failed: {tool}")
        return {"result": None, "error": str(e)}


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Financial Datasets proxy server on port {PORT}")
    if not os.environ.get("FINANCIAL_DATASETS_API_KEY", ""):
        logger.warning(
            "FINANCIAL_DATASETS_API_KEY is not set — tool calls will fail. "
            "Set FINANCIAL_DATASETS_API_KEY in .env"
        )
    uvicorn.run(app, host="127.0.0.1", port=PORT)
