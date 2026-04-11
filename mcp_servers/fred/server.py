"""
FRED MCP Server — Port 8002

Wraps the mortada/fredapi Python library as a locally hosted HTTP tool server.
The main app's mcp_client.py calls this via POST /call.

Requires: FRED_API_KEY environment variable.
Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html

Tools:
    get_series       — fetch an economic time series by FRED series ID
    search_series    — full-text search for FRED series by keyword
    get_series_info  — metadata for a specific FRED series
"""

import logging
import os
from pathlib import Path
from typing import Any

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [fred] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="FRED MCP Server", version="1.0.0")

PORT = int(os.environ.get("FRED_MCP_PORT", 8002))

# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_series",
        "description": (
            "Fetch a FRED economic data series. Common series IDs: "
            "FEDFUNDS (Federal Funds Rate), UNRATE (Unemployment Rate), "
            "CPIAUCSL (CPI Inflation), GDP (US GDP), DGS10 (10-Year Treasury), "
            "VIXCLS (VIX), SP500 (S&P 500 daily). "
            "Returns the most recent data points as a dated time series."
        ),
        "parameters": {
            "series_id": {"type": "string", "description": "FRED series ID (e.g. FEDFUNDS, UNRATE, GDP)"},
            "limit": {"type": "integer", "description": "Max number of recent data points to return (default 24)"},
        },
        "required": ["series_id"],
    },
    {
        "name": "search_series",
        "description": (
            "Search FRED for economic data series by keyword. "
            "Returns series metadata including ID, title, frequency, units, and last_updated."
        ),
        "parameters": {
            "query": {"type": "string", "description": "Search text, e.g. 'unemployment rate' or 'federal funds'"},
            "limit": {"type": "integer", "description": "Max results to return (default 10)"},
        },
        "required": ["query"],
    },
    {
        "name": "get_series_info",
        "description": (
            "Get metadata for a specific FRED series: title, frequency, units, "
            "seasonal adjustment, last_updated date, and observation dates."
        ),
        "parameters": {
            "series_id": {"type": "string", "description": "FRED series ID"},
        },
        "required": ["series_id"],
    },
]


# ── Request schema ──────────────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    tool: str
    arguments: dict = {}


# ── FRED client factory ─────────────────────────────────────────────────────────

def get_fred_client():
    """Create a Fred client. Reads FRED_API_KEY fresh each call so restarts aren't needed."""
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY is not set. Set it in .env or as an environment variable. "
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    from fredapi import Fred
    return Fred(api_key=api_key)


def _serialize_value(v: Any) -> Any:
    """Convert pandas/numpy types to JSON-serializable Python types."""
    if hasattr(v, "isoformat"):  # datetime / Timestamp
        return v.isoformat()
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    return v


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "server": "fred",
        "port": PORT,
        "api_key_set": bool(os.environ.get("FRED_API_KEY", "")),
    }


@app.get("/tools")
def list_tools():
    return {"tools": TOOLS}


@app.post("/call")
def call_tool(req: ToolCallRequest):
    tool = req.tool
    args = req.arguments
    logger.info(f"Tool call: {tool} args={args}")

    try:
        if tool == "get_series":
            return _get_series(args)
        elif tool == "search_series":
            return _search_series(args)
        elif tool == "get_series_info":
            return _get_series_info(args)
        else:
            return {"result": None, "error": f"Unknown tool: {tool}"}
    except Exception as e:
        logger.exception(f"Tool call failed: {tool}")
        return {"result": None, "error": str(e)}


# ── Tool implementations ────────────────────────────────────────────────────────

def _get_series(args: dict) -> dict:
    series_id = args.get("series_id", "").strip().upper()
    limit = int(args.get("limit", 24))
    if not series_id:
        return {"result": None, "error": "series_id is required"}
    fred = get_fred_client()
    series = fred.get_series(series_id)
    data = [
        {
            "date": str(date.date()),
            "value": float(val) if val == val else None,  # NaN check
        }
        for date, val in series.tail(limit).items()
    ]
    return {
        "result": {
            "series_id": series_id,
            "count": len(data),
            "data": data,
        },
        "error": None,
    }


def _search_series(args: dict) -> dict:
    query = args.get("query", "").strip()
    limit = int(args.get("limit", 10))
    if not query:
        return {"result": None, "error": "query is required"}
    fred = get_fred_client()
    results_df = fred.search(query, limit=limit)
    if results_df is None or results_df.empty:
        return {"result": [], "error": None}
    records = []
    for _, row in results_df.iterrows():
        records.append({k: _serialize_value(v) for k, v in row.items()})
    return {"result": records, "error": None}


def _get_series_info(args: dict) -> dict:
    series_id = args.get("series_id", "").strip().upper()
    if not series_id:
        return {"result": None, "error": "series_id is required"}
    fred = get_fred_client()
    info = fred.get_series_info(series_id)
    result = {k: _serialize_value(v) for k, v in info.items()}
    return {"result": result, "error": None}


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting FRED MCP server on port {PORT}")
    if not os.environ.get("FRED_API_KEY", ""):
        logger.warning(
            "FRED_API_KEY is not set — tool calls will fail. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    uvicorn.run(app, host="127.0.0.1", port=PORT)
