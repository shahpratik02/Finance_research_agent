"""
Source normalizer.

Converts raw MCP tool responses into uniform SourceRecord instances.
Called by the researcher and reviewer after every mcp_client.call_tool().

Entry point:
    record = normalize(run_id, provider, tool, arguments, raw_response)
    # returns SourceRecord, or None if the response was an error / empty

Source ID format:
    {run_id[:8]}-{provider_prefix}-{uuid4[:8]}
    e.g.  a3f9b12c-yhoo-d7e4c091

Provider prefix map:
    yahoo_finance      → yhoo
    fred               → fred
    financial_datasets → find
    open_web_search    → webs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.schemas import MCPProvider, SourceRecord

logger = logging.getLogger(__name__)

_PROVIDER_PREFIX: dict[str, str] = {
    MCPProvider.yahoo_finance:      "yhoo",
    MCPProvider.fred:               "fred",
    MCPProvider.financial_datasets: "find",
    MCPProvider.open_web_search:    "webs",
    MCPProvider.rag_document:       "ragd",
}


# ── Public entry point ─────────────────────────────────────────────────────────

def normalize(
    run_id:       str,
    provider:     MCPProvider | str,
    tool:         str,
    arguments:    dict[str, Any],
    raw_response: dict[str, Any],
) -> SourceRecord | None:
    """
    Convert a raw MCP tool response to a SourceRecord.

    Returns None (and logs a warning) if:
      - raw_response["error"] is set, or
      - raw_response["result"] is None or empty.

    Args:
        run_id:       The current run's UUID string.
        provider:     MCPProvider enum value or string key.
        tool:         Tool name as called (e.g. "get_ticker_info").
        arguments:    Arguments dict passed to the tool.
        raw_response: The dict returned by mcp_client.call_tool().

    Returns:
        A validated SourceRecord, or None on failure.
    """
    provider = MCPProvider(provider) if isinstance(provider, str) else provider

    # ── Guard: server-reported errors ─────────────────────────────────────────
    if raw_response.get("error"):
        logger.warning(
            f"[normalizer] {tool} returned error: {raw_response['error']!r} — skipping"
        )
        return None

    result = raw_response.get("result")
    if result is None or result == [] or result == {}:
        logger.warning(f"[normalizer] {tool} returned empty result — skipping")
        return None

    # ── Dispatch to per-provider extractor ─────────────────────────────────────
    try:
        fields = _extract(provider, tool, arguments, result)
    except Exception as e:
        logger.warning(f"[normalizer] Failed to extract fields for {tool}: {e}")
        return None

    if not fields:
        return None

    source_id = _make_source_id(run_id, provider)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    return SourceRecord(
        source_id=source_id,
        run_id=run_id,
        provider=provider,
        tool=tool,
        title=fields.get("title", tool),
        uri=fields.get("uri"),
        retrieved_at=now,
        published_at=fields.get("published_at"),
        entity=fields.get("entity"),
        content_summary=fields.get("content_summary", ""),
        raw_excerpt=fields.get("raw_excerpt", ""),
        structured_payload=result if isinstance(result, dict) else {"data": result},
    )


# ── Source ID generation ───────────────────────────────────────────────────────

def _make_source_id(run_id: str, provider: MCPProvider) -> str:
    prefix  = run_id[:8].replace("-", "")
    prov    = _PROVIDER_PREFIX[provider]
    uid     = uuid4().hex[:8]
    return f"{prefix}-{prov}-{uid}"


# ── Extraction dispatcher ──────────────────────────────────────────────────────

def _extract(
    provider:  MCPProvider,
    tool:      str,
    arguments: dict,
    result:    Any,
) -> dict | None:
    if provider == MCPProvider.yahoo_finance:
        return _yahoo(tool, arguments, result)
    if provider == MCPProvider.fred:
        return _fred(tool, arguments, result)
    if provider == MCPProvider.financial_datasets:
        return _findata(tool, arguments, result)
    if provider == MCPProvider.open_web_search:
        return _websearch(tool, arguments, result)
    return None


# ── Yahoo Finance extractors ───────────────────────────────────────────────────

def _yahoo_quote_time_utc(info: dict) -> str | None:
    """Format yfinance ``regularMarketTime`` (unix) for grounding claims."""
    ts = _g(info, "regularMarketTime")
    if ts is None:
        return None
    try:
        sec = float(ts)
        return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        return None


def _yahoo(tool: str, args: dict, result: Any) -> dict | None:
    symbol = args.get("symbol", "").upper()

    if tool == "get_ticker_info":
        info = result if isinstance(result, dict) else {}
        name   = _g(info, "longName") or _g(info, "shortName") or symbol
        sector = _g(info, "sector") or "N/A"
        price  = _g(info, "currentPrice") or _g(info, "regularMarketPrice")
        mcap   = _g(info, "marketCap")
        pe     = _g(info, "trailingPE")
        quote_utc = _yahoo_quote_time_utc(info)
        summary = (
            f"{name} ({sector}). "
            f"Price: {_fmt(price)}. "
            f"Market cap: {_fmt_big(mcap)}. "
            f"P/E: {_fmt(pe)}."
        )
        excerpt = (
            f"currentPrice={_fmt(price)}, marketCap={_fmt_big(mcap)}, "
            f"trailingPE={_fmt(pe)}, sector={sector}"
        )
        if quote_utc:
            summary = f"{summary} Quote snapshot: {quote_utc}."
            excerpt = f"{excerpt}, quote_snapshot_utc={quote_utc}"
        return {
            "entity":          symbol,
            "title":           f"{symbol} — Company Info",
            "content_summary": summary,
            "raw_excerpt":     excerpt,
        }

    if tool == "get_ticker_news":
        news = result if isinstance(result, list) else []
        headlines = "; ".join(
            _g(n, "title") or "" for n in news[:5] if _g(n, "title")
        )
        return {
            "entity":          symbol,
            "title":           f"{symbol} — Recent News",
            "content_summary": f"{len(news)} news articles for {symbol}.",
            "raw_excerpt":     headlines[:600],
        }

    if tool == "get_price_history":
        rows = _g(result, "data") or []
        period = _g(result, "period") or args.get("period", "")
        latest = rows[-1] if rows else {}
        return {
            "entity":          symbol,
            "title":           f"{symbol} — Price History ({period})",
            "content_summary": (
                f"{len(rows)} price points for {symbol}. "
                f"Latest close: {_fmt(_g(latest, 'close'))} on {_g(latest, 'date')}."
            ),
            "raw_excerpt": (
                "; ".join(
                    f"{r.get('date')}: close={_fmt(r.get('close'))}"
                    for r in rows[-5:]
                )
            ),
        }

    if tool == "search":
        quotes = _g(result, "quotes") or []
        news   = _g(result, "news") or []
        return {
            "entity":          args.get("query", ""),
            "title":           f"Yahoo Finance Search: {args.get('query', '')}",
            "content_summary": f"Found {len(quotes)} quotes and {len(news)} news items.",
            "raw_excerpt": (
                "; ".join(_g(q, "longname") or _g(q, "symbol") or "" for q in quotes[:5])
            ),
        }

    return None


# ── FRED extractors ────────────────────────────────────────────────────────────

def _fred(tool: str, args: dict, result: Any) -> dict | None:
    series_id = args.get("series_id", "").upper()

    if tool == "get_series":
        data   = _g(result, "data") or []
        latest = data[-1] if data else {}
        return {
            "entity":          series_id,
            "title":           f"FRED: {series_id}",
            "content_summary": (
                f"{len(data)} data points for {series_id}. "
                f"Latest: {_g(latest, 'value')} on {_g(latest, 'date')}."
            ),
            "raw_excerpt": (
                "; ".join(f"{d.get('date')}: {d.get('value')}" for d in data[-6:])
            ),
        }

    if tool == "search_series":
        records = result if isinstance(result, list) else []
        titles  = "; ".join(
            _g(r, "title") or _g(r, "id") or "" for r in records[:5]
        )
        return {
            "entity":          args.get("query", ""),
            "title":           f"FRED Series Search: {args.get('query', '')}",
            "content_summary": f"Found {len(records)} FRED series.",
            "raw_excerpt":     titles[:600],
        }

    if tool == "get_series_info":
        title = _g(result, "title") or series_id
        freq  = _g(result, "frequency") or ""
        units = _g(result, "units") or ""
        updated = _g(result, "last_updated") or ""
        return {
            "entity":          series_id,
            "title":           f"FRED: {title}",
            "content_summary": f"{title}. Frequency: {freq}. Units: {units}. Last updated: {updated}.",
            "raw_excerpt":     f"series_id={series_id}, frequency={freq}, units={units}, last_updated={updated}",
        }

    return None


# ── Financial Datasets extractors ──────────────────────────────────────────────

def _findata(tool: str, args: dict, result: Any) -> dict | None:
    ticker = args.get("ticker", "").upper()

    # Statement tools: income / balance / cash flow
    if tool == "get_income_statement":
        items = _first_list(result, ["income_statements"])
        latest = items[0] if items else {}
        rev = _g(latest, "revenue") or _g(latest, "total_revenue")
        ni  = _g(latest, "net_income")
        per = _g(latest, "period") or _g(latest, "fiscal_period")
        return {
            "entity":          ticker,
            "title":           f"{ticker} — Income Statement",
            "content_summary": f"{ticker} income statement. Period: {per}. Revenue: {_fmt_big(rev)}. Net income: {_fmt_big(ni)}.",
            "raw_excerpt":     f"period={per}, revenue={_fmt_big(rev)}, net_income={_fmt_big(ni)}",
        }

    if tool == "get_balance_sheet":
        items = _first_list(result, ["balance_sheets"])
        latest = items[0] if items else {}
        assets = _g(latest, "total_assets")
        equity = _g(latest, "total_equity") or _g(latest, "shareholders_equity")
        debt   = _g(latest, "total_debt") or _g(latest, "long_term_debt")
        return {
            "entity":          ticker,
            "title":           f"{ticker} — Balance Sheet",
            "content_summary": f"{ticker} balance sheet. Assets: {_fmt_big(assets)}. Equity: {_fmt_big(equity)}. Debt: {_fmt_big(debt)}.",
            "raw_excerpt":     f"total_assets={_fmt_big(assets)}, equity={_fmt_big(equity)}, debt={_fmt_big(debt)}",
        }

    if tool == "get_cash_flow_statement":
        items = _first_list(result, ["cash_flow_statements"])
        latest = items[0] if items else {}
        ocf = _g(latest, "operating_cash_flow")
        fcf = _g(latest, "free_cash_flow")
        return {
            "entity":          ticker,
            "title":           f"{ticker} — Cash Flow Statement",
            "content_summary": f"{ticker} cash flows. Operating CF: {_fmt_big(ocf)}. Free CF: {_fmt_big(fcf)}.",
            "raw_excerpt":     f"operating_cash_flow={_fmt_big(ocf)}, free_cash_flow={_fmt_big(fcf)}",
        }

    if tool in ("get_financial_metrics", "get_financial_metrics_snapshot"):
        if tool == "get_financial_metrics_snapshot":
            snap = _g(result, "snapshot") or (result if isinstance(result, dict) else {})
            items = [snap]
        else:
            items = _first_list(result, ["financial_metrics"])
        latest = items[0] if items else {}
        pe  = _g(latest, "pe_ratio") or _g(latest, "price_to_earnings_ratio")
        pb  = _g(latest, "pb_ratio") or _g(latest, "price_to_book_ratio")
        ev  = _g(latest, "ev_to_ebitda")
        gm  = _g(latest, "gross_margin")
        return {
            "entity":          ticker,
            "title":           f"{ticker} — Financial Metrics",
            "content_summary": f"{ticker} metrics. P/E: {_fmt(pe)}. P/B: {_fmt(pb)}. EV/EBITDA: {_fmt(ev)}. Gross margin: {_fmt(gm)}.",
            "raw_excerpt":     f"pe_ratio={_fmt(pe)}, pb_ratio={_fmt(pb)}, ev_to_ebitda={_fmt(ev)}, gross_margin={_fmt(gm)}",
        }

    if tool in ("get_stock_prices", "get_stock_price_snapshot"):
        if tool == "get_stock_price_snapshot":
            snap  = _g(result, "snapshot") or (result if isinstance(result, dict) else {})
            price = _g(snap, "price") or _g(snap, "close")
            return {
                "entity":          ticker,
                "title":           f"{ticker} — Price Snapshot",
                "content_summary": f"{ticker} latest price: {_fmt(price)}.",
                "raw_excerpt":     f"price={_fmt(price)}",
            }
        prices = _first_list(result, ["prices"])
        latest = prices[0] if prices else {}
        return {
            "entity":          ticker,
            "title":           f"{ticker} — Stock Prices",
            "content_summary": f"{len(prices)} price records for {ticker}. Latest close: {_fmt(_g(latest, 'close'))}.",
            "raw_excerpt":     "; ".join(
                f"{p.get('date')}: {_fmt(p.get('close'))}" for p in prices[:5]
            ),
        }

    if tool == "get_company_facts":
        facts = _g(result, "company_facts") or (result if isinstance(result, dict) else {})
        name  = _g(facts, "name") or ticker
        sect  = _g(facts, "sector") or ""
        desc  = (_g(facts, "description") or "")[:300]
        return {
            "entity":          ticker,
            "title":           f"{ticker} — Company Facts",
            "content_summary": f"{name}. Sector: {sect}. {desc}",
            "raw_excerpt":     f"name={name}, sector={sect}",
        }

    if tool == "get_news":
        items = _first_list(result, ["news"])
        headlines = "; ".join(_g(n, "title") or "" for n in items[:5])
        return {
            "entity":          ticker or "market",
            "title":           f"{ticker or 'Market'} — News",
            "content_summary": f"{len(items)} news items.",
            "raw_excerpt":     headlines[:600],
        }

    if tool == "get_analyst_estimates":
        items = _first_list(result, ["analyst_estimates"])
        latest = items[0] if items else {}
        rev_est = _g(latest, "estimated_revenue") or _g(latest, "revenue_estimate")
        eps_est = _g(latest, "estimated_eps") or _g(latest, "eps_estimate")
        return {
            "entity":          ticker,
            "title":           f"{ticker} — Analyst Estimates",
            "content_summary": f"{ticker} analyst consensus. Revenue est: {_fmt_big(rev_est)}. EPS est: {_fmt(eps_est)}.",
            "raw_excerpt":     f"estimated_revenue={_fmt_big(rev_est)}, estimated_eps={_fmt(eps_est)}",
        }

    if tool in ("get_filings", "get_filing_items"):
        items = _first_list(result, ["filings", "filing_items", "items"])
        return {
            "entity":          ticker,
            "title":           f"{ticker} — SEC Filings",
            "content_summary": f"{len(items)} filing records for {ticker}.",
            "raw_excerpt":     "; ".join(
                _g(f, "form_type") or _g(f, "type") or "" for f in items[:5]
            ),
        }

    if tool == "get_insider_trades":
        items = _first_list(result, ["insider_trades", "trades"])
        return {
            "entity":          ticker,
            "title":           f"{ticker} — Insider Trades",
            "content_summary": f"{len(items)} insider trade records for {ticker}.",
            "raw_excerpt":     "; ".join(
                f"{_g(t, 'name') or 'unknown'}: {_g(t, 'transaction_type') or ''} {_fmt_big(_g(t, 'value'))}"
                for t in items[:5]
            ),
        }

    if tool == "screen_stocks":
        items = _first_list(result, ["stocks", "results", "matches"])
        summaries = []
        for s in items[:8]:
            sym = _g(s, "ticker") or _g(s, "symbol") or "?"
            mcap = _g(s, "market_cap")
            summaries.append(f"{sym} (mkt cap {_fmt_big(mcap)})")
        return {
            "entity":          args.get("sector", args.get("exchange", "screen")),
            "title":           "Stock Screener Results",
            "content_summary": f"Screen returned {len(items)} stocks.",
            "raw_excerpt":     "; ".join(summaries),
        }

    return None


# ── Open Web Search extractors ─────────────────────────────────────────────────

def _websearch(tool: str, args: dict, result: Any) -> dict | None:
    if tool in ("search", "web_search", "search_web"):
        query   = _g(result, "query") or args.get("query", "")
        results = _g(result, "results") or []
        excerpt = "; ".join(
            f"{r.get('title', '')}: {r.get('snippet', '')[:120]}"
            for r in results[:4]
        )
        return {
            "entity":          query,
            "title":           f"Web Search: {query}",
            "uri":             None,
            "content_summary": f"{_g(result, 'count') or len(results)} results for \"{query}\".",
            "raw_excerpt":     excerpt[:600],
        }

    if tool == "fetch_web":
        url     = _g(result, "url") or args.get("url", "")
        content = _g(result, "content") or ""
        return {
            "entity":          url,
            "title":           f"Web Page: {url[:80]}",
            "uri":             url,
            "content_summary": f"Fetched {_g(result, 'chars') or len(content)} characters from {url}.",
            "raw_excerpt":     content[:600],
        }

    return None


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _g(obj: Any, key: str) -> Any:
    """Safe dict get — returns None for missing keys or non-dicts."""
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def _fmt(value: Any) -> str:
    """Format a numeric value to 2 decimal places, or return 'N/A'."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_big(value: Any) -> str:
    """Format large numbers with B/M/K suffixes."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.2f}K"
    return f"${v:.2f}"


def _first_list(result: Any, keys: list[str]) -> list:
    """Try each key in order; return the first non-empty list found."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for k in keys:
            val = result.get(k)
            if isinstance(val, list) and val:
                return val
    return []
