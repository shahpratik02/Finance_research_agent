#!/usr/bin/env python3
"""
test_mcp_servers.py

Smoke-tests all 4 MCP tool servers.
Run from the project root (servers must be running first):

    ./start_mcp_servers.sh
    python3 test_mcp_servers.py

Each test hits a real endpoint with a real tool call and prints a one-line verdict.
"""

import json
import sys
import urllib.request
import urllib.error

BASE = {
    "yahoo_finance":      "http://127.0.0.1:8001",
    "fred":               "http://127.0.0.1:8002",
    "financial_datasets": "http://127.0.0.1:8003",
    "open_websearch":     "http://127.0.0.1:8004",
}

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

results = []


# ── helpers ────────────────────────────────────────────────────────────────────

def get(url, timeout=15):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post(url, payload, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check(label, base_url, tool, arguments, expect_fn, skip_if_no_key=False):
    """Run one tool call and evaluate the result."""
    url = f"{base_url}/call"
    try:
        resp = post(url, {"tool": tool, "arguments": arguments})
        error = resp.get("error")
        result = resp.get("result")

        if skip_if_no_key and error and "API_KEY" in error:
            print(f"  {WARN}  [{label}] SKIPPED — no API key set")
            results.append(("skip", label))
            return

        if error:
            print(f"  {FAIL}  [{label}] ERROR — {error[:120]}")
            results.append(("fail", label))
            return

        verdict = expect_fn(result)
        if verdict is True:
            print(f"  {PASS}  [{label}] PASSED")
            results.append(("pass", label))
        else:
            print(f"  {FAIL}  [{label}] UNEXPECTED — {verdict}")
            results.append(("fail", label))

    except urllib.error.URLError as e:
        print(f"  {FAIL}  [{label}] UNREACHABLE — {e.reason}")
        results.append(("fail", label))
    except Exception as e:
        print(f"  {FAIL}  [{label}] EXCEPTION — {e}")
        results.append(("fail", label))


# ── 1. Health checks ───────────────────────────────────────────────────────────

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  1. Health Checks")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

for name, base in BASE.items():
    try:
        r = get(f"{base}/health")
        if r.get("status") == "ok":
            key_note = ""
            if "api_key_set" in r:
                key_note = f" (api_key={'✓' if r['api_key_set'] else '✗ not set'})"
            print(f"  {PASS}  {name} is UP on {base}{key_note}")
            results.append(("pass", f"health:{name}"))
        else:
            print(f"  {FAIL}  {name} unexpected response: {r}")
            results.append(("fail", f"health:{name}"))
    except Exception as e:
        print(f"  {FAIL}  {name} UNREACHABLE — {e}")
        results.append(("fail", f"health:{name}"))


# ── 2. Tool listings ───────────────────────────────────────────────────────────

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  2. Tool Listings  (GET /tools)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

for name, base in BASE.items():
    try:
        r = get(f"{base}/tools")
        tools = r.get("tools", [])
        names = [t["name"] for t in tools]
        print(f"  {PASS}  {name}: {len(tools)} tools → {names}")
        results.append(("pass", f"tools:{name}"))
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        results.append(("fail", f"tools:{name}"))


# ── 3. Yahoo Finance tool calls ────────────────────────────────────────────────

base = BASE["yahoo_finance"]
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  3. Yahoo Finance  (port 8001)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

check(
    "get_ticker_info: AAPL", base,
    "get_ticker_info", {"symbol": "AAPL"},
    lambda r: True if isinstance(r, dict) and r.get("longName") else f"missing longName, got keys: {list(r.keys())[:5]}"
)

check(
    "get_price_history: MSFT 1mo", base,
    "get_price_history", {"symbol": "MSFT", "period": "1mo", "interval": "1d"},
    lambda r: True if isinstance(r, dict) and len(r.get("data", [])) > 5 else f"expected >5 rows, got {len(r.get('data',[]))}"
)

check(
    "get_ticker_news: NVDA", base,
    "get_ticker_news", {"symbol": "NVDA", "count": 3},
    lambda r: True if isinstance(r, list) and len(r) > 0 else f"expected list with items, got {type(r)}"
)

check(
    "search: Tesla", base,
    "search", {"query": "Tesla stock", "search_type": "all"},
    # quotes can be empty for some queries — just check the call succeeds and returns a dict
    lambda r: True if isinstance(r, dict) else f"expected dict, got {type(r)}"
)


# ── 4. FRED tool calls ─────────────────────────────────────────────────────────

base = BASE["fred"]
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  4. FRED  (port 8002)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

check(
    "get_series: FEDFUNDS (last 6 months)", base,
    "get_series", {"series_id": "FEDFUNDS", "limit": 6},
    lambda r: True if isinstance(r, dict) and len(r.get("data", [])) > 0 else f"no data returned: {r}",
    skip_if_no_key=True,
)

check(
    "search_series: inflation", base,
    "search_series", {"query": "inflation", "limit": 5},
    lambda r: True if isinstance(r, list) and len(r) > 0 else f"no results: {r}",
    skip_if_no_key=True,
)

check(
    "get_series_info: UNRATE", base,
    "get_series_info", {"series_id": "UNRATE"},
    lambda r: True if isinstance(r, dict) and r.get("id") == "UNRATE" else f"expected id=UNRATE, got: {r}",
    skip_if_no_key=True,
)


# ── 5. Financial Datasets tool calls ───────────────────────────────────────────

base = BASE["financial_datasets"]
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  5. Financial Datasets  (port 8003)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

check(
    "get_stock_price_snapshot: NVDA", base,
    "get_stock_price_snapshot", {"ticker": "NVDA"},
    lambda r: True if isinstance(r, dict) and "snapshot" in r else f"unexpected response shape: {list(r.keys()) if isinstance(r,dict) else r}",
    skip_if_no_key=True,
)

check(
    "get_company_facts: AAPL", base,
    "get_company_facts", {"ticker": "AAPL"},
    lambda r: True if isinstance(r, dict) and "company_facts" in r else f"unexpected shape: {list(r.keys()) if isinstance(r,dict) else r}",
    skip_if_no_key=True,
)

check(
    "get_income_statement: MSFT annual 2", base,
    "get_income_statement", {"ticker": "MSFT", "period": "annual", "limit": 2},
    lambda r: True if isinstance(r, dict) and "income_statements" in r else f"unexpected shape: {list(r.keys()) if isinstance(r,dict) else r}",
    skip_if_no_key=True,
)

check(
    "get_news: TSLA", base,
    "get_news", {"ticker": "TSLA", "limit": 3},
    lambda r: True if isinstance(r, dict) and "news" in r else f"unexpected shape: {list(r.keys()) if isinstance(r,dict) else r}",
    skip_if_no_key=True,
)


# ── 6. Open Web Search tool calls ─────────────────────────────────────────────

base = BASE["open_websearch"]
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  6. Open Web Search  (port 8004)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

check(
    "search: Fed interest rate decision 2025", base,
    "search", {"query": "Fed interest rate decision 2025", "limit": 5},
    lambda r: True if isinstance(r, dict) and r.get("count", 0) > 0 else f"0 results returned — check network"
)

check(
    "search: S&P 500 outlook", base,
    "search", {"query": "S&P 500 market outlook", "limit": 3},
    lambda r: True if isinstance(r, dict) and r.get("count", 0) > 0 else f"0 results"
)

check(
    "fetch_web: Yahoo Finance page", base,
    "fetch_web", {"url": "https://finance.yahoo.com/topic/stock-market-news/", "max_chars": 2000},
    lambda r: True if isinstance(r, dict) and r.get("chars", 0) > 100 else f"too short: {r.get('chars')} chars"
)


# ── Summary ────────────────────────────────────────────────────────────────────

passed = sum(1 for s, _ in results if s == "pass")
failed = sum(1 for s, _ in results if s == "fail")
skipped = sum(1 for s, _ in results if s == "skip")
total = len(results)

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Results: {passed} passed  {failed} failed  {skipped} skipped  ({total} total)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if skipped:
    print(f"\n  {WARN}  {skipped} test(s) skipped — add API keys to .env and re-run:")
    print("       FRED_API_KEY=...                (free at fred.stlouisfed.org)")
    print("       FINANCIAL_DATASETS_API_KEY=...  (from financialdatasets.ai)")

if failed:
    print(f"\n  {FAIL}  {failed} test(s) failed — check logs/mcp_servers/<server>.log")
    sys.exit(1)
else:
    print(f"\n  All non-skipped tests passed! Servers are ready.")
