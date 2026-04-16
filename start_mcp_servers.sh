#!/usr/bin/env bash
# start_mcp_servers.sh
#
# Starts all 4 MCP tool servers in the background.
# Run this before starting the main Finance Research Agent application.
#
# Usage:
#   ./start_mcp_servers.sh
#
# Logs are written to logs/mcp_servers/<server_name>.log
# PIDs are stored in logs/mcp_servers/<server_name>.pid

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
LOGS_DIR="$PROJECT_DIR/logs/mcp_servers"
mkdir -p "$LOGS_DIR"

# ── Load .env ──────────────────────────────────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
    echo "Loaded .env from $PROJECT_DIR"
else
    echo "Warning: .env not found. Servers that need API keys may fail."
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Starting Finance Research MCP Servers  ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Yahoo Finance — port 8001 ───────────────────────────────────────────────
echo "[1/4] Yahoo Finance MCP server → http://127.0.0.1:8001"
$PYTHON "$PROJECT_DIR/mcp_servers/yahoo_finance/server.py" \
    > "$LOGS_DIR/yahoo_finance.log" 2>&1 &
echo $! > "$LOGS_DIR/yahoo_finance.pid"
echo "      PID: $(cat "$LOGS_DIR/yahoo_finance.pid")"

# ── 2. FRED — port 8002 ────────────────────────────────────────────────────────
echo "[2/4] FRED MCP server        → http://127.0.0.1:8002"
$PYTHON "$PROJECT_DIR/mcp_servers/fred/server.py" \
    > "$LOGS_DIR/fred.log" 2>&1 &
echo $! > "$LOGS_DIR/fred.pid"
echo "      PID: $(cat "$LOGS_DIR/fred.pid")"

# ── 3. Financial Datasets — port 8003 ─────────────────────────────────────────
echo "[3/4] Financial Datasets MCP → http://127.0.0.1:8003"
$PYTHON "$PROJECT_DIR/mcp_servers/financial_datasets/server.py" \
    > "$LOGS_DIR/financial_datasets.log" 2>&1 &
echo $! > "$LOGS_DIR/financial_datasets.pid"
echo "      PID: $(cat "$LOGS_DIR/financial_datasets.pid")"

# ── 4. Open Web Search — port 8004 ────────────────────────────────────────────
echo "[4/4] Open Web Search (Python/DuckDuckGo) → http://127.0.0.1:8004"
$PYTHON "$PROJECT_DIR/mcp_servers/open_websearch/server.py" \
    > "$LOGS_DIR/open_websearch.log" 2>&1 &
echo $! > "$LOGS_DIR/open_websearch.pid"
echo "      PID: $(cat "$LOGS_DIR/open_websearch.pid")"

# ── Wait for servers to warm up ────────────────────────────────────────────────
echo ""
echo "Waiting 15 seconds for servers to start (Yahoo Finance loads slowly)..."
sleep 15

# ── Run health checks ──────────────────────────────────────────────────────────
echo ""
echo "Health checks:"

check_health() {
    local name=$1
    local url=$2
    local result
    result=$(curl -s --max-time 20 "$url" 2>/dev/null || echo "TIMEOUT")
    if echo "$result" | grep -q '"status".*"ok"'; then
        echo "  ✓ $name is UP"
    elif [ "$result" = "TIMEOUT" ]; then
        echo "  ✗ $name TIMEOUT — check logs/$name.log"
    else
        echo "  ✗ $name returned unexpected response — check logs/$name.log"
        echo "    Response: $result"
    fi
}

check_health "yahoo_finance (8001)" "http://127.0.0.1:8001/health"
check_health "fred (8002)"          "http://127.0.0.1:8002/health"
check_health "financial_datasets (8003)" "http://127.0.0.1:8003/health"
curl -s --max-time 5 --noproxy '*' "http://127.0.0.1:8004/health" > /dev/null 2>&1 \
    && echo "  ✓ open_websearch (8004) is UP" \
    || echo "  ✗ open_websearch (8004) not responding — check logs/open_websearch.log"

echo ""
echo "Logs: $LOGS_DIR"
echo "Run ./stop_mcp_servers.sh to shut down all servers."
echo ""
