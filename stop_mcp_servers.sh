#!/usr/bin/env bash
# stop_mcp_servers.sh
#
# Stops all 4 MCP tool servers that were started by start_mcp_servers.sh.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="$PROJECT_DIR/logs/mcp_servers"

echo "Stopping MCP servers..."

stop_server() {
    local name=$1
    local pid_file="$LOGS_DIR/$name.pid"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "  ✓ Stopped $name (PID $pid)"
        else
            echo "  - $name was not running (PID $pid)"
        fi
        rm -f "$pid_file"
    else
        echo "  - No PID file for $name (may not have been started)"
    fi
}

stop_server "yahoo_finance"
stop_server "fred"
stop_server "financial_datasets"
stop_server "open_websearch"

# Fallback: kill any orphaned processes by script name (catches stale PID file cases)
pkill -f "mcp_servers/(fred|financial_datasets|yahoo_finance|open_websearch)/server.py" 2>/dev/null || true

echo ""
echo "All MCP servers stopped."
