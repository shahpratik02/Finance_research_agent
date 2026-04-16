"""
Debug trace logger.

When DEBUG_TRACE=true in .env, writes detailed logs to logs/debug_trace.log:
- Full LLM prompts (messages sent)
- Full LLM responses (content, tool_calls, parsed)
- MCP tool call inputs and outputs

Disabled by default. Enable for development/debugging only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import settings

# ── Dedicated file logger ──────────────────────────────────────────────────────

_trace_logger = logging.getLogger("debug_trace")
_trace_logger.propagate = False  # don't send to root logger / stdout

if settings.debug_trace:
    _logs_dir = Path(__file__).resolve().parent.parent / "logs"
    _logs_dir.mkdir(exist_ok=True)
    _handler = logging.FileHandler(_logs_dir / "debug_trace.log", mode="a", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _trace_logger.addHandler(_handler)
    _trace_logger.setLevel(logging.DEBUG)
else:
    _trace_logger.addHandler(logging.NullHandler())
    _trace_logger.setLevel(logging.CRITICAL)


def _json(obj: object, max_len: int = 5000) -> str:
    """Compact JSON serialization with length cap."""
    try:
        s = json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + f"\n... [truncated at {max_len} chars]"
    return s


# ── Public trace functions ─────────────────────────────────────────────────────

def trace_llm_request(
    phase: str,
    messages: list[dict],
    schema_name: str | None = None,
    tools: list[dict] | None = None,
) -> None:
    """Log the full messages list being sent to the LLM."""
    if not settings.debug_trace:
        return
    tool_names = [t["function"]["name"] for t in tools] if tools else None
    _trace_logger.debug(
        f"\n{'='*60}\n"
        f"LLM REQUEST — phase={phase}\n"
        f"  schema={schema_name}  tools={tool_names}\n"
        f"  messages ({len(messages)}):\n{_json(messages)}\n"
    )


def trace_llm_response(
    phase: str,
    content: str | None,
    tool_calls: list | None = None,
    parsed: object | None = None,
) -> None:
    """Log the LLM response."""
    if not settings.debug_trace:
        return
    tc_summary = None
    if tool_calls:
        tc_summary = [{"name": tc.name, "args": tc.arguments} for tc in tool_calls]
    _trace_logger.debug(
        f"\nLLM RESPONSE — phase={phase}\n"
        f"  content ({len(content or '')} chars):\n{_json(content, max_len=3000)}\n"
        f"  tool_calls: {_json(tc_summary)}\n"
        f"  parsed: {'yes' if parsed else 'no'}\n"
    )


def trace_tool_call(tool: str, arguments: dict, provider: str) -> None:
    """Log an outgoing MCP tool call."""
    if not settings.debug_trace:
        return
    _trace_logger.debug(
        f"\nMCP TOOL CALL\n"
        f"  tool={tool}  provider={provider}\n"
        f"  arguments: {_json(arguments, max_len=1000)}\n"
    )


def trace_tool_result(tool: str, result: dict) -> None:
    """Log the raw MCP tool response."""
    if not settings.debug_trace:
        return
    _trace_logger.debug(
        f"\nMCP TOOL RESULT\n"
        f"  tool={tool}\n"
        f"  response: {_json(result, max_len=3000)}\n"
    )


def trace_phase(phase: str, message: str = "") -> None:
    """Mark a pipeline phase transition."""
    if not settings.debug_trace:
        return
    _trace_logger.debug(
        f"\n{'#'*60}\n"
        f"PHASE: {phase}  {message}\n"
        f"{'#'*60}\n"
    )


def trace_run_start(run_id: str, query: str) -> None:
    """Write a clear delimiter at the start of a new pipeline run."""
    if not settings.debug_trace:
        return
    _trace_logger.debug(
        f"\n\n{'='*70}\n"
        f"NEW RUN  run_id={run_id}\n"
        f"  query: {query}\n"
        f"{'='*70}\n"
    )
