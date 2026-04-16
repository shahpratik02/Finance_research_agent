"""
Researcher agent.

Responsibilities:
- Accept the planner output (and optionally a RetryInstruction for a second pass).
- Call the SGLang model (via sglang_client) with the researcher prompt and tool access.
- Invoke MCP tools (via mcp_client) to retrieve sources, staying within the tool budget.
- Normalize each raw tool response to a SourceRecord via source_normalizer.
- Return a structured ResearchResult (subquestion_answers, claims, gaps).

Settings used: temperature=0.0, max_tokens=2200, enable_thinking=False, tool_calls=True.
Tool budget: 20 calls on first pass, 6 calls on retry pass.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import settings
from app.mcp_client import call_tool, list_tools, ToolCallError, provider_for_tool
from app.schemas import (
    MCPProvider,
    PlannerOutput,
    QueryInput,
    ResearchResult,
    RetryInstruction,
    SourceRecord,
)
from app.sglang_client import RESEARCHER_PROFILE, chat
from app.source_normalizer import normalize

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "researcher.txt"

# ── Tool definitions for the LLM (OpenAI function-calling format) ──────────────

_TOOL_DEFS: list[dict[str, Any]] | None = None


def _get_tool_defs() -> list[dict[str, Any]]:
    """Build OpenAI-format tool definitions from the live MCP server manifests.
    Cached after first call."""
    global _TOOL_DEFS
    if _TOOL_DEFS is not None:
        return _TOOL_DEFS

    # Only expose the most useful tools to keep context usage low.
    # 23 tools overwhelms a small model with a 16K context window.
    _ALLOWED_TOOLS: set[str] = {
        # Yahoo Finance — fast market snapshots
        "get_ticker_info", "get_ticker_news", "get_price_history",
        # FRED — macro data
        "get_series", "search_series",
        # Financial Datasets — fundamentals
        "get_income_statement", "get_financial_metrics", "get_analyst_estimates",
        # Open Web Search
        "search",  # will be aliased to web_search below
    }

    defs: list[dict[str, Any]] = []
    for provider in ["yahoo_finance", "fred", "financial_datasets", "open_web_search"]:
        tools = list_tools(provider)
        for t in tools:
            name = t["name"]
            if name not in _ALLOWED_TOOLS:
                continue
            # Skip Yahoo's "search" — it overlaps with web_search and confuses the model
            if provider == "yahoo_finance" and name == "search":
                continue
            # Alias: the open_web_search server exposes "search" but we want the
            # LLM to call it as "web_search" to avoid ambiguity with Yahoo's "search".
            if provider == "open_web_search" and name == "search":
                name = "web_search"
            params = t.get("parameters", {})
            required = t.get("required", [])
            defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required,
                    },
                },
            })
    _TOOL_DEFS = defs
    return _TOOL_DEFS


def _tool_list_text() -> str:
    """Human-readable summary of available tools for the prompt."""
    lines: list[str] = []
    for td in _get_tool_defs():
        fn = td["function"]
        params = ", ".join(
            f"{k}: {v.get('type', 'string')}"
            for k, v in fn["parameters"].get("properties", {}).items()
        )
        lines.append(f"- {fn['name']}({params}) — {fn['description']}")
    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def research(
    query_input: QueryInput,
    plan: PlannerOutput,
    run_id: str,
    retry_instruction: RetryInstruction | None = None,
) -> tuple[ResearchResult, list[SourceRecord]]:
    """
    Run the researcher agent: call the LLM in a tool-calling loop, collect
    sources from MCP servers, and return structured claims.

    Args:
        query_input:        The user's original query.
        plan:               Planner output with subquestions and suggested tools.
        run_id:             Current run UUID (used for source ID generation).
        retry_instruction:  If present, this is a retry pass with focused scope.

    Returns:
        (ResearchResult, list[SourceRecord]) — the claims/gaps and all sources
        collected during this pass.
    """
    budget = (
        retry_instruction.remaining_tool_budget
        if retry_instruction
        else settings.researcher_tool_budget
    )
    tool_calls_made = 0
    sources: list[SourceRecord] = []
    source_summaries: list[str] = []  # for injecting into conversation

    # Build system prompt
    system_prompt = _build_prompt(query_input, plan, budget, retry_instruction, run_id)

    # Start conversation
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_input.query},
    ]

    tool_defs = _get_tool_defs()

    logger.info(
        f"[researcher] Starting {'retry' if retry_instruction else 'initial'} pass "
        f"budget={budget}"
    )

    # ── Tool-calling loop ──────────────────────────────────────────────────────
    # The LLM may request tool calls. We execute them, feed results back, and
    # repeat until the LLM returns a final text response (no more tool calls)
    # or budget is exhausted.
    max_rounds = budget + 3  # safety cap to avoid infinite loops
    for _round in range(max_rounds):
        # If budget is spent, do a final call without tools to force a response
        if tool_calls_made >= budget:
            logger.info(f"[researcher] Tool budget exhausted ({tool_calls_made}/{budget}), requesting final answer")
            result = chat(messages, profile=RESEARCHER_PROFILE, response_schema=ResearchResult)
            if result.parsed:
                return _finalize(result.parsed, sources, run_id)
            break

        result = chat(messages, profile=RESEARCHER_PROFILE, tools=tool_defs)

        # No tool calls → LLM is ready to give final answer
        if not result.tool_calls:
            # Try parsing the content as ResearchResult
            if result.parsed:
                return _finalize(result.parsed, sources, run_id)
            # Content present but not parsed — do one more call with schema enforcement
            if result.content:
                messages.append({"role": "assistant", "content": result.content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Now compile all the evidence you gathered into the required JSON output. "
                        "Include all claims with source_ids, subquestion answers, and gaps."
                    ),
                })
                final = chat(messages, profile=RESEARCHER_PROFILE, response_schema=ResearchResult)
                if final.parsed:
                    return _finalize(final.parsed, sources, run_id)
            break

        # ── Execute each tool call ─────────────────────────────────────────────
        # Build assistant message with tool_calls for the conversation
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": result.content or ""}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in result.tool_calls
        ]
        messages.append(assistant_msg)

        for tc in result.tool_calls:
            if tool_calls_made >= budget:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": "Tool budget exhausted"}),
                })
                continue

            tool_calls_made += 1
            logger.info(f"[researcher] Tool call {tool_calls_made}/{budget}: {tc.name}({tc.arguments})")

            try:
                raw = call_tool(tc.name, tc.arguments)
            except ToolCallError as e:
                logger.warning(f"[researcher] Tool call failed: {e}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": str(e)}),
                })
                continue

            # Normalize to a SourceRecord
            try:
                provider = provider_for_tool(tc.name)
            except ToolCallError:
                provider = "open_web_search"

            record = normalize(run_id, provider, tc.name, tc.arguments, raw)
            if record:
                sources.append(record)
                summary = (
                    f"[{record.source_id}] ({record.provider.value}/{record.tool}) "
                    f"{record.title}: {record.content_summary}"
                )
                source_summaries.append(summary)
                # Feed a concise version back to the LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({
                        "source_id": record.source_id,
                        "title": record.title,
                        "content_summary": record.content_summary,
                        "raw_excerpt": record.raw_excerpt[:500],
                    }),
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": "No usable data returned"}),
                })

    # ── Fallback: force a final structured response ────────────────────────────
    logger.info("[researcher] Forcing final structured output")
    messages.append({
        "role": "user",
        "content": (
            "Your tool budget is now spent. Based on all the evidence collected so far, "
            "produce the final JSON output with subquestion_answers, claims, and gaps. "
            f"Use run_id prefix '{run_id[:8]}' for claim_ids."
        ),
    })
    final = chat(messages, profile=RESEARCHER_PROFILE, response_schema=ResearchResult)
    if final.parsed:
        return _finalize(final.parsed, sources, run_id)

    # Absolute fallback: return empty result
    logger.error("[researcher] Could not get a valid ResearchResult from the model")
    return ResearchResult(subquestion_answers=[], claims=[], gaps=["Research failed to produce results"]), sources


# ── Helpers ────────────────────────────────────────────────────────────────────

def _finalize(
    result: ResearchResult,
    sources: list[SourceRecord],
    run_id: str,
) -> tuple[ResearchResult, list[SourceRecord]]:
    """Log and return the final result."""
    logger.info(
        f"[researcher] Done — claims={len(result.claims)} "
        f"gaps={len(result.gaps)} sources={len(sources)}"
    )
    return result, sources


def _build_prompt(
    query_input: QueryInput,
    plan: PlannerOutput,
    budget: int,
    retry_instruction: RetryInstruction | None,
    run_id: str,
) -> str:
    """Load the prompt template and fill in dynamic fields."""
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    plan_json = plan.model_dump_json(indent=2)

    retry_section = ""
    if retry_instruction:
        retry_section = (
            "## RETRY PASS\n\n"
            "This is a retry pass. Focus ONLY on these areas:\n"
            f"- Reason: {retry_instruction.retry_reason}\n"
            f"- Focus subquestions: {json.dumps(retry_instruction.focus_subquestions)}\n"
            f"- Unsupported claims to fix: {json.dumps([c.claim_text for c in retry_instruction.unsupported_claims])}\n"
            f"- Gaps to fill: {json.dumps(retry_instruction.gaps_to_fill)}\n"
            f"- Already retrieved source IDs (do NOT re-fetch): {json.dumps(retry_instruction.already_retrieved_source_ids)}\n"
            f"- Suggested tools: {json.dumps(retry_instruction.suggested_tools)}\n"
            "Do NOT redo work that already succeeded. Only address the gaps above."
        )

    return template.format(
        as_of=query_input.as_of.strftime("%Y-%m-%d %H:%M UTC"),
        plan=plan_json,
        tool_budget=budget,
        retry_section=retry_section,
        run_id_prefix=run_id[:8],
    )
