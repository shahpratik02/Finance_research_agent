"""
Reviewer agent.

Responsibilities:
- Accept the ResearchResult and all normalized sources.
- Call the SGLang model (via sglang_client) with the reviewer prompt.
- Optionally invoke MCP tools (via mcp_client) for limited targeted re-checking.
- Return a structured ClaimReviewSet (verdicts per claim, global_decision with retry flag).

This agent does NOT write the final report. It only produces claim verdicts.

Settings used: temperature=0.0, max_tokens=2400, enable_thinking=True, tool_calls=True.
Tool budget: 6 calls maximum. Thinking trace is stripped before passing output downstream.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import settings
from app.mcp_client import call_tool, ToolCallError, provider_for_tool
from app.researcher import _get_tool_defs  # reuse the same cached tool defs
from app.schemas import (
    ClaimReviewSet,
    PlannerOutput,
    QueryInput,
    ResearchResult,
    SourceRecord,
)
from app.sglang_client import REVIEWER_PROFILE, chat
from app.source_normalizer import normalize

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "reviewer.txt"


# ── Public API ─────────────────────────────────────────────────────────────────

def review(
    query_input: QueryInput,
    plan: PlannerOutput,
    research_result: ResearchResult,
    sources: list[SourceRecord],
    run_id: str,
) -> tuple[ClaimReviewSet, list[SourceRecord]]:
    """
    Verify researcher claims against their cited sources.

    Args:
        query_input:     The user's original query.
        plan:            Planner output.
        research_result: The researcher's claims, subquestion answers, and gaps.
        sources:         All SourceRecords collected during research.
        run_id:          Current run UUID.

    Returns:
        (ClaimReviewSet, new_sources) — verdicts for every claim, plus any
        new sources collected during re-checking.
    """
    budget = settings.reviewer_tool_budget
    tool_calls_made = 0
    new_sources: list[SourceRecord] = []

    system_prompt = _build_prompt(query_input, plan, research_result, sources, budget)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Review every claim and produce the ClaimReviewSet JSON."},
    ]

    tool_defs = _get_tool_defs()

    logger.info(
        f"[reviewer] Starting review — claims={len(research_result.claims)} "
        f"sources={len(sources)} budget={budget}"
    )

    # ── Tool-calling loop (limited) ────────────────────────────────────────────
    max_rounds = budget + 3
    for _round in range(max_rounds):
        # Budget spent → force final answer without tools
        if tool_calls_made >= budget:
            logger.info(f"[reviewer] Tool budget exhausted ({tool_calls_made}/{budget})")
            result = chat(messages, profile=REVIEWER_PROFILE, response_schema=ClaimReviewSet)
            if result.parsed:
                return _finalize(result.parsed, new_sources)
            break

        result = chat(messages, profile=REVIEWER_PROFILE, tools=tool_defs)

        # No tool calls → LLM is done
        if not result.tool_calls:
            if result.parsed:
                return _finalize(result.parsed, new_sources)
            # Content but not parsed — force one schema-constrained call
            if result.content:
                messages.append({"role": "assistant", "content": result.content})
                messages.append({
                    "role": "user",
                    "content": "Now output the final ClaimReviewSet JSON with all verdicts.",
                })
                final = chat(messages, profile=REVIEWER_PROFILE, response_schema=ClaimReviewSet)
                if final.parsed:
                    return _finalize(final.parsed, new_sources)
            break

        # ── Execute tool calls ─────────────────────────────────────────────────
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
            logger.info(f"[reviewer] Recheck {tool_calls_made}/{budget}: {tc.name}({tc.arguments})")

            try:
                raw = call_tool(tc.name, tc.arguments)
            except ToolCallError as e:
                logger.warning(f"[reviewer] Tool call failed: {e}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": str(e)}),
                })
                continue

            try:
                provider = provider_for_tool(tc.name)
            except ToolCallError:
                provider = "open_web_search"

            record = normalize(run_id, provider, tc.name, tc.arguments, raw)
            if record:
                new_sources.append(record)
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

    # ── Fallback ───────────────────────────────────────────────────────────────
    logger.info("[reviewer] Forcing final structured output")
    messages.append({
        "role": "user",
        "content": "Produce the final ClaimReviewSet JSON now with all claim verdicts and the global decision.",
    })
    final = chat(messages, profile=REVIEWER_PROFILE, response_schema=ClaimReviewSet)
    if final.parsed:
        return _finalize(final.parsed, new_sources)

    # Absolute fallback: mark everything as unsupported — never auto-verify
    logger.error("[reviewer] Could not get valid ClaimReviewSet — marking all claims unsupported")
    from app.schemas import ClaimReview, ClaimVerdict, GlobalDecision
    fallback_reviews = [
        ClaimReview(
            claim_id=c.claim_id,
            verdict=ClaimVerdict.unsupported,
            notes="Reviewer failed to produce structured output; claim unverified",
            final_source_ids=[],
        )
        for c in research_result.claims
    ]
    return ClaimReviewSet(
        claim_reviews=fallback_reviews,
        global_decision=GlobalDecision(
            needs_retry=True,
            retry_focus_subquestions=[sq.subquestion for sq in research_result.subquestion_answers],
            unsupported_claim_ids=[c.claim_id for c in research_result.claims],
        ),
    ), new_sources


# ── Helpers ────────────────────────────────────────────────────────────────────

def _finalize(
    review_set: ClaimReviewSet,
    new_sources: list[SourceRecord],
) -> tuple[ClaimReviewSet, list[SourceRecord]]:
    approved = len(review_set.approved())
    rejected = len(review_set.rejected())
    logger.info(
        f"[reviewer] Done — approved={approved} rejected={rejected} "
        f"needs_retry={review_set.global_decision.needs_retry} "
        f"new_sources={len(new_sources)}"
    )
    return review_set, new_sources


def _build_prompt(
    query_input: QueryInput,
    plan: PlannerOutput,
    research_result: ResearchResult,
    sources: list[SourceRecord],
    budget: int,
) -> str:
    """Load the reviewer prompt template and fill in dynamic fields."""
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    # Format sources as a readable list
    source_lines: list[str] = []
    for s in sources:
        source_lines.append(
            f"- [{s.source_id}] provider={s.provider.value} tool={s.tool}\n"
            f"  title: {s.title}\n"
            f"  summary: {s.content_summary}\n"
            f"  excerpt: {s.raw_excerpt[:400]}"
        )
    sources_text = "\n\n".join(source_lines) if source_lines else "(no sources)"

    # Format claims
    claims_lines: list[str] = []
    for c in research_result.claims:
        claims_lines.append(
            f"- [{c.claim_id}] support={c.support_type.value} sources={c.source_ids}\n"
            f"  \"{c.text}\""
        )
    claims_text = "\n".join(claims_lines) if claims_lines else "(no claims)"

    # Format gaps
    gaps_text = "\n".join(f"- {g}" for g in research_result.gaps) if research_result.gaps else "(none)"

    return template.format(
        as_of=query_input.as_of.strftime("%Y-%m-%d %H:%M UTC"),
        query=query_input.query,
        sources=sources_text,
        claims=claims_text,
        gaps=gaps_text,
    )
