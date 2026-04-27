"""
Formatter agent.

Responsibilities:
- Accept only the verified and partially_verified claims (filtered by orchestrator).
- Accept the resolved source references for those claims.
- Call the LLM module (via llm_client) with the formatter prompt to produce report sections.
- Return a structured FinalReport (title, executive_summary, sections, unverified_items, reference_source_ids).

This agent has no MCP tool access. It performs no verification.
It only writes the report from the approved claims it receives.

Settings used: temperature=0.0, max_tokens=2000, enable_thinking=False, tool_calls=False.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.schemas import (
    Claim,
    ClaimReview,
    ClaimVerdict,
    FinalReport,
    OutputStyle,
    QueryInput,
    SourceRecord,
)
from app.llm_client import FORMATTER_PROFILE, chat

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "formatter.txt"


# ── Public API ─────────────────────────────────────────────────────────────────

def format_report(
    query_input: QueryInput,
    approved_reviews: list[ClaimReview],
    claims: list[Claim],
    sources: list[SourceRecord],
    rejected_reviews: list[ClaimReview],
    gaps: list[str],
) -> FinalReport:
    """
    Write the final report from approved claims only.

    Args:
        query_input:       The user's original query.
        approved_reviews:  ClaimReviews with verdict verified or partially_verified.
        claims:            Original Claim objects from the researcher (for text lookup).
        sources:           All SourceRecords (for resolving source context).
        rejected_reviews:  ClaimReviews that were unsupported/contradicted (for caveats).
        gaps:              Gaps from the researcher (for caveats).

    Returns:
        FinalReport with title, sections, executive_summary, references.

    Raises:
        ValueError: If the model cannot produce a valid FinalReport.
    """
    if not approved_reviews:
        logger.warning("[formatter] No approved claims — producing minimal report")
        return _empty_report(query_input, gaps, rejected_reviews, claims)

    system_prompt = _build_prompt(
        query_input, approved_reviews, claims, sources, rejected_reviews, gaps,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Write the {query_input.output_style.value} report."},
    ]

    logger.info(
        f"[formatter] Formatting report — approved={len(approved_reviews)} "
        f"style={query_input.output_style.value}"
    )

    # No tools — single structured call
    result = chat(messages, profile=FORMATTER_PROFILE, response_schema=FinalReport)

    if result.parsed is None:
        raise ValueError(
            f"Formatter returned no structured output.\nRaw content: {result.content}"
        )

    report: FinalReport = result.parsed
    logger.info(
        f"[formatter] Done — title={report.title!r} "
        f"sections={len(report.sections)} refs={len(report.reference_source_ids)}"
    )
    return report


# ── Helpers ────────────────────────────────────────────────────────────────────

def _empty_report(
    query_input: QueryInput,
    gaps: list[str],
    rejected_reviews: list[ClaimReview],
    claims: list[Claim],
) -> FinalReport:
    """Produce a minimal report when no claims were approved."""
    _ = claims  # same signature as full formatter path; body uses reviewer notes only
    unverified = [
        "Rejected assertions are listed by id only — do not treat quoted figures in "
        "those claims as facts."
    ]
    for r in rejected_reviews:
        unverified.append(f"{r.claim_id} ({r.verdict.value}): {r.notes}")
    unverified.extend(gaps)

    from app.schemas import ReportSection
    return FinalReport(
        title=f"Research report: {query_input.query[:80]}",
        as_of=query_input.as_of.strftime("%Y-%m-%d %H:%M UTC"),
        output_style=query_input.output_style,
        executive_summary=[
            "Insufficient verified evidence was found to produce a substantive report.",
            "All claims were either unsupported or contradicted by the available sources.",
        ],
        sections=[
            ReportSection(
                heading="Findings",
                paragraphs=[
                    "The research process could not produce claims that passed "
                    "independent verification. No substantive conclusions can be drawn."
                ],
            )
        ],
        unverified_items=unverified,
        reference_source_ids=[],
    )


def _build_prompt(
    query_input: QueryInput,
    approved_reviews: list[ClaimReview],
    claims: list[Claim],
    sources: list[SourceRecord],
    rejected_reviews: list[ClaimReview],
    gaps: list[str],
) -> str:
    """Load the formatter prompt template and fill in dynamic fields."""
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    claim_map = {c.claim_id: c for c in claims}
    source_map = {s.source_id: s for s in sources}

    # Build approved claims text with source context
    approved_lines: list[str] = []
    for rev in approved_reviews:
        original = claim_map.get(rev.claim_id)
        if not original:
            continue
        verdict_label = "VERIFIED" if rev.verdict == ClaimVerdict.verified else "PARTIALLY VERIFIED"
        src_details = []
        for sid in rev.final_source_ids:
            src = source_map.get(sid)
            if src:
                src_details.append(f"    [{sid}] {src.title} ({src.provider.value})")
        sources_block = "\n".join(src_details) if src_details else "    (no source details)"
        approved_lines.append(
            f"- [{rev.claim_id}] ({verdict_label})\n"
            f"  \"{original.text}\"\n"
            f"  Sources:\n{sources_block}"
        )
    approved_text = "\n\n".join(approved_lines) if approved_lines else "(none)"

    # Build unverified items text
    unverified_lines: list[str] = []
    for rev in rejected_reviews:
        original = claim_map.get(rev.claim_id)
        text = original.text if original else rev.claim_id
        unverified_lines.append(f"- {text} (reviewer: {rev.notes})")
    for g in gaps:
        unverified_lines.append(f"- {g}")
    unverified_text = "\n".join(unverified_lines) if unverified_lines else "(none)"

    as_of_str = query_input.as_of.strftime("%Y-%m-%d %H:%M UTC")

    return template.format(
        query=query_input.query,
        as_of=as_of_str,
        output_style=query_input.output_style.value,
        approved_claims=approved_text,
        unverified_items=unverified_text,
    )
