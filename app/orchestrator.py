"""
Pipeline orchestrator.

Responsibilities:
- Coordinate the full research pipeline for a single user query.
- Call: planner → researcher → reviewer → (optional retry) → formatter → renderer.
- Build the RetryInstruction when the Reviewer requests a retry.
- Save all artifacts (sources, claims, reviews, final report) to SQLite via db.py.
- Return the completed run artifact.

This module contains no LLM logic — it only wires the other modules together.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import settings
from app.db import (
    delete_run,
    init_db,
    insert_claim,
    insert_report,
    insert_review,
    insert_run,
    insert_source,
    list_all_run_ids,
    update_run_status,
)
from app.formatter import format_report
from app.models import Claim as ClaimRow
from app.models import Report as ReportRow
from app.models import Review as ReviewRow
from app.models import Run, RunStatus
from app.models import Source as SourceRow
from app.planner import plan
from app.renderer import render
from app.researcher import research
from app.reviewer import review
from app.trace import trace_phase, trace_run_start
from app.schemas import (
    ClaimReviewSet,
    FinalReport,
    PlannerOutput,
    QueryInput,
    ResearchResult,
    RetryInstruction,
    SourceRecord,
    UnsupportedClaimDetail,
)

logger = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """What run_pipeline returns to the caller."""
    run_id: str
    status: str
    markdown: str
    report: FinalReport | None = None
    error: str | None = None


# ── Public API ─────────────────────────────────────────────────────────────────

def run_pipeline(query_input: QueryInput) -> PipelineResult:
    """
    Execute the full research pipeline for one user query.

    Steps:
        1. Create run record
        2. Plan (subquestions + tools)
        3. Research (gather evidence, produce claims)
        4. Review (verify claims)
        5. Optional single retry if reviewer requests it
        6. Format (write report from approved claims only)
        7. Render (convert to markdown)
        8. Save all artifacts
        9. Return result

    On failure at any stage, the run is marked as failed and the error
    is returned in PipelineResult.error.
    """
    run_id = uuid4().hex
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Ensure database is ready
    init_db()

    # Clean up previous runs
    _cleanup_old_runs()

    # 1. Create run record
    run = Run(run_id=run_id, query_text=query_input.query, created_at=now)
    insert_run(run)
    logger.info(f"[orchestrator] Started run {run_id} — query={query_input.query!r}")
    trace_run_start(run_id, query_input.query)

    try:
        # 2. Plan
        logger.info("[orchestrator] Phase: PLANNER")
        trace_phase("PLANNER", f"query={query_input.query!r}")
        planner_output = plan(query_input)

        # 3. Research (initial pass)
        logger.info("[orchestrator] Phase: RESEARCHER (initial)")
        trace_phase("RESEARCHER", "initial pass")
        research_result, research_sources = research(
            query_input, planner_output, run_id,
        )
        all_sources: list[SourceRecord] = list(research_sources)

        # Save sources from initial pass
        _save_sources(run_id, research_sources)

        # 4. Review
        logger.info("[orchestrator] Phase: REVIEWER")
        trace_phase("REVIEWER", f"{len(research_result.claims)} claims to review")
        review_set, reviewer_sources = review(
            query_input, planner_output, research_result, all_sources, run_id,
        )
        all_sources.extend(reviewer_sources)
        _save_sources(run_id, reviewer_sources)

        # 5. Optional single retry
        retry_used = False
        if review_set.global_decision.needs_retry and not retry_used:
            logger.info("[orchestrator] Phase: RETRY")
            trace_phase("RETRY", f"{len(review_set.rejected())} rejected claims")
            retry_instruction = _build_retry_instruction(
                review_set, research_result, all_sources, planner_output,
            )
            retry_result, retry_sources = research(
                query_input, planner_output, run_id,
                retry_instruction=retry_instruction,
            )
            all_sources.extend(retry_sources)
            _save_sources(run_id, retry_sources)

            # Merge claims: keep original claims, add new ones from retry
            merged_result = _merge_research_results(research_result, retry_result)

            # Re-review all claims with full source set
            logger.info("[orchestrator] Phase: REVIEWER (post-retry)")
            trace_phase("REVIEWER", "post-retry re-review")
            review_set, post_retry_sources = review(
                query_input, planner_output, merged_result, all_sources, run_id,
            )
            all_sources.extend(post_retry_sources)
            _save_sources(run_id, post_retry_sources)

            research_result = merged_result
            retry_used = True

        # Save claims
        _save_claims(run_id, research_result)

        # Save reviews
        _save_reviews(run_id, review_set)

        # 6. Format — only approved claims go to the formatter
        logger.info("[orchestrator] Phase: FORMATTER")
        approved = review_set.approved()
        rejected = review_set.rejected()
        trace_phase("FORMATTER", f"{len(approved)} approved, {len(rejected)} rejected")

        final_report = format_report(
            query_input=query_input,
            approved_reviews=approved,
            claims=research_result.claims,
            sources=all_sources,
            rejected_reviews=rejected,
            gaps=research_result.gaps,
        )

        # 7. Render
        logger.info("[orchestrator] Phase: RENDERER")
        trace_phase("RENDERER")
        markdown = render(final_report)

        # 8. Save report
        report_row = ReportRow(
            report_id=uuid4().hex,
            run_id=run_id,
            title=final_report.title,
            report_markdown=markdown,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        insert_report(report_row)

        # 8b. Save markdown file to reports/
        _save_report_file(run_id, query_input.query, markdown)

        # 9. Mark run as completed
        update_run_status(run_id, RunStatus.COMPLETED)
        logger.info(f"[orchestrator] Run {run_id} completed successfully")

        return PipelineResult(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            markdown=markdown,
            report=final_report,
        )

    except Exception as e:
        logger.exception(f"[orchestrator] Run {run_id} failed: {e}")
        update_run_status(run_id, RunStatus.FAILED)
        return PipelineResult(
            run_id=run_id,
            status=RunStatus.FAILED,
            markdown="",
            error=str(e),
        )


# ── RetryInstruction builder ──────────────────────────────────────────────────

def _build_retry_instruction(
    review_set: ClaimReviewSet,
    research_result: ResearchResult,
    all_sources: list[SourceRecord],
    planner_output: PlannerOutput,
) -> RetryInstruction:
    """Build a RetryInstruction from the reviewer's rejection details."""
    rejected = review_set.rejected()
    claim_map = {c.claim_id: c for c in research_result.claims}

    unsupported_details = []
    for rev in rejected:
        original = claim_map.get(rev.claim_id)
        unsupported_details.append(UnsupportedClaimDetail(
            claim_id=rev.claim_id,
            claim_text=original.text if original else rev.claim_id,
            rejection_reason=rev.notes,
        ))

    # Focus subquestions come from the reviewer's global decision
    focus_subs = review_set.global_decision.retry_focus_subquestions
    if not focus_subs:
        # Fallback: use all planner subquestions
        focus_subs = planner_output.subquestions

    return RetryInstruction(
        retry_reason=(
            f"{len(rejected)} claim(s) were unsupported or contradicted. "
            "Additional evidence is needed."
        ),
        focus_subquestions=focus_subs,
        unsupported_claims=unsupported_details,
        gaps_to_fill=research_result.gaps,
        already_retrieved_source_ids=[s.source_id for s in all_sources],
        suggested_tools=planner_output.suggested_tools,
        remaining_tool_budget=settings.retry_tool_budget,
    )


# ── Merge helper ──────────────────────────────────────────────────────────────

def _merge_research_results(
    original: ResearchResult,
    retry: ResearchResult,
) -> ResearchResult:
    """Combine the original and retry research results, deduplicating by claim_id."""
    existing_ids = {c.claim_id for c in original.claims}
    new_claims = [c for c in retry.claims if c.claim_id not in existing_ids]

    existing_subs = {sa.subquestion for sa in original.subquestion_answers}
    new_subs = [sa for sa in retry.subquestion_answers if sa.subquestion not in existing_subs]

    # Gaps: keep original gaps, add any new ones from retry that aren't already listed
    all_gaps = list(original.gaps)
    for g in retry.gaps:
        if g not in all_gaps:
            all_gaps.append(g)

    return ResearchResult(
        subquestion_answers=original.subquestion_answers + new_subs,
        claims=original.claims + new_claims,
        gaps=all_gaps,
    )


# ── Cleanup ─────────────────────────────────────────────────────────────────

def _cleanup_old_runs() -> None:
    """Delete all previous runs before starting a new one."""
    run_ids = list_all_run_ids()
    if not run_ids:
        return
    logger.info(f"[orchestrator] Cleaning up {len(run_ids)} old run(s)")
    for rid in run_ids:
        delete_run(rid)


# ── DB persistence helpers ────────────────────────────────────────────────────

def _save_sources(run_id: str, sources: list[SourceRecord]) -> None:
    """Convert SourceRecords to db rows and insert."""
    for s in sources:
        row = SourceRow(
            source_id=s.source_id,
            run_id=run_id,
            provider=s.provider.value,
            tool=s.tool,
            title=s.title,
            content_summary=s.content_summary,
            raw_excerpt=s.raw_excerpt,
            structured_payload_json=json.dumps(s.structured_payload, default=str),
            retrieved_at=s.retrieved_at,
            uri=s.uri,
            published_at=s.published_at,
            entity=s.entity,
        )
        insert_source(row)


def _save_claims(run_id: str, research_result: ResearchResult) -> None:
    """Convert schema Claims to db rows and insert."""
    for c in research_result.claims:
        row = ClaimRow(
            claim_id=c.claim_id,
            run_id=run_id,
            claim_text=c.text,
            source_ids_json=json.dumps(c.source_ids),
            support_type=c.support_type.value,
        )
        insert_claim(row)


def _save_reviews(run_id: str, review_set: ClaimReviewSet) -> None:
    """Convert ClaimReviews to db rows and insert."""
    for rev in review_set.claim_reviews:
        row = ReviewRow(
            review_id=uuid4().hex,
            run_id=run_id,
            claim_id=rev.claim_id,
            verdict=rev.verdict.value,
            notes=rev.notes,
            final_source_ids_json=json.dumps(rev.final_source_ids),
        )
        insert_review(row)


# ── File-based report storage ─────────────────────────────────────────────────

_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _save_report_file(run_id: str, query: str, markdown: str) -> None:
    """Write the rendered markdown to reports/<run_id>.md."""
    _REPORTS_DIR.mkdir(exist_ok=True)
    path = _REPORTS_DIR / f"{run_id}.md"
    path.write_text(markdown, encoding="utf-8")
    logger.info(f"[orchestrator] Report saved to {path}")
