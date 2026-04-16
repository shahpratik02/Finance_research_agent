"""
FastAPI route definitions.

Responsibilities:
- POST /research  — accept a user query, trigger the pipeline, return the final report.
- GET  /runs/{run_id} — retrieve stored artifacts for a past run.
- GET  /runs/{run_id}/sources — retrieve the normalized sources for a run.
- DELETE /runs/{run_id} — delete a run and all its artifacts.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db import delete_run, get_run, get_sources_for_run, get_claims_for_run, get_reviews_for_run, get_report_for_run
from app.orchestrator import run_pipeline, PipelineResult
from app.schemas import QueryInput

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    output_style: str = "memo"  # memo | brief | full


class ResearchResponse(BaseModel):
    run_id: str
    status: str
    markdown: str
    error: str | None = None


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    """Run the full research pipeline and return the markdown report."""
    logger.info(f"Received research request: {req.query!r}")
    query_input = QueryInput(query=req.query, output_style=req.output_style)
    result: PipelineResult = run_pipeline(query_input)
    return ResearchResponse(
        run_id=result.run_id,
        status=result.status,
        markdown=result.markdown,
        error=result.error,
    )


@router.get("/runs/{run_id}")
def get_run_details(run_id: str):
    """Return run metadata, claims, reviews, and report."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    claims = get_claims_for_run(run_id)
    reviews = get_reviews_for_run(run_id)
    report = get_report_for_run(run_id)
    return {
        "run": {"run_id": run.run_id, "query": run.query_text, "status": run.status, "created_at": str(run.created_at)},
        "claims": [{"claim_id": c.claim_id, "text": c.claim_text, "support_type": c.support_type} for c in claims],
        "reviews": [{"claim_id": r.claim_id, "verdict": r.verdict, "notes": r.notes} for r in reviews],
        "report": {"title": report.title, "markdown": report.report_markdown} if report else None,
    }


@router.get("/runs/{run_id}/sources")
def get_run_sources(run_id: str):
    """Return all normalized sources for a run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    sources = get_sources_for_run(run_id)
    return {
        "run_id": run_id,
        "sources": [
            {
                "source_id": s.source_id,
                "provider": s.provider,
                "tool": s.tool,
                "title": s.title,
                "uri": s.uri,
                "entity": s.entity,
                "content_summary": s.content_summary,
            }
            for s in sources
        ],
    }


@router.delete("/runs/{run_id}")
def delete_run_endpoint(run_id: str):
    """Delete a run and all its artifacts."""
    deleted = delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"deleted": run_id}
