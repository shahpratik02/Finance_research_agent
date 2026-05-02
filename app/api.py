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
import threading
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from markdown_it import MarkdownIt
from pydantic import BaseModel

from app.db import delete_run, get_run, get_sources_for_run, get_claims_for_run, get_reviews_for_run, get_report_for_run
from app.orchestrator import run_pipeline, PipelineResult
from app.schemas import QueryInput

logger = logging.getLogger(__name__)

router = APIRouter()
_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent
_DASHBOARD_HTML = _APP_DIR / "ui" / "dashboard.html"
_REPORTS_DIR = _ROOT / "reports"
_UI_LOCK = threading.Lock()
_UI_JOBS: dict[str, dict] = {}
_MAX_JOB_LOGS = 800
_MD = MarkdownIt("commonmark", {"html": False})


# ── Request / Response models ──────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    output_style: str = "memo"  # memo | brief | full
    documents: list[str] | None = None  # optional raw texts — triggers RAG before external research
    documents_folder: str | None = None  # optional directory path — text files indexed for RAG


class ResearchResponse(BaseModel):
    run_id: str
    status: str
    markdown: str
    error: str | None = None


class UIRunRequest(BaseModel):
    query: str
    output_style: str = "memo"
    documents_folder: str | None = None


class UIJobStartResponse(BaseModel):
    job_id: str
    status: str


# ── UI run tracking ────────────────────────────────────────────────────────────

def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _append_log(job_id: str, line: str) -> None:
    with _UI_LOCK:
        job = _UI_JOBS.get(job_id)
        if not job:
            return
        logs = job.setdefault("logs", [])
        logs.append(line)
        if len(logs) > _MAX_JOB_LOGS:
            job["logs"] = logs[-_MAX_JOB_LOGS:]


class _ThreadLogHandler(logging.Handler):
    """Capture logs emitted from the worker thread handling one UI job."""

    def __init__(self, job_id: str, thread_id: int) -> None:
        super().__init__()
        self.job_id = job_id
        self.thread_id = thread_id
        self.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        try:
            msg = self.format(record)
        except Exception:
            msg = str(record.getMessage())
        _append_log(self.job_id, msg)


def _run_ui_job(job_id: str, req: UIRunRequest) -> None:
    thread_id = threading.get_ident()
    handler = _ThreadLogHandler(job_id, thread_id)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        with _UI_LOCK:
            job = _UI_JOBS[job_id]
            job["status"] = "running"
            job["started_at"] = _iso_now()
        _append_log(job_id, f"{_iso_now()} [ui] INFO Started UI job {job_id}")

        result = run_pipeline(
            QueryInput(
                query=req.query,
                output_style=req.output_style,
                documents_folder=req.documents_folder or None,
            )
        )
        with _UI_LOCK:
            job = _UI_JOBS[job_id]
            job["status"] = "completed" if result.status == "completed" else "failed"
            job["completed_at"] = _iso_now()
            try:
                markdown_html = _MD.render(result.markdown or "")
            except Exception:
                markdown_html = f"<pre>{escape(result.markdown or '')}</pre>"
            job["result"] = {
                "run_id": result.run_id,
                "status": result.status,
                "markdown": result.markdown,
                "markdown_html": markdown_html,
                "error": result.error,
            }
    except Exception as e:
        logger.exception("[ui] Background run failed")
        with _UI_LOCK:
            job = _UI_JOBS[job_id]
            job["status"] = "failed"
            job["completed_at"] = _iso_now()
            job["result"] = {
                "run_id": None,
                "status": "failed",
                "markdown": "",
                "markdown_html": "<p>No markdown</p>",
                "error": str(e),
            }
    finally:
        root_logger.removeHandler(handler)


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/healthz")
def healthz():
    """Basic app health endpoint."""
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
def demo_page():
    """Built-in dashboard UI for running and monitoring pipeline jobs."""
    if not _DASHBOARD_HTML.is_file():
        return HTMLResponse(
            content="<p>Dashboard template missing: app/ui/dashboard.html</p>",
            status_code=500,
            headers={"Cache-Control": "no-store, max-age=0"},
        )
    return HTMLResponse(
        content=_DASHBOARD_HTML.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    """Run the full research pipeline and return the markdown report."""
    logger.info(f"Received research request: {req.query!r}")
    query_input = QueryInput(
        query=req.query,
        output_style=req.output_style,
        documents=req.documents,
        documents_folder=req.documents_folder,
    )
    result: PipelineResult = run_pipeline(query_input)
    return ResearchResponse(
        run_id=result.run_id,
        status=result.status,
        markdown=result.markdown,
        error=result.error,
    )


@router.post("/ui/jobs", response_model=UIJobStartResponse)
def start_ui_job(req: UIRunRequest):
    """Queue a background pipeline run for the dashboard UI."""
    job_id = uuid4().hex[:12]
    with _UI_LOCK:
        _UI_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": _iso_now(),
            "started_at": None,
            "completed_at": None,
            "request": req.model_dump(),
            "logs": [],
            "result": None,
        }
    t = threading.Thread(target=_run_ui_job, args=(job_id, req), daemon=True)
    t.start()
    return UIJobStartResponse(job_id=job_id, status="queued")


@router.get("/ui/jobs/{job_id}")
def get_ui_job(job_id: str):
    """Get live status + captured logs for a dashboard job."""
    with _UI_LOCK:
        job = _UI_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="UI job not found")
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "logs": list(job.get("logs", [])),
            "result": job.get("result"),
        }


@router.get("/ui/reports")
def list_report_markdown_files():
    """List markdown report files for quick UI preview."""
    if not _REPORTS_DIR.exists():
        return {"reports": []}
    rows: list[dict] = []
    for path in sorted(_REPORTS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        run_id = path.stem
        rows.append({
            "run_id": run_id,
            "filename": path.name,
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        })
    return {"reports": rows}


@router.get("/ui/reports/{run_id}")
def get_report_markdown(run_id: str):
    """Read markdown file content for preview panel."""
    path = _REPORTS_DIR / f"{run_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report markdown not found")
    markdown = path.read_text(encoding="utf-8")
    try:
        markdown_html = _MD.render(markdown)
    except Exception:
        markdown_html = f"<pre>{escape(markdown)}</pre>"
    return {"run_id": run_id, "markdown": markdown, "markdown_html": markdown_html}


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
