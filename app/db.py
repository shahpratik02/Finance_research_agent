"""
SQLite database layer.

All database access goes through this module. Uses Python's built-in
sqlite3 — no ORM. Every function opens and closes its own connection so
the module is safe to call from anywhere without managing connection state.

Usage:
    from app.db import init_db, insert_run, get_run, ...
    init_db()   # call once at startup
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from app.config import settings
from app.models import Claim, Report, Review, Run, Source

logger = logging.getLogger(__name__)

# ISO-8601 format used for all datetime columns.
_DT_FMT = "%Y-%m-%dT%H:%M:%S"


# ── Connection helper ──────────────────────────────────────────────────────────

@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    """Yield an auto-committing SQLite connection, then close it."""
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(settings.sqlite_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Schema creation ────────────────────────────────────────────────────────────

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'running'
);
"""

_CREATE_SOURCES = """
CREATE TABLE IF NOT EXISTS sources (
    source_id               TEXT PRIMARY KEY,
    run_id                  TEXT NOT NULL,
    provider                TEXT NOT NULL,
    tool                    TEXT NOT NULL,
    title                   TEXT NOT NULL,
    content_summary         TEXT NOT NULL,
    raw_excerpt             TEXT NOT NULL,
    structured_payload_json TEXT NOT NULL DEFAULT '{}',
    retrieved_at            TEXT NOT NULL,
    uri                     TEXT,
    published_at            TEXT,
    entity                  TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""

_CREATE_CLAIMS = """
CREATE TABLE IF NOT EXISTS claims (
    claim_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    claim_text      TEXT NOT NULL,
    source_ids_json TEXT NOT NULL DEFAULT '[]',
    support_type    TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""

_CREATE_REVIEWS = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id             TEXT PRIMARY KEY,
    run_id                TEXT NOT NULL,
    claim_id              TEXT NOT NULL,
    verdict               TEXT NOT NULL,
    notes                 TEXT NOT NULL,
    final_source_ids_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (run_id)   REFERENCES runs(run_id),
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
);
"""

_CREATE_REPORTS = """
CREATE TABLE IF NOT EXISTS reports (
    report_id       TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    report_markdown TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


def init_db() -> None:
    """Create all tables if they do not already exist. Safe to call on every startup."""
    with _conn() as con:
        con.execute(_CREATE_RUNS)
        con.execute(_CREATE_SOURCES)
        con.execute(_CREATE_CLAIMS)
        con.execute(_CREATE_REVIEWS)
        con.execute(_CREATE_REPORTS)
    logger.info(f"[db] Initialised database at {settings.sqlite_path}")


# ── Datetime helpers ───────────────────────────────────────────────────────────

def _dt_str(dt: datetime | None) -> str | None:
    return dt.strftime(_DT_FMT) if dt else None


def _parse_dt(s: str | None) -> datetime | None:
    return datetime.strptime(s, _DT_FMT) if s else None


# ── runs ───────────────────────────────────────────────────────────────────────

def insert_run(run: Run) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO runs (run_id, query_text, created_at, status) VALUES (?,?,?,?)",
            (run.run_id, run.query_text, _dt_str(run.created_at), run.status),
        )
    logger.debug(f"[db] Inserted run {run.run_id!r}")


def update_run_status(run_id: str, status: str) -> None:
    with _conn() as con:
        con.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))
    logger.debug(f"[db] Run {run_id!r} → status={status!r}")


def get_run(run_id: str) -> Run | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        return None
    return Run(
        run_id=row["run_id"],
        query_text=row["query_text"],
        created_at=_parse_dt(row["created_at"]),
        status=row["status"],
    )


# ── sources ────────────────────────────────────────────────────────────────────

def insert_source(src: Source) -> None:
    with _conn() as con:
        con.execute(
            """INSERT OR IGNORE INTO sources
               (source_id, run_id, provider, tool, title, content_summary,
                raw_excerpt, structured_payload_json, retrieved_at, uri,
                published_at, entity)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                src.source_id, src.run_id, src.provider, src.tool, src.title,
                src.content_summary, src.raw_excerpt, src.structured_payload_json,
                _dt_str(src.retrieved_at), src.uri,
                _dt_str(src.published_at), src.entity,
            ),
        )
    logger.debug(f"[db] Inserted source {src.source_id!r}")


def get_sources_for_run(run_id: str) -> list[Source]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM sources WHERE run_id=? ORDER BY retrieved_at", (run_id,)
        ).fetchall()
    return [
        Source(
            source_id=r["source_id"], run_id=r["run_id"],
            provider=r["provider"], tool=r["tool"], title=r["title"],
            content_summary=r["content_summary"], raw_excerpt=r["raw_excerpt"],
            structured_payload_json=r["structured_payload_json"],
            retrieved_at=_parse_dt(r["retrieved_at"]),
            uri=r["uri"], published_at=_parse_dt(r["published_at"]),
            entity=r["entity"],
        )
        for r in rows
    ]


def get_source(source_id: str) -> Source | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM sources WHERE source_id=?", (source_id,)
        ).fetchone()
    if row is None:
        return None
    return Source(
        source_id=row["source_id"], run_id=row["run_id"],
        provider=row["provider"], tool=row["tool"], title=row["title"],
        content_summary=row["content_summary"], raw_excerpt=row["raw_excerpt"],
        structured_payload_json=row["structured_payload_json"],
        retrieved_at=_parse_dt(row["retrieved_at"]),
        uri=row["uri"], published_at=_parse_dt(row["published_at"]),
        entity=row["entity"],
    )


# ── claims ─────────────────────────────────────────────────────────────────────

def insert_claim(claim: Claim) -> None:
    with _conn() as con:
        con.execute(
            """INSERT OR IGNORE INTO claims
               (claim_id, run_id, claim_text, source_ids_json, support_type)
               VALUES (?,?,?,?,?)""",
            (claim.claim_id, claim.run_id, claim.claim_text,
             claim.source_ids_json, claim.support_type),
        )
    logger.debug(f"[db] Inserted claim {claim.claim_id!r}")


def get_claims_for_run(run_id: str) -> list[Claim]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM claims WHERE run_id=?", (run_id,)
        ).fetchall()
    return [
        Claim(
            claim_id=r["claim_id"], run_id=r["run_id"],
            claim_text=r["claim_text"], source_ids_json=r["source_ids_json"],
            support_type=r["support_type"],
        )
        for r in rows
    ]


# ── reviews ────────────────────────────────────────────────────────────────────

def insert_review(review: Review) -> None:
    with _conn() as con:
        con.execute(
            """INSERT OR IGNORE INTO reviews
               (review_id, run_id, claim_id, verdict, notes, final_source_ids_json)
               VALUES (?,?,?,?,?,?)""",
            (review.review_id, review.run_id, review.claim_id,
             review.verdict, review.notes, review.final_source_ids_json),
        )
    logger.debug(f"[db] Inserted review {review.review_id!r}")


def get_reviews_for_run(run_id: str) -> list[Review]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM reviews WHERE run_id=?", (run_id,)
        ).fetchall()
    return [
        Review(
            review_id=r["review_id"], run_id=r["run_id"],
            claim_id=r["claim_id"], verdict=r["verdict"],
            notes=r["notes"], final_source_ids_json=r["final_source_ids_json"],
        )
        for r in rows
    ]


# ── reports ────────────────────────────────────────────────────────────────────

def insert_report(report: Report) -> None:
    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO reports
               (report_id, run_id, title, report_markdown, created_at)
               VALUES (?,?,?,?,?)""",
            (report.report_id, report.run_id, report.title,
             report.report_markdown, _dt_str(report.created_at)),
        )
    logger.debug(f"[db] Inserted report {report.report_id!r}")


def get_report_for_run(run_id: str) -> Report | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM reports WHERE run_id=?", (run_id,)
        ).fetchone()
    if row is None:
        return None
    return Report(
        report_id=row["report_id"], run_id=row["run_id"],
        title=row["title"], report_markdown=row["report_markdown"],
        created_at=_parse_dt(row["created_at"]),
    )


# ── Convenience: resolve source ids → metadata for the renderer ───────────────

def resolve_sources(source_ids: list[str]) -> list[Source]:
    """Fetch multiple sources by id in one query. Preserves input order."""
    if not source_ids:
        return []
    placeholders = ",".join("?" * len(source_ids))
    with _conn() as con:
        rows = con.execute(
            f"SELECT * FROM sources WHERE source_id IN ({placeholders})",
            source_ids,
        ).fetchall()
    by_id = {
        r["source_id"]: Source(
            source_id=r["source_id"], run_id=r["run_id"],
            provider=r["provider"], tool=r["tool"], title=r["title"],
            content_summary=r["content_summary"], raw_excerpt=r["raw_excerpt"],
            structured_payload_json=r["structured_payload_json"],
            retrieved_at=_parse_dt(r["retrieved_at"]),
            uri=r["uri"], published_at=_parse_dt(r["published_at"]),
            entity=r["entity"],
        )
        for r in rows
    }
    return [by_id[sid] for sid in source_ids if sid in by_id]


# ── Run deletion ───────────────────────────────────────────────────────────────

def list_all_run_ids() -> list[str]:
    """Return all run_ids ordered by created_at (oldest first)."""
    with _conn() as con:
        rows = con.execute("SELECT run_id FROM runs ORDER BY created_at").fetchall()
    return [r["run_id"] for r in rows]


def delete_run(run_id: str) -> bool:
    """Delete a run and all its associated artifacts. Returns True if the run existed."""
    with _conn() as con:
        existing = con.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not existing:
            return False
        con.execute("DELETE FROM reports WHERE run_id=?", (run_id,))
        con.execute("DELETE FROM reviews WHERE run_id=?", (run_id,))
        con.execute("DELETE FROM claims  WHERE run_id=?", (run_id,))
        con.execute("DELETE FROM sources WHERE run_id=?", (run_id,))
        con.execute("DELETE FROM runs    WHERE run_id=?", (run_id,))
    logger.info(f"[db] Deleted run {run_id!r} and all artifacts")
    return True
