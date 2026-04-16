"""
Database table definitions as Python dataclasses.

One dataclass per SQLite table. These are plain data containers used by
db.py for insert/query operations. They mirror the five tables exactly.

Do not add business logic here — that belongs in the agent modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


class RunStatus:
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class Run:
    """One row in the `runs` table — one per user query."""
    run_id:     str
    query_text: str
    created_at: datetime
    status:     str = RunStatus.RUNNING


@dataclass
class Source:
    """One row in the `sources` table — one per normalised MCP result."""
    source_id:               str
    run_id:                  str
    provider:                str
    tool:                    str
    title:                   str
    content_summary:         str
    raw_excerpt:             str
    structured_payload_json: str          # JSON string
    retrieved_at:            datetime
    uri:                     str | None = None
    published_at:            datetime | None = None
    entity:                  str | None = None


@dataclass
class Claim:
    """One row in the `claims` table — one Researcher claim."""
    claim_id:        str
    run_id:          str
    claim_text:      str
    source_ids_json: str   # JSON array string, e.g. '["src_001", "src_002"]'
    support_type:    str


@dataclass
class Review:
    """One row in the `reviews` table — one Reviewer verdict."""
    review_id:            str
    run_id:               str
    claim_id:             str
    verdict:              str
    notes:                str
    final_source_ids_json: str   # JSON array string


@dataclass
class Report:
    """One row in the `reports` table — the final markdown report."""
    report_id:       str
    run_id:          str
    title:           str
    report_markdown: str
    created_at:      datetime
