"""
Report renderer.

Responsibilities:
- Accept a FinalReport structured object.
- Resolve reference_source_ids to full source metadata (title, URL, provider) via the db.
- Render the report as a markdown string following the required section order:
    1. Title + as-of timestamp
    2. Executive summary
    3. Main analysis sections
    4. Risks / caveats
    5. Unverified or conflicting items
    6. References

Returns a plain markdown string ready for display or storage.
"""

from __future__ import annotations

import logging

from app.db import resolve_sources
from app.models import Source
from app.schemas import FinalReport

logger = logging.getLogger(__name__)


def render(report: FinalReport) -> str:
    """
    Convert a FinalReport into a markdown string.

    Source ids in reference_source_ids are resolved from the database to
    produce a formatted reference list with titles, providers, and URIs.
    """
    source_lookup = _resolve_references(report.reference_source_ids)

    parts: list[str] = []

    # 1. Title + timestamp
    parts.append(f"# {report.title}")
    parts.append(f"*As of {report.as_of}*\n")

    # 2. Executive summary
    parts.append("## Executive Summary\n")
    for bullet in report.executive_summary:
        parts.append(f"- {bullet}")
    parts.append("")

    # 3. Main analysis sections
    for section in report.sections:
        parts.append(f"## {section.heading}\n")
        for para in section.paragraphs:
            parts.append(para)
            parts.append("")

    # 4. Unverified / caveats
    if report.unverified_items:
        parts.append("## Caveats & Unverified Items\n")
        for item in report.unverified_items:
            parts.append(f"- {item}")
        parts.append("")

    # 5. References
    if report.reference_source_ids:
        parts.append("## References\n")
        for i, sid in enumerate(report.reference_source_ids, 1):
            src = source_lookup.get(sid)
            if src:
                line = f"{i}. **{src.title}** ({src.provider})"
                if src.uri:
                    line += f" — {src.uri}"
            else:
                line = f"{i}. {sid} (source details not found)"
            parts.append(line)
        parts.append("")

    md = "\n".join(parts)
    logger.info(f"[renderer] Rendered {len(md)} chars of markdown")
    return md


def _resolve_references(source_ids: list[str]) -> dict[str, Source]:
    """Resolve source ids from the database into a lookup dict."""
    if not source_ids:
        return {}
    sources = resolve_sources(source_ids)
    return {s.source_id: s for s in sources}
