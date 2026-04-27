"""
Load RAG corpus from inline strings and/or a folder of text and PDF files.

Produces ordered (text, file_metadata) pairs for chunking. Metadata is carried
through to the vector store and SourceRecord payloads for downstream LLMs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from app.config import settings

logger = logging.getLogger(__name__)

# Repository root (parent of ``app/``) — relative ``documents_folder`` values resolve here.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_documents_folder(folder_str: str) -> Path:
    """
    Resolve and validate a user-supplied folder path.

    - **Relative** paths (e.g. ``documents``, ``./memos``) are resolved against the
      project root, not the process current working directory.
    - **Absolute** paths are tried first. If that path does not exist but looks like
      a mistaken root-relative single segment (e.g. ``/documents`` meaning a folder
      named ``documents`` next to ``app/``), ``<project>/documents`` is tried.
    """
    raw = str(folder_str).strip()
    expanded = Path(raw).expanduser()
    candidates: list[Path] = []

    if expanded.is_absolute():
        first = expanded.resolve()
        candidates.append(first)
        parts = first.parts
        # POSIX only: ``/documents`` is usually ``<project>/documents``, not filesystem root.
        if len(parts) == 2 and parts[0] == "/":
            under_project = (_PROJECT_ROOT / parts[1]).resolve()
            if under_project != first:
                candidates.append(under_project)
    else:
        candidates.append((_PROJECT_ROOT / expanded).resolve())

    seen: set[Path] = set()
    ordered: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for c in ordered:
        if c.is_dir():
            if c != ordered[0]:
                logger.info(
                    f"[rag_corpus] documents_folder {raw!r} not found at {ordered[0]}, "
                    f"using project-relative path {c}"
                )
            return c

    tried = ", ".join(str(x) for x in ordered)
    raise ValueError(
        f"documents_folder does not exist or is not a directory: {raw!r} (tried: {tried}). "
        f"Use a path relative to the project (e.g. documents) or a full absolute path."
    )


def _extract_pdf_text(data: bytes) -> tuple[str, int]:
    """Return (plain text, page_count). Empty string if parsing fails."""
    try:
        reader = PdfReader(BytesIO(data))
        pages = reader.pages
        parts: list[str] = []
        for page in pages:
            parts.append(page.extract_text() or "")
        text = "\n\n".join(parts).strip()
        return text, len(pages)
    except Exception as e:
        logger.warning(f"[rag_corpus] PDF parse failed: {e}")
        return "", 0


def load_folder_files(folder: Path) -> list[tuple[str, dict[str, Any]]]:
    """
    Read all supported text and PDF files under ``folder`` (recursive).

    Returns (file_text, metadata) where metadata includes paths and mtime
    for vector DB filtering and LLM context.
    """
    exts = settings.rag_folder_extensions
    max_bytes = settings.rag_max_file_bytes
    out: list[tuple[str, dict[str, Any]]] = []

    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file():
            continue
        suf = file_path.suffix.lower()
        if suf not in exts:
            continue
        try:
            stat = file_path.stat()
        except OSError as e:
            logger.warning(f"[rag_corpus] Skip unreadable file {file_path}: {e}")
            continue
        if stat.st_size > max_bytes:
            logger.warning(
                f"[rag_corpus] Skip file over RAG_MAX_FILE_BYTES ({max_bytes}): {file_path}"
            )
            continue

        try:
            raw = file_path.read_bytes()
        except OSError as e:
            logger.warning(f"[rag_corpus] Skip unreadable file {file_path}: {e}")
            continue

        page_count = 0
        if suf == ".pdf":
            text, page_count = _extract_pdf_text(raw)
        else:
            text = raw.decode("utf-8", errors="replace").strip()

        if not text:
            continue
        try:
            rel = str(file_path.resolve().relative_to(folder.resolve()))
        except ValueError:
            rel = file_path.name
        mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        meta: dict[str, Any] = {
            "filename": file_path.name,
            "source_relpath": rel.replace("\\", "/"),
            "file_ext": suf,
            "folder_root": str(folder.resolve()),
            "file_mtime_iso": mtime_iso,
            "file_size_bytes": stat.st_size,
            "page_count": page_count,
        }
        out.append((text, meta))

    return out


def build_corpus_from_input(
    inline_docs: list[str] | None,
    folder: Path | None,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Merge folder files (first, sorted) then inline document strings.

    Each item is (full_document_text, metadata).
    """
    corpus: list[tuple[str, dict[str, Any]]] = []

    if folder is not None:
        corpus.extend(load_folder_files(folder))

    if inline_docs:
        for i, doc in enumerate(inline_docs):
            t = doc.strip()
            if not t:
                continue
            corpus.append(
                (
                    t,
                    {
                        "filename": f"inline_{i + 1}.txt",
                        "source_relpath": f"inline/{i + 1}.txt",
                        "file_ext": ".txt",
                        "folder_root": "",
                        "file_mtime_iso": "",
                        "file_size_bytes": len(t.encode("utf-8", errors="replace")),
                        "page_count": 0,
                    },
                )
            )

    return corpus
