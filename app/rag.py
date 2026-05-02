"""
RAG phase: load inline and/or folder-backed documents, chunk them, retrieve passages
(Chroma vector DB + metadata when embeddings are configured; otherwise TF–IDF),
then extract structured claims via a single structured LLM call.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.config import settings
from app.llm_client import RAG_PROFILE, chat, embed_texts
from app.rag_corpus import build_corpus_from_input, resolve_documents_folder
from app.schemas import (
    MCPProvider,
    PlannerOutput,
    QueryInput,
    RagAdequacy,
    RagPhaseOutput,
    ResearchResult,
    SourceRecord,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag.txt"


@dataclass
class ChunkRecord:
    """One text span with provenance metadata (stored in Chroma + SourceRecord)."""

    flat_idx: int
    doc_index: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    filename: str
    source_relpath: str
    file_ext: str
    file_mtime_iso: str
    folder_root: str
    file_size_bytes: int
    page_count: int = 0  # PDF page count when source is .pdf; else 0

    def to_vector_metadata(self) -> dict[str, Any]:
        return {
            "flat_idx": self.flat_idx,
            "doc_index": self.doc_index,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "filename": self.filename,
            "source_relpath": self.source_relpath,
            "file_ext": self.file_ext,
            "file_mtime_iso": self.file_mtime_iso,
            "folder_root": self.folder_root,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
        }

    def to_source_payload(self) -> dict[str, Any]:
        pages = f" · {self.page_count} pdf pages" if self.page_count and self.file_ext == ".pdf" else ""
        return {
            **self.to_vector_metadata(),
            "retrieval_hint": (
                f"{self.filename}{pages} · chars {self.char_start}–{self.char_end} · {self.source_relpath}"
            ),
        }


def normalize_documents(query_input: QueryInput) -> list[str] | None:
    """Return non-empty stripped inline document strings, or None."""
    raw = query_input.documents
    if not raw:
        return None
    cleaned = [d.strip() for d in raw if d and str(d).strip()]
    return cleaned or None


def normalize_folder_path(query_input: QueryInput) -> Path | None:
    """Return resolved folder path if ``documents_folder`` is set."""
    raw = query_input.documents_folder
    if not raw or not str(raw).strip():
        return None
    return resolve_documents_folder(str(raw).strip())


def should_run_rag(query_input: QueryInput) -> bool:
    """True when inline docs and/or a documents folder should trigger RAG."""
    return (
        normalize_documents(query_input) is not None
        or normalize_folder_path(query_input) is not None
    )


def run_rag_phase(
    query_input: QueryInput,
    planner_output: PlannerOutput,
    run_id: str,
) -> tuple[RagAdequacy, ResearchResult, list[SourceRecord]]:
    """
    Retrieve from user documents / folder and produce a ResearchResult-shaped bundle
    plus an adequacy label.
    """
    if not should_run_rag(query_input):
        raise ValueError("run_rag_phase requires documents and/or documents_folder")

    folder = normalize_folder_path(query_input)
    inline = normalize_documents(query_input)
    corpus = build_corpus_from_input(inline, folder)
    if not corpus:
        empty = ResearchResult(
            subquestion_answers=[],
            claims=[],
            gaps=["No readable text found in the given folder or inline documents."],
        )
        return RagAdequacy.none, empty, []

    flat = _corpus_to_chunks(corpus)
    if not flat:
        empty = ResearchResult(
            subquestion_answers=[],
            claims=[],
            gaps=["No usable text in provided documents after chunking."],
        )
        return RagAdequacy.none, empty, []

    sources, retrieval_mode = _select_chunks(query_input, planner_output, flat, run_id)

    template = _PROMPT_PATH.read_text(encoding="utf-8")
    subquestions_block = "\n".join(f"- {sq}" for sq in planner_output.subquestions)
    claim_id_prefix = f"clm_{run_id[:8]}"
    passages_block = "\n\n".join(_format_passage_for_prompt(s) for s in sources)
    source_ids_line = ", ".join(s.source_id for s in sources)
    system_prompt = template.format(
        subquestions_block=subquestions_block,
        query=query_input.query,
        passages_block=passages_block,
        claim_id_prefix=claim_id_prefix,
        source_ids_line=source_ids_line,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Extract the RagPhaseOutput JSON now."},
    ]

    logger.info(
        f"[rag] Retrieval — chunks_total={len(flat)} selected={len(sources)} "
        f"mode={retrieval_mode}"
    )

    result = chat(messages, profile=RAG_PROFILE, response_schema=RagPhaseOutput)
    if result.parsed is None:
        raise ValueError(
            f"RAG extractor returned no structured output.\nRaw: {result.content}"
        )

    parsed = result.parsed
    # Do not retry when the model already judged docs irrelevant (adequacy none).
    # A follow-up "must include at least one claim" would force wrong-entity or hallucinated facts.
    if not parsed.claims and sources and parsed.adequacy != RagAdequacy.none:
        logger.warning(
            "[rag] RAG extractor returned zero claims with non-empty passages; retrying once"
        )
        messages.append({"role": "assistant", "content": result.content or ""})
        messages.append({
            "role": "user",
            "content": (
                "Your last output had zero claims but passages are present. "
                "If the excerpts concern a different company or ticker than the user query, "
                "keep adequacy none and zero claims and explain in gaps — never attribute "
                "those excerpts to the queried company. "
                "Otherwise output RagPhaseOutput again: include at least one Claim quoting "
                "specific numbers or fiscal periods from the excerpts, each claim "
                f"citing only these source_ids: {source_ids_line}. "
                "If the query asks for stock price or live market state but the PDF "
                "only has financial statements, set adequacy to partial—not none."
            ),
        })
        retry = chat(messages, profile=RAG_PROFILE, response_schema=RagPhaseOutput)
        if retry.parsed is not None:
            parsed = retry.parsed
    adequacy = _effective_adequacy(parsed)
    research = ResearchResult(
        subquestion_answers=parsed.subquestion_answers,
        claims=parsed.claims,
        gaps=parsed.gaps,
    )
    logger.info(f"[rag] Done — adequacy={adequacy.value} claims={len(research.claims)}")
    return adequacy, research, sources


def _effective_adequacy(parsed: RagPhaseOutput) -> RagAdequacy:
    if not parsed.claims:
        return RagAdequacy.none
    return parsed.adequacy


def _corpus_to_chunks(corpus: list[tuple[str, dict[str, Any]]]) -> list[ChunkRecord]:
    out: list[ChunkRecord] = []
    max_chunks = settings.rag_max_chunks
    size = settings.rag_chunk_size
    overlap = settings.rag_chunk_overlap
    flat_counter = 0

    for doc_index, (text, meta) in enumerate(corpus):
        if len(out) >= max_chunks:
            break
        for chunk_index, (cs, ce, chunk_text) in enumerate(_chunk_text_spans(text, size, overlap)):
            if len(out) >= max_chunks:
                break
            out.append(
                ChunkRecord(
                    flat_idx=flat_counter,
                    doc_index=doc_index,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_start=cs,
                    char_end=ce,
                    filename=str(meta.get("filename", "")),
                    source_relpath=str(meta.get("source_relpath", "")),
                    file_ext=str(meta.get("file_ext", "")),
                    file_mtime_iso=str(meta.get("file_mtime_iso", "")),
                    folder_root=str(meta.get("folder_root", "")),
                    file_size_bytes=int(meta.get("file_size_bytes", 0) or 0),
                    page_count=int(meta.get("page_count", 0) or 0),
                )
            )
            flat_counter += 1
    return out


def _chunk_text_spans(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    text = text.strip()
    if not text:
        return []
    if size <= 0:
        return [(0, len(text), text)]
    overlap = min(max(0, overlap), size - 1) if size > 1 else 0
    spans: list[tuple[int, int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        spans.append((start, end, text[start:end]))
        if end >= n:
            break
        nxt = end - overlap
        if nxt <= start:
            nxt = start + 1
        start = nxt
    return spans


def _retrieval_query(query: str, planner_output: PlannerOutput) -> str:
    subs = "\n".join(f"- {sq}" for sq in planner_output.subquestions)
    return f"{query.rstrip()}\n\nRelated sub-questions:\n{subs}"


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _lexical_top_indices(query: str, chunk_texts: list[str], top_k: int) -> list[int]:
    if not chunk_texts:
        return []
    qt = set(_tokenize(query))
    if not qt:
        return list(range(min(top_k, len(chunk_texts))))

    df: dict[str, int] = {}
    tfs: list[dict[str, int]] = []
    for ch in chunk_texts:
        toks = _tokenize(ch)
        tfs.append({t: toks.count(t) for t in set(toks)} if toks else {})
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    n_docs = len(chunk_texts)
    scores: list[float] = []
    for tf in tfs:
        sc = 0.0
        for t in qt:
            if t not in df:
                continue
            idf = math.log((1 + n_docs) / (1 + df[t])) + 1.0
            cnt = tf.get(t, 0)
            tf_w = (1 + math.log(cnt)) if cnt > 0 else 0.0
            sc += tf_w * idf
        scores.append(sc)

    ranked = sorted(range(len(chunk_texts)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]


def _embedding_top_indices(query: str, chunk_texts: list[str], top_k: int) -> list[int]:
    model_id = settings.embedding_model_id
    assert model_id
    qv = np.array(embed_texts([query], model_id)[0], dtype=np.float64)
    batch = 48
    sims: list[float] = []
    for i in range(0, len(chunk_texts), batch):
        batch_texts = chunk_texts[i : i + batch]
        ev = np.array(embed_texts(batch_texts, model_id), dtype=np.float64)
        qn = np.linalg.norm(qv)
        en = np.linalg.norm(ev, axis=1)
        cos = (ev @ qv) / (qn * en + 1e-9)
        sims.extend(cos.tolist())
    ranked = sorted(range(len(chunk_texts)), key=lambda i: sims[i], reverse=True)
    return ranked[:top_k]


def _select_chunks(
    query_input: QueryInput,
    planner_output: PlannerOutput,
    flat: list[ChunkRecord],
    run_id: str,
) -> tuple[list[SourceRecord], str]:
    """
    Return selected sources and the retrieval mode actually used:
    ``chroma``, ``embed`` (in-process cosine), or ``lexical`` (TF–IDF).
    """
    retrieval_query = _retrieval_query(query_input.query, planner_output)
    texts = [c.text for c in flat]
    top_k = min(settings.rag_top_k, len(flat))

    idxs: list[int]
    mode: str
    if settings.embedding_model_id and settings.rag_use_chroma:
        try:
            from app.rag_vector_store import index_and_query

            idxs = index_and_query(run_id, flat, retrieval_query, top_k)
            mode = "chroma"
        except Exception as e:
            logger.warning(f"[rag] Chroma vector retrieval failed ({e}); using in-process embeddings")
            try:
                idxs = _embedding_top_indices(retrieval_query, texts, top_k)
                mode = "embed"
            except Exception as e2:
                logger.warning(f"[rag] Embedding retrieval failed ({e2}); using lexical")
                idxs = _lexical_top_indices(query_input.query, texts, top_k)
                mode = "lexical"
    elif settings.embedding_model_id:
        try:
            idxs = _embedding_top_indices(retrieval_query, texts, top_k)
            mode = "embed"
        except Exception as e:
            logger.warning(f"[rag] Embedding retrieval failed ({e}); using lexical")
            idxs = _lexical_top_indices(query_input.query, texts, top_k)
            mode = "lexical"
    else:
        idxs = _lexical_top_indices(query_input.query, texts, top_k)
        mode = "lexical"

    if not idxs:
        idxs = _lexical_top_indices(query_input.query, texts, top_k)
        mode = "lexical"

    selected = [flat[i] for i in idxs if 0 <= i < len(flat)]
    return [_chunk_to_source(run_id, c) for c in selected], mode


def _format_passage_for_prompt(s: SourceRecord) -> str:
    pl = s.structured_payload
    fn = pl.get("filename", "")
    rel = pl.get("source_relpath", "")
    cs = pl.get("char_start", "")
    ce = pl.get("char_end", "")
    mt = pl.get("file_mtime_iso", "")
    root = pl.get("folder_root", "")
    pc = pl.get("page_count", 0) or 0
    pages = f" pages={pc}" if int(pc) > 0 else ""
    hint = (
        f"file={fn} path={rel} chars={cs}-{ce}{pages} mtime={mt}"
        + (f" root={root}" if root else "")
    )
    return f"[{s.source_id}] {hint}\n{s.raw_excerpt}"


def _chunk_to_source(run_id: str, chunk: ChunkRecord) -> SourceRecord:
    prefix = run_id[:8].replace("-", "")
    uid = uuid4().hex[:8]
    source_id = f"{prefix}-ragd-{uid}"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    excerpt = chunk.text[:4000] if len(chunk.text) > 4000 else chunk.text
    summary = chunk.text[:400].replace("\n", " ")
    if len(chunk.text) > 400:
        summary += "..."
    pl = chunk.to_source_payload()
    title = f"{chunk.filename} · §{chunk.chunk_index + 1} · {chunk.source_relpath}"
    file_uri: str | None = None
    if chunk.folder_root:
        try:
            file_uri = (Path(chunk.folder_root) / chunk.source_relpath).resolve().as_uri()
        except (OSError, ValueError):
            file_uri = None
    return SourceRecord(
        source_id=source_id,
        run_id=run_id,
        provider=MCPProvider.rag_document,
        tool="rag_chunk",
        title=title,
        uri=file_uri,
        retrieved_at=now,
        entity=chunk.filename,
        content_summary=summary,
        raw_excerpt=excerpt,
        structured_payload=pl,
    )
