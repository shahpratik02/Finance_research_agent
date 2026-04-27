"""
ChromaDB-backed vector index for RAG chunks.

Chunks are embedded with the same OpenAI-compatible embedding API as retrieval
queries. Each vector row stores metadata (file path, offsets, mtime, etc.) so
downstream prompts and tools can reason about provenance.

If indexing or querying fails, callers should fall back to in-process cosine / lexical search.
"""

from __future__ import annotations

import logging
import os
from typing import Any

# Before importing chromadb: telemetry uses PostHog; wrong combo spams errors in logs.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings

from app.config import settings
from app.llm_client import embed_texts

logger = logging.getLogger(__name__)


def _chroma_metadata(chunk: Any) -> dict[str, Any]:
    """Chroma accepts str, int, float, bool — normalize everything."""
    m = chunk.to_vector_metadata()
    out: dict[str, Any] = {}
    for k, v in m.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = v
        else:
            s = str(v)
            if len(s) > 2000:
                s = s[:1997] + "..."
            out[k] = s
    return out


def index_and_query(
    run_id: str,
    chunks: list[Any],
    retrieval_query: str,
    top_k: int,
) -> list[int]:
    """
    Create a run-scoped collection, upsert all chunk embeddings + metadata,
    query by embedding of ``retrieval_query``, then delete the collection.

    Returns ``flat_idx`` values ordered by relevance (best first).
    """
    if not chunks:
        return []

    model_id = settings.embedding_model_id
    if not model_id:
        raise RuntimeError("embedding model required for Chroma RAG")

    settings.chroma_rag_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_rag_path),
        settings=Settings(anonymized_telemetry=False),
    )

    collection_name = f"rag_{run_id.replace('-', '')}"
    try:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [c.text for c in chunks]
        ids = [str(c.flat_idx) for c in chunks]
        metadatas = [_chroma_metadata(c) for c in chunks]

        batch = 64
        for start in range(0, len(texts), batch):
            end = start + batch
            sub_texts = texts[start:end]
            sub_ids = ids[start:end]
            sub_meta = metadatas[start:end]
            sub_emb = embed_texts(sub_texts, model_id)
            collection.add(
                ids=sub_ids,
                documents=sub_texts,
                embeddings=sub_emb,
                metadatas=sub_meta,
            )

        q_emb = embed_texts([retrieval_query], model_id)[0]
        n_results = min(top_k, len(chunks))
        # Chroma 0.5.x: ``include`` may not list ``ids``; ``flat_idx`` is stored on metadatas.
        result = collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metas_row = (result.get("metadatas") or [[]])[0]
        ranked: list[int] = []
        for md in metas_row:
            if not md:
                continue
            fi = md.get("flat_idx")
            if fi is None:
                continue
            ranked.append(int(fi))
        return ranked
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as e:
            logger.debug(f"[rag_vector_store] cleanup collection {collection_name}: {e}")
