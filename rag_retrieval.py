"""Small, local and auditable document retrieval for EcoMRV reports.

This is deliberately retrieval-only: it returns source-labelled excerpts and
does not generate unsupported environmental conclusions. It provides a safe
baseline until an embedding/vector-store deployment and approved LLM are added.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    source: str
    chunk_index: int
    excerpt: str
    score: int

    def as_dict(self):
        return {"document_id": self.document_id, "source": self.source,
                "chunk_index": self.chunk_index, "excerpt": self.excerpt, "score": self.score,
                "retrieval_method": "local lexical overlap"}


def ingest_document(data: bytes, filename: str) -> dict:
    """Create traceable chunks from UTF-8/plain-text monitoring documentation."""
    if len(data) > 5 * 1024 * 1024:
        raise ValueError("Supporting documents must be smaller than 5 MB.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Use UTF-8 .txt or .md documentation.") from exc
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("The supporting document is empty.")
    chunks = [text[index:index + 900] for index in range(0, len(text), 750)]
    return {"id": hashlib.sha256(data).hexdigest(), "source": filename, "sha256": hashlib.sha256(data).hexdigest(),
            "text": text, "chunks": chunks, "ingested_at": "session"}


def retrieve(documents: list[dict], query: str, limit: int = 4) -> list[dict]:
    terms = set(TOKEN_RE.findall(query.lower()))
    hits = []
    for document in documents:
        for index, chunk in enumerate(document.get("chunks", [])):
            score = len(terms & set(TOKEN_RE.findall(chunk.lower())))
            if score:
                hits.append(RetrievedChunk(document["id"], document["source"], index, chunk, score))
    return [hit.as_dict() for hit in sorted(hits, key=lambda row: (-row.score, row.source, row.chunk_index))[:limit]]
