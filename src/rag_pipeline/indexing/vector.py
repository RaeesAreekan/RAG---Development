from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from rag_pipeline.models import Chunk, RetrievalResult


class VectorIndex:
    """Persistent Chroma-backed dense vector index."""

    def __init__(
        self,
        path: Path,
        index_version: str,
    ) -> None:
        self.path = path
        self.index_version = index_version
        self.path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=f"dense_{index_version.replace('-', '_')}",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")

        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings, # type: ignore
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "source_path": chunk.metadata.get("source_path", ""),
                    "section_heading": chunk.section_heading or "",
                    "strategy": chunk.strategy,
                    "chunk_index": chunk.chunk_index,
                    "index_version": self.index_version,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if self.collection.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = result["documents"][0] # type: ignore
        metadatas = result["metadatas"][0] # type: ignore
        distances = result["distances"][0] # type: ignore
        ids = result["ids"][0]

        retrieval_results: list[RetrievalResult] = []

        for rank, (
            chunk_id,
            text,
            metadata,
            distance,
        ) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            score = max(0.0, min(1.0, 1.0 - float(distance)))

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    text=text,
                    score=score,
                    rank=rank,
                    retrieval_method="dense",
                    raw_score=float(distance),
                    source_path=str(metadata.get("source_path", "")),
                    section_heading=str(
                        metadata.get("section_heading", "")
                    )
                    or None,
                    metadata={
                        "index_version": self.index_version,
                        "strategy": metadata.get("strategy"),
                    },
                )
            )

        return retrieval_results