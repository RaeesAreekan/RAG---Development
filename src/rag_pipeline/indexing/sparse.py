from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag_pipeline.models import Chunk, RetrievalResult


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[\w./:-]+\b", text.lower())


class SparseIndex:
    """Persisted BM25 index over the same chunks as the dense index."""

    def __init__(
        self,
        path: Path,
        index_version: str,
        bm25: BM25Okapi,
        chunks: list[Chunk],
    ) -> None:
        self.path = path
        self.index_version = index_version
        self.bm25 = bm25
        self.chunks = chunks

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        path: Path,
        index_version: str,
    ) -> "SparseIndex":
        if not chunks:
            raise ValueError("Cannot build sparse index from empty chunks")

        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        index = cls(
            path=path,
            index_version=index_version,
            bm25=bm25,
            chunks=chunks,
        )
        index.save()

        return index

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / f"sparse_{self.index_version}.pkl"

        with target.open("wb") as file:
            pickle.dump(
                {
                    "index_version": self.index_version,
                    "bm25": self.bm25,
                    "chunks": self.chunks,
                },
                file,
            )

    @classmethod
    def load(
        cls,
        path: Path,
        index_version: str,
    ) -> "SparseIndex":
        target = path / f"sparse_{index_version}.pkl"

        if not target.exists():
            raise FileNotFoundError(f"Sparse index not found: {target}")

        with target.open("rb") as file:
            payload = pickle.load(file)

        if payload["index_version"] != index_version:
            raise ValueError("Sparse index version mismatch")

        return cls(
            path=path,
            index_version=index_version,
            bm25=payload["bm25"],
            chunks=payload["chunks"],
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[RetrievalResult] = []

        for rank, chunk_index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[chunk_index]
            raw_score = float(scores[chunk_index])

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=raw_score,
                    rank=rank,
                    retrieval_method="sparse",
                    raw_score=raw_score,
                    source_path=chunk.metadata.get("source_path"),
                    section_heading=chunk.section_heading,
                    metadata={
                        "index_version": self.index_version,
                        "strategy": chunk.strategy,
                    },
                )
            )

        return results