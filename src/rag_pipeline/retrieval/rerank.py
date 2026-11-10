from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_pipeline.models import RetrievalResult


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        ...


class KeywordOverlapReranker:
    """
    Deterministic reranker for tests and offline development.

    This is not a semantic model. It gives us a stable baseline before
    adding the optional sentence-transformers cross-encoder.
    """

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be blank")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_terms = set(query.lower().split())
        scored: list[tuple[float, RetrievalResult]] = []

        for candidate in candidates:
            candidate_terms = set(candidate.text.lower().split())
            overlap = len(query_terms & candidate_terms)

            normalized_score = overlap / max(len(query_terms), 1)

            scored.append(
                (
                    normalized_score,
                    candidate.model_copy(
                        update={
                            "score": normalized_score,
                            "retrieval_method": "reranked",
                            "metadata": {
                                **candidate.metadata,
                                "reranker": "keyword-overlap",
                            },
                        }
                    ),
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            candidate.model_copy(update={"rank": rank})
            for rank, (_, candidate) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]


class CrossEncoderReranker:
    """
    Optional local cross-encoder reranker.

    Install the optional dependency with:

        python -m pip install -e ".[reranker]"
    """

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Install the reranker dependency with "
                '`pip install -e ".[reranker]"`'
            ) from exc

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be blank")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        pairs = [
            [query, candidate.text]
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        scored = [
            (
                float(score),
                candidate.model_copy(
                    update={
                        "score": float(score),
                        "retrieval_method": "cross-encoder-reranked",
                        "metadata": {
                            **candidate.metadata,
                            "reranker": "cross-encoder",
                        },
                    }
                ),
            )
            for candidate, score in zip(candidates, scores)
        ]

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            candidate.model_copy(update={"rank": rank})
            for rank, (_, candidate) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]