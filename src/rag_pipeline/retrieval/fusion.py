from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rag_pipeline.models import RetrievalResult


def rrf_fuse(
    dense_results: Sequence[RetrievalResult],
    sparse_results: Sequence[RetrievalResult],
    k_constant: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[RetrievalResult]:
    """
    Combine dense and sparse rankings using weighted RRF.

    RRF uses ranks instead of raw scores because dense and sparse scores
    have different meanings and ranges.
    """
    if k_constant < 1:
        raise ValueError("k_constant must be at least 1")

    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("retrieval weights must not be negative")

    scores: defaultdict[str, float] = defaultdict(float)
    representative: dict[str, RetrievalResult] = {}

    for result in dense_results:
        scores[result.chunk_id] += dense_weight / (
            k_constant + result.rank
        )
        representative[result.chunk_id] = result

    for result in sparse_results:
        scores[result.chunk_id] += sparse_weight / (
            k_constant + result.rank
        )
        representative.setdefault(result.chunk_id, result)

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    fused: list[RetrievalResult] = []

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        original = representative[chunk_id]

        fused.append(
            original.model_copy(
                update={
                    "score": scores[chunk_id],
                    "rank": rank,
                    "retrieval_method": "hybrid-rrf",
                    "raw_score": original.raw_score,
                    "metadata": {
                        **original.metadata,
                        "rrf_score": scores[chunk_id],
                    },
                }
            )
        )

    return fused


def weighted_score_fuse(
    dense_results: Sequence[RetrievalResult],
    sparse_results: Sequence[RetrievalResult],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[RetrievalResult]:
    """
    Optional raw-score fusion.

    This is useful experimentally, but RRF is the default because raw
    dense and BM25 scores are not naturally calibrated against each other.
    """
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("retrieval weights must not be negative")

    scores: defaultdict[str, float] = defaultdict(float)
    representative: dict[str, RetrievalResult] = {}

    for result in dense_results:
        scores[result.chunk_id] += dense_weight * result.score
        representative[result.chunk_id] = result

    for result in sparse_results:
        scores[result.chunk_id] += sparse_weight * result.score
        representative.setdefault(result.chunk_id, result)

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    return [
        representative[chunk_id].model_copy(
            update={
                "score": scores[chunk_id],
                "rank": rank,
                "retrieval_method": "hybrid-weighted",
                "metadata": {
                    **representative[chunk_id].metadata,
                    "weighted_score": scores[chunk_id],
                },
            }
        )
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    ]