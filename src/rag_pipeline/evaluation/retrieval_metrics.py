from __future__ import annotations

import math
from collections.abc import Sequence

from rag_pipeline.evaluation.schema import (
    EvaluationCase,
    RetrievalMetricResult,
)
from rag_pipeline.models import RetrievalResult


def _retrieved_ids(
    results: Sequence[RetrievalResult],
    k: int,
) -> list[str]:
    return [
        result.chunk_id
        for result in results[:k]
    ]


def recall_at_k(
    case: EvaluationCase,
    results: Sequence[RetrievalResult],
    k: int,
) -> float:
    relevant_ids = {
        chunk_id
        for chunk_id, grade in case.graded_chunk_judgments.items()
        if grade > 0
    }

    if not relevant_ids:
        return 0.0

    retrieved = set(_retrieved_ids(results, k))

    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(
    case: EvaluationCase,
    results: Sequence[RetrievalResult],
    k: int,
) -> float:
    relevant_ids = {
        chunk_id
        for chunk_id, grade in case.graded_chunk_judgments.items()
        if grade > 0
    }

    if k == 0:
        return 0.0

    retrieved = set(_retrieved_ids(results, k))

    return len(retrieved & relevant_ids) / k


def reciprocal_rank(
    case: EvaluationCase,
    results: Sequence[RetrievalResult],
) -> float:
    relevant_ids = {
        chunk_id
        for chunk_id, grade in case.graded_chunk_judgments.items()
        if grade > 0
    }

    for rank, result in enumerate(results, start=1):
        if result.chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def _dcg(
    grades: list[int],
) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 2)
        for rank, grade in enumerate(grades)
    )


def ndcg_at_k(
    case: EvaluationCase,
    results: Sequence[RetrievalResult],
    k: int,
) -> float:
    if not case.graded_chunk_judgments:
        return 0.0

    retrieved_grades = [
        case.graded_chunk_judgments.get(
            result.chunk_id,
            0,
        )
        for result in results[:k]
    ]

    ideal_grades = sorted(
        case.graded_chunk_judgments.values(),
        reverse=True,
    )[:k]

    ideal_dcg = _dcg(ideal_grades)

    if ideal_dcg == 0.0:
        return 0.0

    return _dcg(retrieved_grades) / ideal_dcg


def compute_retrieval_metrics(
    case: EvaluationCase,
    results: Sequence[RetrievalResult],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalMetricResult:
    return RetrievalMetricResult(
        recall_at_k={
            k: recall_at_k(case, results, k)
            for k in k_values
        },
        # Recall = tp/tp+fn
        precision_at_k={
            k: precision_at_k(case, results, k)
            for k in k_values
        },
        # Precision - tp/tp+fp
        reciprocal_rank=reciprocal_rank(case, results),
        ndcg_at_k={
            k: ndcg_at_k(case, results, k)
            for k in k_values
        },
    )