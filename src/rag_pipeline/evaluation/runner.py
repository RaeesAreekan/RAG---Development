from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from rag_pipeline.evaluation.answer_metrics import (
    compute_answer_metrics,
)
from rag_pipeline.evaluation.report import (
    EvaluationCaseResult,
    EvaluationReport,
)
from rag_pipeline.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
)
from rag_pipeline.evaluation.schema import (
    EvaluationCase,
    load_benchmark_cases,
)
from rag_pipeline.generation.schemas import GroundedAnswerResult
from rag_pipeline.retrieval.engine import RetrievalTrace

from time import perf_counter
from typing import Any

from rag_pipeline.evaluation.cache import (
    CachedEvaluationCase,
    load_cached_case,
    make_cache_key,
    save_cached_case,
)


Retriever = Callable[[str], RetrievalTrace]
Answerer = Callable[
    [str, RetrievalTrace],
    GroundedAnswerResult,
]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def _average_retrieval_metrics(
    metrics: Sequence,
    k_values: Sequence[int],
):
    return {
        "recall_at_k": {
            k: _mean(
                [
                    result.recall_at_k[k]
                    for result in metrics
                ]
            )
            for k in k_values
        },
        "precision_at_k": {
            k: _mean(
                [
                    result.precision_at_k[k]
                    for result in metrics
                ]
            )
            for k in k_values
        },
        "reciprocal_rank": _mean(
            [
                result.reciprocal_rank
                for result in metrics
            ]
        ),
        "ndcg_at_k": {
            k: _mean(
                [
                    result.ndcg_at_k[k]
                    for result in metrics
                ]
            )
            for k in k_values
        },
    }


def run_evaluation(
    dataset: Path,
    retriever: Retriever,
    answerer: Answerer,
    k_values: Sequence[int] = (1, 3, 5, 10),
    cache_dir: Path | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """
    Run retrieval and answer evaluation against approved cases.

    The retriever and answerer are injected so this function can be tested
    offline with fake providers.
    """
    cases = load_benchmark_cases(dataset)

    retrieval_results = []
    generated_answers = {}
    citation_checks = {}
    case_results = []

    metadata = dict(run_metadata or {})
    cache_hits = 0
    started_at = perf_counter()
    for case in cases:
        cache_key = make_cache_key(
            case_id=case.case_id,
            question=case.question,
            metadata=metadata,
        )
        
        cached = None

        if cache_dir is not None:
            cached = load_cached_case(
                cache_dir=cache_dir,
                cache_key=cache_key,
            )

        if cached is not None:
            cache_hits += 1
            trace = cached.retrieval_trace
            grounded_answer = cached.grounded_answer
        else:

            trace = retriever(case.question)

            grounded_answer = answerer(
                case.question,
                trace,
            )
            if cache_dir is not None:
                save_cached_case(
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    case=CachedEvaluationCase(
                        case_id=case.case_id,
                        question=case.question,
                        retrieval_trace=trace,
                        grounded_answer=grounded_answer,
                    ),
                )
        retrieval_metrics = compute_retrieval_metrics(
            case=case,
            results=trace.final_results,
            k_values=k_values,
        )
        retrieval_results.append(retrieval_metrics)
        generated_answers[case.case_id] = (
            grounded_answer.answer
        )
        citation_checks[case.case_id] = (
            grounded_answer.citation_checks
        )

        case_results.append(
            EvaluationCaseResult(
                case_id=case.case_id,
                retrieval_metrics=retrieval_metrics,
                grounded_answer=grounded_answer,
            )
        )

    answer_metrics = compute_answer_metrics(
        cases=cases,
        answers=generated_answers,
        citation_checks=citation_checks,
    )

    aggregate_retrieval = (
        _average_retrieval_metrics(
            metrics=retrieval_results,
            k_values=k_values,
        )
    )
    elapsed_seconds = perf_counter() - started_at

    metadata.update(
        {
            "dataset": str(dataset),
            "case_count": len(cases),
            "cache_enabled": cache_dir is not None,
            "cache_hits": cache_hits,
            "cache_misses": len(cases) - cache_hits,
            "duration_seconds": round(
                elapsed_seconds,
                3,
            ),
        }
    )
    return EvaluationReport(
        dataset_path=str(dataset),
        case_count=len(cases),
        retrieval_metrics=aggregate_retrieval, # type: ignore
        answer_metrics=answer_metrics,
        cases=case_results,
        run_metadata=metadata,
    )