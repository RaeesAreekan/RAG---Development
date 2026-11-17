from rag_pipeline.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
)
from rag_pipeline.evaluation.schema import EvaluationCase
from rag_pipeline.models import RetrievalResult

from pathlib import Path

from rag_pipeline.evaluation.schema import load_evaluation_cases

def make_result(
    chunk_id: str,
    rank: int,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=f"Text for {chunk_id}",
        score=1.0 / rank,
        rank=rank,
        retrieval_method="hybrid-rrf",
    )


def make_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="case-001",
        question="How are requests authenticated?",
        question_type="lookup",
        graded_chunk_judgments={
            "chunk-essential": 3,
            "chunk-useful": 2,
            "chunk-weak": 1,
            "chunk-irrelevant": 0,
        },
        review_status="human-approved",
    )


def test_retrieval_metrics_are_calculated() -> None:
    case = make_case()

    results = [
        make_result("chunk-irrelevant", rank=1),
        make_result("chunk-essential", rank=2),
        make_result("chunk-useful", rank=3),
    ]

    metrics = compute_retrieval_metrics(
        case=case,
        results=results,
        k_values=(1, 3),
    )

    assert metrics.recall_at_k[1] == 0.0
    assert metrics.recall_at_k[3] == 2 / 3
    assert metrics.precision_at_k[1] == 0.0
    assert metrics.precision_at_k[3] == 2 / 3
    assert metrics.reciprocal_rank == 0.5


def test_ndcg_rewards_important_chunks_at_the_top() -> None:
    case = make_case()

    good_results = [
        make_result("chunk-essential", rank=1),
        make_result("chunk-useful", rank=2),
    ]

    bad_results = [
        make_result("chunk-useful", rank=1),
        make_result("chunk-essential", rank=2),
    ]

    good_metrics = compute_retrieval_metrics(
        case=case,
        results=good_results,
        k_values=(2,),
    )

    bad_metrics = compute_retrieval_metrics(
        case=case,
        results=bad_results,
        k_values=(2,),
    )

    assert good_metrics.ndcg_at_k[2] > bad_metrics.ndcg_at_k[2]


def test_unanswerable_case_has_zero_retrieval_recall() -> None:
    case = EvaluationCase(
        case_id="case-unanswerable",
        question="What database is used?",
        question_type="unanswerable",
        expected_abstention=True,
        graded_chunk_judgments={},
        review_status="human-approved",
    )

    results = [
        make_result("chunk-1", rank=1),
    ]

    metrics = compute_retrieval_metrics(
        case=case,
        results=results,
        k_values=(1,),
    )

    assert metrics.recall_at_k[1] == 0.0
    assert metrics.reciprocal_rank == 0.0


def test_jsonl_dataset_can_be_loaded(tmp_path: Path) -> None:
    dataset = tmp_path / "golden.jsonl"

    dataset.write_text(
        (
            '{"case_id":"case-1",'
            '"question":"What is authentication?",'
            '"question_type":"lookup",'
            '"review_status":"human-approved"}\n'
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(dataset)

    assert len(cases) == 1
    assert cases[0].case_id == "case-1"