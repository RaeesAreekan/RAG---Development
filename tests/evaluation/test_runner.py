from pathlib import Path

from rag_pipeline.evaluation.runner import run_evaluation
from rag_pipeline.generation.schemas import GroundedAnswerResult
from rag_pipeline.models import (
    CitationCheck,
    Claim,
    ConfidenceReport,
    GeneratedAnswer,
    RetrievalResult,
)
from rag_pipeline.retrieval.engine import (
    RetrievalConfig,
    RetrievalMode,
    RetrievalTrace,
)


def make_trace(question: str) -> RetrievalTrace:
    result = RetrievalResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Requests are authenticated using API keys.",
        score=1.0,
        rank=1,
        retrieval_method="hybrid-rrf",
    )

    return RetrievalTrace(
        query=question,
        mode=RetrievalMode.HYBRID,
        final_results=[result],
        index_version="test-index",
        config=RetrievalConfig(),
    )


def make_answer(
    question: str,
    trace: RetrievalTrace,
) -> GroundedAnswerResult:
    answer = GeneratedAnswer(
        answer="Requests are authenticated using API keys.",
        claims=[
            Claim(
                text="Requests are authenticated using API keys.",
                citations=["chunk-1"],
            )
        ],
        abstained=False,
    )

    citation_check = CitationCheck(
        claim_text="Requests are authenticated using API keys.",
        citation_id="chunk-1",
        citation_exists=True,
        supported=True,
        entailment_score=1.0,
    )

    confidence = ConfidenceReport(
        retrieval_score=1.0,
        citation_coverage=1.0,
        answer_completeness=1.0,
        composite_score=1.0,
        should_abstain=False,
    )

    return GroundedAnswerResult(
        answer=answer,
        citation_checks=[citation_check],
        confidence=confidence,
    )


def test_evaluation_runner(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "golden.jsonl"

    dataset.write_text(
        (
            '{"case_id":"case-1",'
            '"question":"How are requests authenticated?",'
            '"question_type":"lookup",'
            '"reference_answer":"Requests are authenticated using API keys.",'
            '"graded_chunk_judgments":{"chunk-1":3},'
            '"required_evidence_units":["API keys"],'
            '"review_status":"human-approved"}\n'
        ),
        encoding="utf-8",
    )

    report = run_evaluation(
        dataset=dataset,
        retriever=make_trace,
        answerer=make_answer,
        k_values=(1,),
    )

    assert report.case_count == 1
    assert report.retrieval_metrics.recall_at_k[1] == 1.0
    assert report.answer_metrics.citation_coverage == 1.0