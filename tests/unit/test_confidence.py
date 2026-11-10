from rag_pipeline.models import (
    Claim,
    CitationCheck,
    GeneratedAnswer,
    RetrievalResult,
)
from rag_pipeline.retrieval.engine import (
    RetrievalConfig,
    RetrievalMode,
    RetrievalTrace,
)
from rag_pipeline.verification.confidence import (
    ConfidenceThresholds,
    score_confidence,
)


def make_trace(score: float) -> RetrievalTrace:
    result = RetrievalResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="API keys authenticate requests.",
        score=score,
        rank=1,
        retrieval_method="hybrid-rrf",
    )

    return RetrievalTrace(
        query="How are requests authenticated?",
        mode=RetrievalMode.HYBRID,
        dense_results=[result],
        sparse_results=[result],
        fused_results=[result],
        final_results=[result],
        index_version="corpus-test",
        config=RetrievalConfig(
            rrf_k_constant=60,
        ),
    )


def test_well_supported_answer_has_high_confidence() -> None:
    answer = GeneratedAnswer(
        answer="Requests use API keys.",
        claims=[
            Claim(
                text="Requests use API keys.",
                citations=["chunk-1"],
            )
        ],
    )

    checks = [
        CitationCheck(
            claim_text="Requests use API keys.",
            citation_id="chunk-1",
            citation_exists=True,
            supported=True,
            entailment_score=1.0,
        )
    ]

    report = score_confidence(
        trace=make_trace(score=0.02),
        answer=answer,
        citation_checks=checks,
    )

    assert report.citation_coverage == 1.0
    assert report.should_abstain is False
    assert report.composite_score > 0.5


def test_weak_answer_triggers_abstention() -> None:
    answer = GeneratedAnswer(
        answer="The system uses a database.",
        claims=[
            Claim(
                text="The system uses a database.",
                citations=["missing"],
            )
        ],
    )

    checks = [
        CitationCheck(
            claim_text="The system uses a database.",
            citation_id="missing",
            citation_exists=False,
            supported=False,
            entailment_score=0.0,
        )
    ]

    report = score_confidence(
        trace=make_trace(score=0.001),
        answer=answer,
        citation_checks=checks,
        thresholds=ConfidenceThresholds(
            retrieval_threshold=0.35,
            citation_threshold=0.60,
        ),
    )

    assert report.should_abstain is True
    assert report.reasons