from rag_pipeline.models import (
    Claim,
    GeneratedAnswer,
    RetrievalResult,
)
from rag_pipeline.verification.citations import verify_citations


def make_evidence() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk-valid",
            document_id="doc-1",
            text="API keys authenticate requests.",
            score=0.9,
            rank=1,
            retrieval_method="hybrid-rrf",
        )
    ]


def test_valid_citation_is_supported_deterministically() -> None:
    answer = GeneratedAnswer(
        answer="Requests use API keys.",
        claims=[
            Claim(
                text="Requests use API keys.",
                citations=["chunk-valid"],
            )
        ],
    )

    checks = verify_citations(
        answer=answer,
        evidence=make_evidence(),
    )

    assert len(checks) == 1
    assert checks[0].citation_exists is True
    assert checks[0].supported is True


def test_missing_citation_is_rejected() -> None:
    answer = GeneratedAnswer(
        answer="Requests use OAuth.",
        claims=[
            Claim(
                text="Requests use OAuth.",
                citations=["chunk-does-not-exist"],
            )
        ],
    )

    checks = verify_citations(
        answer=answer,
        evidence=make_evidence(),
    )

    assert checks[0].citation_exists is False
    assert checks[0].supported is False


def test_claim_without_citation_is_rejected() -> None:
    answer = GeneratedAnswer(
        answer="Requests use API keys.",
        claims=[
            Claim(
                text="Requests use API keys.",
                citations=[],
            )
        ],
    )

    checks = verify_citations(
        answer=answer,
        evidence=make_evidence(),
    )

    assert checks[0].citation_id == "<missing>"
    assert checks[0].supported is False