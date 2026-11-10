from rag_pipeline.generation.prompts import build_grounded_prompt
from rag_pipeline.models import RetrievalResult


def make_evidence() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk-auth",
            document_id="doc-fastapi",
            text="API keys authenticate requests.",
            score=0.9,
            rank=1,
            retrieval_method="hybrid-rrf",
            source_path="data/corpus/fastapi/sample.md",
            section_heading="Authentication",
        ),
        RetrievalResult(
            chunk_id="chunk-health",
            document_id="doc-deployment",
            text="The service should expose a health endpoint.",
            score=0.7,
            rank=2,
            retrieval_method="hybrid-rrf",
            source_path="data/corpus/deployment/sample.md",
            section_heading="Deployment",
        ),
    ]


def test_prompt_contains_question_and_chunk_ids() -> None:
    prompt = build_grounded_prompt(
        question="How are requests authenticated?",
        evidence=make_evidence(),
    )

    assert "How are requests authenticated?" in prompt
    assert "chunk-auth" in prompt
    assert "chunk-health" in prompt
    assert "API keys authenticate requests." in prompt


def test_prompt_rejects_empty_evidence() -> None:
    try:
        build_grounded_prompt(
            question="What is authentication?",
            evidence=[],
        )
    except ValueError as exc:
        assert "evidence" in str(exc)
    else:
        raise AssertionError("Expected empty evidence error")