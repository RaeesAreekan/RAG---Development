from __future__ import annotations

from fastapi.testclient import TestClient

from pathlib import Path

from rag_pipeline.api.app import app
from rag_pipeline.api.dependencies import (
    RAGService,
    get_rag_service,
    get_settings
)
from rag_pipeline.generation.schemas import GroundedAnswerResult
from rag_pipeline.models import (
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

from rag_pipeline.config import Settings


def fake_service() -> RAGService:
    def retriever(question: str) -> RetrievalTrace:
        result = RetrievalResult(
            chunk_id="chunk-test-001",
            document_id="doc-test-001",
            text=(
                "Requests are authenticated using API keys."
            ),
            score=1.0,
            rank=1,
            retrieval_method="fake",
        )

        return RetrievalTrace(
            query=question,
            mode=RetrievalMode.HYBRID,
            final_results=[result],
            index_version="test-index",
            config=RetrievalConfig(),
        )

    def answer(
        question: str,
        retrieval: RetrievalTrace,
    ) -> GroundedAnswerResult:
        generated = GeneratedAnswer(
            answer=(
                "Requests are authenticated using API keys."
            ),
            claims=[
                Claim(
                    text=(
                        "Requests are authenticated using API keys."
                    ),
                    citations=["chunk-test-001"],
                )
            ],
            abstained=False,
            model_name="fake-model",
        )

        confidence = ConfidenceReport(
            retrieval_score=1.0,
            citation_coverage=1.0,
            answer_completeness=1.0,
            composite_score=1.0,
            should_abstain=False,
        )

        return GroundedAnswerResult(
            answer=generated,
            citation_checks=[],
            confidence=confidence,
        )

    class FakeGenerationService:
        def answer(
            self,
            question: str,
            retrieval: RetrievalTrace,
        ) -> GroundedAnswerResult:
            return answer(question, retrieval)

    return RAGService(
        settings=None,  # type: ignore[arg-type]
        generation_service=FakeGenerationService(),  # type: ignore[arg-type]
        retriever=retriever,
    )


app.dependency_overrides[get_rag_service] = fake_service

client = TestClient(app)


def test_ask_returns_grounded_response() -> None:
    response = client.post(
        "/v1/ask",
        json={
            "question": "How are requests authenticated?"
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["question"] == (
        "How are requests authenticated?"
    )

    assert payload["answer"]["answer"][
        "abstained"
    ] is False

    assert payload["retrieval"]["mode"] == "hybrid"
    assert payload["retrieval"]["index_version"] == (
        "test-index"
    )

    assert len(
        payload["retrieval"]["results"]
    ) == 1


def test_ask_rejects_blank_question() -> None:
    response = client.post(
        "/v1/ask",
        json={
            "question": ""
        },
    )

    assert response.status_code == 422


def test_documents_endpoint(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    document = corpus / "sample.md"
    document.write_text(
        (
            "# Example Documentation\n\n"
            "## Authentication\n\n"
            "Requests use API keys."
        ),
        encoding="utf-8",
    )

    test_settings = Settings(
        corpus_path=corpus,
        index_path=tmp_path / "indexes",
    )

    app.dependency_overrides[get_settings] = (
        lambda: test_settings
    )

    response = client.get(
        "/v1/documents"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["corpus_version"].startswith(
        "corpus-"
    )

    assert len(payload["documents"]) == 1
    assert payload["documents"][0]["source_path"] == (
        "sample.md"
    )

    app.dependency_overrides.pop(
        get_settings,
        None,
    )