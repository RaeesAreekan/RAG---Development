from rag_pipeline.models import GeneratedAnswer
from rag_pipeline.providers.fake import FakeEmbeddingProvider, FakeGenerationProvider
from rag_pipeline.providers.openai import OpenAIEmbeddingProvider


def test_fake_embedding_provider_is_deterministic() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    first = provider.embed(["authentication API key"])
    second = provider.embed(["authentication API key"])

    assert first == second
    assert len(first) == 1
    assert len(first[0]) == 8


def test_fake_embedding_provider_preserves_batch_order() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    result = provider.embed(["first", "second"])

    assert len(result) == 2
    assert result[0] != result[1]


def test_fake_generation_provider_returns_requested_schema() -> None:
    provider = FakeGenerationProvider()

    result = provider.generate(
        prompt="Answer the question.",
        response_schema=GeneratedAnswer,
    )

    assert isinstance(result, GeneratedAnswer)
    assert result.abstained is False


def test_openai_embedding_provider_accepts_configuration_without_call() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model="text-embedding-3-small",
    )

    assert provider.model == "text-embedding-3-small"