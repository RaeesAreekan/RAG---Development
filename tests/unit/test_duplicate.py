from rag_pipeline.chunking.deduplicate import deduplicate_chunks
from rag_pipeline.models import Chunk


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        chunk_index=0,
        strategy="fixed",
        char_count=len(text),
    )


def test_exact_duplicates_are_removed() -> None:
    chunks = [
        make_chunk("chunk-1", "API keys authenticate requests."),
        make_chunk("chunk-2", "API keys authenticate requests."),
        make_chunk("chunk-3", "Timeouts control request duration."),
    ]

    result = deduplicate_chunks(chunks)

    assert [chunk.chunk_id for chunk in result.kept] == [
        "chunk-1",
        "chunk-3",
    ]
    assert result.exact_duplicate_ids == ["chunk-2"]


def test_near_duplicates_are_removed_when_enabled() -> None:
    chunks = [
        make_chunk("chunk-1", "Authentication uses API keys."),
        make_chunk("chunk-2", "API keys are used for authentication."),
        make_chunk("chunk-3", "Deployment requires a health check."),
    ]

    def fake_embed(texts: list[str]) -> list[list[float]]:
        vectors = []

        for text in texts:
            if "authentication" in text.lower() or "api keys" in text.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])

        return vectors

    result = deduplicate_chunks(
        chunks,
        near_duplicate_threshold=0.95,
        embed=fake_embed, # type: ignore
    )

    assert [chunk.chunk_id for chunk in result.kept] == [
        "chunk-1",
        "chunk-3",
    ]
    assert result.near_duplicate_ids == ["chunk-2"]


def test_near_duplicate_comparisons_are_bounded() -> None:
    chunks = [
        make_chunk(f"chunk-{index}", f"Content {index}")
        for index in range(10)
    ]

    calls = 0

    def fake_embed(texts: list[str]) -> list[list[float]]:
        # Non local makes use of immediate outer scope variable `calls` to count the number of times the embedding function is called.
        nonlocal calls
        calls += 1
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    result = deduplicate_chunks(
        chunks,
        near_duplicate_threshold=0.99,
        embed=fake_embed, # type: ignore
        max_near_duplicate_comparisons=3,
    )

    assert len(result.kept) == 10
    assert calls == 1