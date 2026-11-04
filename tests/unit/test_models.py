from rag_pipeline.models import (
    Claim,
    Chunk,
    CorpusVersion,
    Document,
    GeneratedAnswer,
    make_chunk_id,
)


def test_chunk_id_is_deterministic() -> None:
    first = make_chunk_id("corpus-v1", "document-a", 0)
    second = make_chunk_id("corpus-v1", "document-a", 0)

    assert first == second


def test_chunk_id_changes_when_position_changes() -> None:
    first = make_chunk_id("corpus-v1", "document-a", 0)
    second = make_chunk_id("corpus-v1", "document-a", 1)

    assert first != second


def test_document_requires_non_empty_content() -> None:
    try:
        Document(
            document_id="doc-1",
            source_path="docs/example.md",
            content="",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Empty document content should be rejected")


def test_generated_answer_supports_claim_level_citations() -> None:
    answer = GeneratedAnswer(
        answer="Use the `/health` endpoint.",
        claims=[
            Claim(
                text="The service exposes a `/health` endpoint.",
                citations=["chunk-001"],
            )
        ],
        abstained=False,
    )

    assert answer.claims[0].citations == ["chunk-001"]


def test_corpus_version_contains_file_hashes() -> None:
    corpus = CorpusVersion(
        version_id="corpus-v1",
        file_hashes={"docs/example.md": "abc123"},
    )

    assert corpus.file_hashes["docs/example.md"] == "abc123"