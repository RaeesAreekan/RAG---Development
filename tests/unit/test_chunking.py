from rag_pipeline.chunking.base import ChunkingConfig
from rag_pipeline.chunking.fixed import FixedChunker
from rag_pipeline.chunking.semantic import SemanticChunker
from rag_pipeline.chunking.structure import StructureAwareChunker
from rag_pipeline.models import Document, DocumentSection


def make_document() -> Document:
    return Document(
        document_id="doc-example",
        source_path="docs/example.md",
        title="Example",
        content=(
            "# Authentication\n\n"
            "Use API keys for authentication. "
            "The API key must be sent in the request header.\n\n"
            "## Configuration\n\n"
            "The timeout defaults to thirty seconds. "
            "The retry count can be configured per client."
        ),
        sections=[
            DocumentSection(
                heading="Authentication",
                text=(
                    "Use API keys for authentication. "
                    "The API key must be sent in the request header."
                ),
            ),
            DocumentSection(
                heading="Configuration",
                text=(
                    "The timeout defaults to thirty seconds. "
                    "The retry count can be configured per client."
                ),
            ),
        ],
    )


def make_config() -> ChunkingConfig:
    return ChunkingConfig(
        corpus_version="corpus-test",
        max_chars=100,
        overlap_chars=20,
    )


def test_fixed_chunker_creates_overlapping_chunks() -> None:
    document = make_document()

    chunks = FixedChunker().chunk(document, make_config())

    assert len(chunks) > 1
    assert all(chunk.strategy == "fixed" for chunk in chunks)
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert chunks[0].chunk_index == 0

    first_end = chunks[0].text[-18:]
    assert first_end in chunks[1].text


def test_structure_aware_chunker_preserves_section_heading() -> None:
    document = make_document()

    chunks = StructureAwareChunker().chunk(document, make_config())

    headings = {chunk.section_heading for chunk in chunks}

    assert "Authentication" in headings
    assert "Configuration" in headings
    assert all(chunk.strategy == "structure-aware" for chunk in chunks)


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []

        for text in texts:
            lowered = text.lower()

            if "authentication" in lowered or "api key" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])

        return vectors


def test_semantic_chunker_groups_similar_paragraphs() -> None:
    document = Document(
        document_id="doc-semantic",
        source_path="docs/semantic.md",
        content=(
            "Authentication uses API keys.\n\n"
            "API keys are sent in request headers.\n\n"
            "Deployment requires a health check.\n\n"
            "The health check should be lightweight."
        ),
    )

    chunker = SemanticChunker(embedder=FakeEmbedder()) # type: ignore

    chunks = chunker.chunk(
        document,
        ChunkingConfig(
            corpus_version="corpus-test",
            max_chars=500,
            semantic_similarity_threshold=0.8,
        ),
    )

    assert len(chunks) == 2
    assert chunks[0].strategy == "semantic"
    assert "Authentication" in chunks[0].text
    assert "Deployment" in chunks[1].text


def test_chunk_ids_are_unique() -> None:
    document = make_document()

    chunks = FixedChunker().chunk(document, make_config())
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))