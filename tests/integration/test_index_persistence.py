from pathlib import Path

from rag_pipeline.chunking.base import ChunkingConfig
from rag_pipeline.chunking.fixed import FixedChunker
from rag_pipeline.indexing.sparse import SparseIndex
from rag_pipeline.indexing.vector import VectorIndex
from rag_pipeline.models import Document
from rag_pipeline.providers.fake import FakeEmbeddingProvider


def make_chunks():
    document = Document(
        document_id="doc-index-test",
        source_path="docs/index.md",
        content=(
            "Authentication uses API keys.\n\n"
            "Deployment requires a health endpoint.\n\n"
            "Timeout configuration controls request duration."
        ),
    )

    return FixedChunker().chunk(
        document,
        ChunkingConfig(
            corpus_version="corpus-index-test",
            max_chars=80,
            overlap_chars=10,
        ),
    )


def test_dense_index_persists_and_retrieves(tmp_path: Path) -> None:
    chunks = make_chunks()
    provider = FakeEmbeddingProvider(dimensions=8)
    embeddings = provider.embed([chunk.text for chunk in chunks])

    index = VectorIndex(
        path=tmp_path / "dense",
        index_version="corpus-index-test",
    )
    index.upsert(chunks, embeddings)

    loaded = VectorIndex(
        path=tmp_path / "dense",
        index_version="corpus-index-test",
    )
    results = loaded.search(embeddings[0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == chunks[0].chunk_id
    assert results[0].retrieval_method == "dense"


def test_sparse_index_persists_and_retrieves(tmp_path: Path) -> None:
    chunks = make_chunks()

    index = SparseIndex.build(
        chunks=chunks,
        path=tmp_path / "sparse",
        index_version="corpus-index-test",
    )

    loaded = SparseIndex.load(
        path=tmp_path / "sparse",
        index_version="corpus-index-test",
    )
    results = loaded.search("API keys", top_k=2)

    assert len(results) == 2
    assert results[0].retrieval_method == "sparse"
    assert "API" in results[0].text


def test_sparse_index_rejects_wrong_version(tmp_path: Path) -> None:
    chunks = make_chunks()

    SparseIndex.build(
        chunks=chunks,
        path=tmp_path / "sparse",
        index_version="corpus-index-test",
    )

    try:
        SparseIndex.load(
            path=tmp_path / "sparse",
            index_version="wrong-version",
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected missing-version error")