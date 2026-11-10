from pathlib import Path

from rag_pipeline.chunking.base import ChunkingConfig
from rag_pipeline.chunking.fixed import FixedChunker
from rag_pipeline.indexing.sparse import SparseIndex
from rag_pipeline.indexing.vector import VectorIndex
from rag_pipeline.models import Document
from rag_pipeline.providers.fake import FakeEmbeddingProvider
from rag_pipeline.retrieval.engine import (
    RetrievalConfig,
    RetrievalEngine,
    RetrievalMode,
)
from rag_pipeline.retrieval.rerank import KeywordOverlapReranker


def build_engine(tmp_path: Path) -> RetrievalEngine:
    document = Document(
        document_id="doc-engine-test",
        source_path="docs/engine.md",
        content=(
            "Authentication uses API keys.\n\n"
            "Deployment requires a health endpoint.\n\n"
            "Timeout configuration controls request duration."
        ),
    )

    chunks = FixedChunker().chunk(
        document,
        ChunkingConfig(
            corpus_version="corpus-engine-test",
            max_chars=80,
            overlap_chars=10,
        ),
    )

    embedder = FakeEmbeddingProvider(dimensions=8)
    embeddings = embedder.embed([chunk.text for chunk in chunks])

    dense_index = VectorIndex(
        path=tmp_path / "dense",
        index_version="corpus-engine-test",
    )
    dense_index.upsert(chunks, embeddings)

    sparse_index = SparseIndex.build(
        chunks=chunks,
        path=tmp_path / "sparse",
        index_version="corpus-engine-test",
    )

    return RetrievalEngine(
        dense_index=dense_index,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=KeywordOverlapReranker(),
    )


def test_dense_mode_returns_dense_results(tmp_path: Path) -> None:
    engine = build_engine(tmp_path)

    trace = engine.search(
        query="API keys",
        mode=RetrievalMode.DENSE,
    )

    assert trace.mode == RetrievalMode.DENSE
    assert trace.dense_results
    assert not trace.sparse_results
    assert trace.final_results
    assert all(
        result.retrieval_method == "dense"
        for result in trace.final_results
    )


def test_sparse_mode_returns_sparse_results(tmp_path: Path) -> None:
    engine = build_engine(tmp_path)

    trace = engine.search(
        query="health endpoint",
        mode=RetrievalMode.SPARSE,
    )

    assert trace.sparse_results
    assert not trace.dense_results
    assert all(
        result.retrieval_method == "sparse"
        for result in trace.final_results
    )


def test_hybrid_mode_contains_dense_and_sparse_evidence(
    tmp_path: Path,
) -> None:
    engine = build_engine(tmp_path)

    trace = engine.search(
        query="API keys",
        mode=RetrievalMode.HYBRID,
        config=RetrievalConfig(final_top_k=2),
    )

    assert trace.dense_results
    assert trace.sparse_results
    assert trace.fused_results
    assert len(trace.final_results) == 2


def test_reranked_mode_requires_and_uses_reranker(
    tmp_path: Path,
) -> None:
    engine = build_engine(tmp_path)

    trace = engine.search(
        query="API keys",
        mode=RetrievalMode.HYBRID_RERANKED,
        config=RetrievalConfig(final_top_k=2),
    )

    assert trace.final_results
    assert all(
        result.retrieval_method == "reranked"
        for result in trace.final_results
    )