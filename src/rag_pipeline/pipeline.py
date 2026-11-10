from __future__ import annotations

from pathlib import Path

from rag_pipeline.chunking.base import ChunkingConfig
from rag_pipeline.chunking.deduplicate import deduplicate_chunks
from rag_pipeline.chunking.structure import StructureAwareChunker
from rag_pipeline.config import Settings
from rag_pipeline.indexing.manifest import IndexManifest
from rag_pipeline.indexing.sparse import SparseIndex
from rag_pipeline.indexing.vector import VectorIndex
from rag_pipeline.ingestion.loaders import load_document
from rag_pipeline.ingestion.manifest import build_corpus_manifest
from rag_pipeline.models import Chunk, RetrievalResult
from rag_pipeline.providers.base import EmbeddingProvider
from rag_pipeline.providers.fake import FakeEmbeddingProvider
from rag_pipeline.providers.openai import OpenAIEmbeddingProvider
from rag_pipeline.retrieval.engine import (
    RetrievalConfig,
    RetrievalEngine,
    RetrievalMode,
    RetrievalTrace,
)


def create_embedding_provider(
    settings: Settings,
    provider: EmbeddingProvider | None = None,
) -> EmbeddingProvider:
    """
    Return an explicitly supplied provider, otherwise choose OpenAI when
    a key exists and the deterministic fake provider otherwise.
    """
    if provider is not None:
        return provider

    if settings.openai_api_key:
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )

    return FakeEmbeddingProvider(dimensions=64)



def load_and_chunk_documents(
    corpus_root: Path,
    corpus_version: str,
) -> list[Chunk]:
    documents = []

    for path in sorted(corpus_root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".md",
            ".markdown",
            ".txt",
            ".html",
            ".htm",
            ".pdf",
        }:
            continue

        documents.extend(load_document(path))

    chunk_config = ChunkingConfig(
        corpus_version=corpus_version,
        max_chars=800,
        overlap_chars=120,
    )

    chunker = StructureAwareChunker()
    chunks: list[Chunk] = []

    for document in documents:
        document_chunks = chunker.chunk(
            document=document,
            config=chunk_config,
        )

        for chunk in document_chunks:
            chunks.append(
                chunk.model_copy(
                    update={
                        "metadata": {
                            **chunk.metadata,
                            "source_path": document.source_path,
                            "title": document.title,
                        }
                    }
                )
            )

    return chunks

def build_indexes(
    corpus_root: Path,
    settings: Settings,
    provider: EmbeddingProvider | None = None,
) -> IndexManifest:
    """
    Load, chunk, deduplicate, embed, and persist the corpus indexes.

    The optional provider argument exists mainly for offline tests.
    """
    corpus_root = corpus_root.resolve()

    corpus = build_corpus_manifest(corpus_root)

    chunks = load_and_chunk_documents(
        corpus_root=corpus_root,
        corpus_version=corpus.version_id,
    )

    deduplicated = deduplicate_chunks(chunks)
    chunks = deduplicated.kept

    if not chunks:
        raise ValueError("No chunks available after ingestion")

    embedder = create_embedding_provider(
        settings=settings,
        provider=provider,
    )

    embeddings = embedder.embed(
        [chunk.text for chunk in chunks]
    )

    index_root = settings.index_path / corpus.version_id
    dense_path = index_root / "dense"
    sparse_path = index_root / "sparse"

    dense_index = VectorIndex(
        path=dense_path,
        index_version=corpus.version_id,
    )
    dense_index.upsert(
        chunks=chunks,
        embeddings=embeddings,
    )

    SparseIndex.build(
        chunks=chunks,
        path=sparse_path,
        index_version=corpus.version_id,
    )

    index_manifest = IndexManifest(
        corpus_version=corpus.version_id,
        dense_index_path=str(dense_path.resolve()),
        sparse_index_path=str(sparse_path.resolve()),
        chunk_count=len(chunks),
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        embedding_model=settings.embedding_model,
    )

    index_manifest_path = index_root / "index_manifest.json"
    index_manifest.save(index_manifest_path)

    latest_manifest_path = settings.index_path / "latest_manifest.json"
    index_manifest.save(latest_manifest_path)

    return index_manifest

def load_latest_manifest(settings: Settings) -> IndexManifest:
    manifest_path = settings.index_path / "latest_manifest.json"

    return IndexManifest.load(manifest_path)


def answer_retrieval_query(
    query: str,
    settings: Settings,
    provider: EmbeddingProvider | None = None,
) -> RetrievalTrace:
    """
    Load the latest indexes and return a hybrid retrieval trace.
    """
    manifest = load_latest_manifest(settings)

    dense_index = VectorIndex(
        path=Path(manifest.dense_index_path),
        index_version=manifest.corpus_version,
    )

    sparse_index = SparseIndex.load(
        path=Path(manifest.sparse_index_path),
        index_version=manifest.corpus_version,
    )

    embedder = create_embedding_provider(
        settings=settings,
        provider=provider,
    )

    engine = RetrievalEngine(
        dense_index=dense_index,
        sparse_index=sparse_index,
        embedder=embedder,
    )

    return engine.search(
        query=query,
        mode=RetrievalMode.HYBRID,
        config=RetrievalConfig(
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            final_top_k=settings.final_top_k,
            rrf_k_constant=settings.rrf_k_constant,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
        ),
    )