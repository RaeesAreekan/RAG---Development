from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rag_pipeline.indexing.sparse import SparseIndex
from rag_pipeline.indexing.vector import VectorIndex
from rag_pipeline.models import RetrievalResult
from rag_pipeline.providers.base import EmbeddingProvider
from rag_pipeline.retrieval.dense import retrieve_dense
from rag_pipeline.retrieval.fusion import rrf_fuse
from rag_pipeline.retrieval.rerank import Reranker
from rag_pipeline.retrieval.sparse import retrieve_sparse


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANKED = "hybrid-reranked"


class RetrievalConfig(BaseModel):
    dense_top_k: int = Field(default=10, ge=1)
    sparse_top_k: int = Field(default=10, ge=1)
    final_top_k: int = Field(default=5, ge=1)
    rrf_k_constant: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=0.7, ge=0.0)
    sparse_weight: float = Field(default=0.3, ge=0.0)


class RetrievalTrace(BaseModel):
    query: str
    mode: RetrievalMode
    dense_results: list[RetrievalResult] = Field(default_factory=list)
    sparse_results: list[RetrievalResult] = Field(default_factory=list)
    fused_results: list[RetrievalResult] = Field(default_factory=list)
    final_results: list[RetrievalResult] = Field(default_factory=list)
    index_version: str
    config: RetrievalConfig
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalEngine:
    def __init__(
        self,
        dense_index: VectorIndex,
        sparse_index: SparseIndex,
        embedder: EmbeddingProvider,
        reranker: Reranker | None = None,
    ) -> None:
        if dense_index.index_version != sparse_index.index_version:
            raise ValueError(
                "Dense and sparse indexes must use the same version"
            )

        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.embedder = embedder
        self.reranker = reranker

    def search(
        self,
        query: str,
        mode: RetrievalMode,
        config: RetrievalConfig | None = None,
    ) -> RetrievalTrace:
        if not query.strip():
            raise ValueError("query must not be blank")

        config = config or RetrievalConfig()

        dense_results: list[RetrievalResult] = []
        sparse_results: list[RetrievalResult] = []
        fused_results: list[RetrievalResult] = []

        if mode in {
            RetrievalMode.DENSE,
            RetrievalMode.HYBRID,
            RetrievalMode.HYBRID_RERANKED,
        }:
            dense_results = retrieve_dense(
                query=query,
                index=self.dense_index,
                embedder=self.embedder,
                top_k=config.dense_top_k,
            )

        if mode in {
            RetrievalMode.SPARSE,
            RetrievalMode.HYBRID,
            RetrievalMode.HYBRID_RERANKED,
        }:
            sparse_results = retrieve_sparse(
                query=query,
                index=self.sparse_index,
                top_k=config.sparse_top_k,
            )

        if mode == RetrievalMode.DENSE:
            final_results = dense_results[:config.final_top_k]

        elif mode == RetrievalMode.SPARSE:
            final_results = sparse_results[:config.final_top_k]

        else:
            fused_results = rrf_fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k_constant=config.rrf_k_constant,
                dense_weight=config.dense_weight,
                sparse_weight=config.sparse_weight,
            )

            final_results = fused_results[:config.final_top_k]

            if mode == RetrievalMode.HYBRID_RERANKED:
                if self.reranker is None:
                    raise ValueError(
                        "A reranker is required for hybrid-reranked mode"
                    )

                final_results = self.reranker.rerank(
                    query=query,
                    candidates=fused_results,
                    top_k=config.final_top_k,
                )

        return RetrievalTrace(
            query=query,
            mode=mode,
            dense_results=dense_results,
            sparse_results=sparse_results,
            fused_results=fused_results,
            final_results=final_results,
            index_version=self.dense_index.index_version,
            config=config,
        )