from __future__ import annotations

from rag_pipeline.indexing.vector import VectorIndex
from rag_pipeline.models import RetrievalResult
from rag_pipeline.providers.base import EmbeddingProvider


def retrieve_dense(
    query: str,
    index: VectorIndex,
    embedder: EmbeddingProvider,
    top_k: int = 10,
) -> list[RetrievalResult]:
    if not query.strip():
        raise ValueError("query must not be blank")

    query_embedding = embedder.embed([query])[0]

    return index.search(
        query_embedding=query_embedding,
        top_k=top_k,
    )


