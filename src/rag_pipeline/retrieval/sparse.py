from __future__ import annotations

from rag_pipeline.indexing.sparse import SparseIndex
from rag_pipeline.models import RetrievalResult


def retrieve_sparse(
    query: str,
    index: SparseIndex,
    top_k: int = 10,
) -> list[RetrievalResult]:
    if not query.strip():
        raise ValueError("query must not be blank")

    return index.search(
        query=query,
        top_k=top_k,
    )