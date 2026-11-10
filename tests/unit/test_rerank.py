from rag_pipeline.models import RetrievalResult
from rag_pipeline.retrieval.rerank import KeywordOverlapReranker


def make_result(chunk_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        score=0.1,
        rank=1,
        retrieval_method="hybrid-rrf",
    )


def test_keyword_reranker_prioritizes_term_overlap() -> None:
    candidates = [
        make_result(
            "weak",
            "Deployment requires a health endpoint.",
        ),
        make_result(
            "strong",
            "API keys authenticate API requests.",
        ),
    ]

    reranker = KeywordOverlapReranker()

    results = reranker.rerank(
        query="API keys authenticate requests",
        candidates=candidates,
        top_k=2,
    )

    assert results[0].chunk_id == "strong"
    assert results[0].retrieval_method == "reranked"
    assert results[0].rank == 1


def test_reranker_respects_top_k() -> None:
    candidates = [
        make_result(str(index), f"Candidate {index}")
        for index in range(5)
    ]

    results = KeywordOverlapReranker().rerank(
        query="candidate",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2