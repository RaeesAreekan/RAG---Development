from rag_pipeline.models import RetrievalResult
from rag_pipeline.retrieval.fusion import (
    rrf_fuse,
    weighted_score_fuse,
)


def result(
    chunk_id: str,
    rank: int,
    score: float,
    method: str,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=f"Text for {chunk_id}",
        score=score,
        rank=rank,
        retrieval_method=method,
    )


def test_rrf_fusion_rewards_chunks_present_in_both_rankings() -> None:
    dense = [
        result("a", rank=1, score=0.9, method="dense"),
        result("b", rank=2, score=0.8, method="dense"),
    ]

    sparse = [
        result("b", rank=1, score=8.0, method="sparse"),
        result("c", rank=2, score=7.0, method="sparse"),
    ]

    fused = rrf_fuse(
        dense_results=dense,
        sparse_results=sparse,
        k_constant=1,
        dense_weight=0.5,
        sparse_weight=0.5,
    )

    assert fused[0].chunk_id == "b"
    assert fused[0].retrieval_method == "hybrid-rrf"
    assert fused[0].rank == 1


def test_rrf_fusion_deduplicates_same_chunk() -> None:
    dense = [
        result("same", rank=1, score=0.9, method="dense"),
    ]

    sparse = [
        result("same", rank=1, score=12.0, method="sparse"),
    ]

    fused = rrf_fuse(
        dense_results=dense,
        sparse_results=sparse,
        k_constant=60,
    )

    assert len(fused) == 1
    assert fused[0].chunk_id == "same"


def test_weighted_score_fusion_combines_raw_scores() -> None:
    dense = [
        result("a", rank=1, score=0.8, method="dense"),
    ]

    sparse = [
        result("a", rank=1, score=10.0, method="sparse"),
    ]

    fused = weighted_score_fuse(
        dense_results=dense,
        sparse_results=sparse,
        dense_weight=0.5,
        sparse_weight=0.5,
    )

    assert fused[0].score == 5.4
    assert fused[0].retrieval_method == "hybrid-weighted"