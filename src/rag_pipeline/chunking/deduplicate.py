from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from rag_pipeline.models import Chunk


class DeduplicationResult(BaseModel):
    kept: list[Chunk]
    exact_duplicate_ids: list[str]
    near_duplicate_ids: list[str]


def _normalized_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have equal dimensions")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return sum(a * b for a, b in zip(left, right)) / (
        left_norm * right_norm
    )


def deduplicate_chunks(
    chunks: Sequence[Chunk],
    near_duplicate_threshold: float | None = None,
    embed: Callable[[Sequence[str]], list[list[float]]] | None = None,
    max_near_duplicate_comparisons: int = 10_000,
) -> DeduplicationResult:
    """
    Remove exact duplicates and optionally bounded near duplicates.

    Exact duplicate detection is deterministic and always enabled.
    Near-duplicate detection is optional.
    """
    kept: list[Chunk] = []
    exact_duplicate_ids: list[str] = []
    near_duplicate_ids: list[str] = []

    seen_hashes: set[str] = set()

    for chunk in chunks:
        content_hash = _normalized_hash(chunk.text)

        if content_hash in seen_hashes:
            exact_duplicate_ids.append(chunk.chunk_id)
            continue

        seen_hashes.add(content_hash)
        kept.append(chunk)

    if (
        near_duplicate_threshold is None
        or embed is None
        or len(kept) < 2
    ):
        return DeduplicationResult(
            kept=kept,
            exact_duplicate_ids=exact_duplicate_ids,
            near_duplicate_ids=near_duplicate_ids,
        )

    texts = [chunk.text for chunk in kept]
    embeddings = embed(texts)

    comparison_count = 0
    remove_ids: set[str] = set()

    for index, current_chunk in enumerate(kept):
        for previous_index in range(index):
            if comparison_count >= max_near_duplicate_comparisons:
                break

            comparison_count += 1

            similarity = _cosine_similarity(
                embeddings[index],
                embeddings[previous_index],
            )

            if similarity >= near_duplicate_threshold:
                remove_ids.add(current_chunk.chunk_id)
                near_duplicate_ids.append(current_chunk.chunk_id)
                break

    final_kept = [
        chunk for chunk in kept
        if chunk.chunk_id not in remove_ids
    ]

    return DeduplicationResult(
        kept=final_kept,
        exact_duplicate_ids=exact_duplicate_ids,
        near_duplicate_ids=near_duplicate_ids,
    )