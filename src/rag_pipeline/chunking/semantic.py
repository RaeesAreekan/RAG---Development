from __future__ import annotations

import math
import re

from rag_pipeline.chunking.base import TextEmbedder, ChunkingConfig
from rag_pipeline.models import Chunk, Document, make_chunk_id


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have equal dimensions")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    return dot_product / (left_norm * right_norm)


class SemanticChunker:
    """
    Split on paragraph boundaries when adjacent paragraphs become
    semantically dissimilar.
    """

    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder

    def chunk(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        paragraphs = self._paragraphs(document.content)

        if not paragraphs:
            return []

        embeddings = self.embedder.embed(paragraphs)
        groups: list[list[str]] = [[paragraphs[0]]]

        for index in range(1, len(paragraphs)):
            similarity = cosine_similarity(
                embeddings[index - 1],
                embeddings[index],
            )

            current_group = groups[-1]
            candidate = "\n\n".join([*current_group, paragraphs[index]])

            if (
                similarity >= config.semantic_similarity_threshold
                and len(candidate) <= config.max_chars
            ):
                current_group.append(paragraphs[index])
            else:
                groups.append([paragraphs[index]])

        chunks: list[Chunk] = []

        for index, group in enumerate(groups):
            text = "\n\n".join(group)

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(
                        config.corpus_version,
                        document.document_id,
                        index,
                    ),
                    document_id=document.document_id,
                    text=text,
                    chunk_index=index,
                    strategy="semantic",
                    char_count=len(text),
                )
            )

        return chunks

    @staticmethod
    def _paragraphs(text: str) -> list[str]:
        return [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ]