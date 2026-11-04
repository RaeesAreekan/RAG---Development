from __future__ import annotations

from typing import Protocol, Sequence

from pydantic import BaseModel, Field

from rag_pipeline.models import Chunk, Document


class ChunkingConfig(BaseModel):
    corpus_version: str
    max_chars: int = Field(default=800, ge=50)
    overlap_chars: int = Field(default=120, ge=0)
    semantic_similarity_threshold: float = Field(
        default=0.72,
        ge=-1.0,
        le=1.0,
    )


class Chunker(Protocol):
    def chunk(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        ...


class TextEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...