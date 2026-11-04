from __future__ import annotations

import hashlib
from collections.abc import Sequence

from pydantic import BaseModel

from rag_pipeline.models import Claim, GeneratedAnswer


class FakeEmbeddingProvider:
    """Deterministic embedding provider for tests and offline development."""

    def __init__(self, dimensions: int = 32) -> None:
        if dimensions < 2:
            raise ValueError("dimensions must be at least 2")

        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0

        while len(values) < self.dimensions:
            digest = hashlib.sha256(
                f"{text}:{counter}".encode("utf-8")
            ).digest()

            for byte in digest:
                values.append((byte / 255.0) * 2.0 - 1.0)

                if len(values) == self.dimensions:
                    break

            counter += 1

        return values


class FakeGenerationProvider:
    """Deterministic generation provider that never calls a model."""

    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        if response_schema is GeneratedAnswer:
            return GeneratedAnswer(
                answer="This is a deterministic test answer.",
                claims=[
                    Claim(
                        text="This is a deterministic test claim.",
                        citations=[],
                    )
                ],
                abstained=False,
                model_name="fake",
            )

        return response_schema.model_validate({})