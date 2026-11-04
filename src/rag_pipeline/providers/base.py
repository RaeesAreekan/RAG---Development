from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class GenerationProvider(Protocol):
    def generate(
        self,
        prompt: str,
        response_schema: type[ResponseModel],
    ) -> ResponseModel:
        ...