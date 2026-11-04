from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from rag_pipeline.providers.base import EmbeddingProvider, GenerationProvider


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider using the embeddings endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be blank")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input cannot contain blank text")

        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts),
            encoding_format="float",
        )

        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]


class OpenAIGenerationProvider:
    """OpenAI structured-output generation provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be blank")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        response_schema: type[ResponseModel],
    ) -> ResponseModel:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied context. "
                        "Return the requested structured format."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            text_format=response_schema,
        )

        parsed = response.output_parsed

        if parsed is None:
            raise RuntimeError("OpenAI returned no structured output")

        return parsed