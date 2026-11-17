from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from rag_pipeline.config import Settings
from rag_pipeline.generation.schemas import GroundedAnswerResult
from rag_pipeline.generation.service import GenerationService
from rag_pipeline.pipeline import answer_retrieval_query
from rag_pipeline.providers.openai import (
    OpenAIGenerationProvider,
)
from rag_pipeline.retrieval.engine import RetrievalTrace


Retriever = Callable[[str], RetrievalTrace]


@dataclass
class RAGService:
    settings: Settings
    generation_service: GenerationService
    retriever: Retriever

    def ask(
        self,
        question: str,
    ) -> tuple[RetrievalTrace, GroundedAnswerResult]:
        retrieval = self.retriever(question)

        answer = self.generation_service.answer(
            question=question,
            retrieval=retrieval,
        )

        return retrieval, answer


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


def get_rag_service() -> RAGService:
    settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for the live API."
        )

    generation_provider = OpenAIGenerationProvider(
        api_key=settings.openai_api_key,
        model=settings.generation_model,
    )

    generation_service = GenerationService(
        provider=generation_provider,
    )

    def retriever(question: str) -> RetrievalTrace:
        return answer_retrieval_query(
            query=question,
            settings=settings,
        )

    return RAGService(
        settings=settings,
        generation_service=generation_service,
        retriever=retriever,
    )