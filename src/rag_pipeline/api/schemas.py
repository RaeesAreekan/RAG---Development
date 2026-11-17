from __future__ import annotations

from pydantic import BaseModel, Field

from rag_pipeline.generation.schemas import GroundedAnswerResult
from rag_pipeline.models import RetrievalResult
from rag_pipeline.retrieval.engine import RetrievalMode


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
    )


class RetrievalSummary(BaseModel):
    mode: RetrievalMode
    index_version: str
    results: list[RetrievalResult] = Field(
        default_factory=list
    )


class AskResponse(BaseModel):
    question: str
    answer: GroundedAnswerResult
    retrieval: RetrievalSummary


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    corpus_version: str | None = None


class DocumentSummary(BaseModel):
    document_id: str
    source_path: str
    title: str | None = None
    sections: list[str] = Field(
        default_factory=list
    )


class DocumentListResponse(BaseModel):
    corpus_version: str
    documents: list[DocumentSummary] = Field(
        default_factory=list
    )