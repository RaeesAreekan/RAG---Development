from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def make_chunk_id(
    corpus_version: str,
    document_id: str,
    chunk_index: int,
) -> str:
    """Create a stable chunk identifier."""
    raw = f"{corpus_version}:{document_id}:{chunk_index}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"chunk-{digest}"


class DocumentSection(BaseModel):
    heading: str | None = None
    text: str
    page_number: int | None = Field(default=None, ge=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_path: str
    source_uri: str | None = None
    title: str | None = None
    content: str
    sections: list[DocumentSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("document_id", "source_path", "content")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int = Field(ge=0)
    section_heading: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    strategy: str
    char_count: int = Field(ge=0)
    token_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "document_id", "text", "strategy")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int = Field(ge=1)
    retrieval_method: str
    raw_score: float | None = None
    source_path: str | None = None
    section_heading: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim text must not be blank")
        return value


class GeneratedAnswer(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None
    model_name: str | None = None


class CitationCheck(BaseModel):
    claim_text: str
    citation_id: str
    citation_exists: bool
    supported: bool
    entailment_score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None


class ConfidenceReport(BaseModel):
    retrieval_score: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    answer_completeness: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    should_abstain: bool
    reasons: list[str] = Field(default_factory=list)


class CorpusVersion(BaseModel):
    version_id: str
    file_hashes: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_metadata: list[dict[str, Any]] = Field(default_factory=list)