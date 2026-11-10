from __future__ import annotations

from pydantic import BaseModel, Field

from rag_pipeline.models import (
    CitationCheck,
    ConfidenceReport,
    GeneratedAnswer,
)


class CitationSupportResponse(BaseModel):
    supported: bool = False
    entailment_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    reason: str = "No semantic judge was used."


class GroundedAnswerResult(BaseModel):
    answer: GeneratedAnswer
    citation_checks: list[CitationCheck] = Field(
        default_factory=list,
    )
    confidence: ConfidenceReport