from __future__ import annotations

from rag_pipeline.generation.prompts import build_grounded_prompt
from rag_pipeline.generation.schemas import GroundedAnswerResult
from rag_pipeline.models import GeneratedAnswer
from rag_pipeline.providers.base import GenerationProvider
from rag_pipeline.retrieval.engine import RetrievalTrace
from rag_pipeline.verification.citations import verify_citations
from rag_pipeline.verification.confidence import (
    ConfidenceThresholds,
    score_confidence,
)


class GenerationService:
    def __init__(
        self,
        provider: GenerationProvider,
        confidence_thresholds: ConfidenceThresholds | None = None,
    ) -> None:
        self.provider = provider
        self.confidence_thresholds = (
            confidence_thresholds
            or ConfidenceThresholds()
        )

    def answer(
        self,
        question: str,
        retrieval: RetrievalTrace,
    ) -> GroundedAnswerResult:
        if not question.strip():
            raise ValueError("question must not be blank")

        if not retrieval.final_results:
            answer = GeneratedAnswer(
                answer="",
                claims=[],
                abstained=True,
                abstention_reason="No evidence was retrieved.",
                model_name="none",
            )

            confidence = score_confidence(
                trace=retrieval,
                answer=answer,
                citation_checks=[],
                thresholds=self.confidence_thresholds,
            )

            return GroundedAnswerResult(
                answer=answer,
                citation_checks=[],
                confidence=confidence,
            )

        prompt = build_grounded_prompt(
            question=question,
            evidence=retrieval.final_results,
        )

        generated = self.provider.generate(
            prompt=prompt,
            response_schema=GeneratedAnswer,
        )

        citation_checks = verify_citations(
            answer=generated,
            evidence=retrieval.final_results,
        )

        confidence = score_confidence(
            trace=retrieval,
            answer=generated,
            citation_checks=citation_checks,
            thresholds=self.confidence_thresholds,
        )

        if confidence.should_abstain:
            generated = generated.model_copy(
                update={
                    "abstained": True,
                    "abstention_reason": " ".join(
                        confidence.reasons
                    ),
                }
            )

        return GroundedAnswerResult(
            answer=generated,
            citation_checks=citation_checks,
            confidence=confidence,
        )