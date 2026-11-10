from __future__ import annotations

from pydantic import BaseModel, Field

from rag_pipeline.models import (
    CitationCheck,
    ConfidenceReport,
    GeneratedAnswer,
)
from rag_pipeline.retrieval.engine import RetrievalTrace


class ConfidenceThresholds(BaseModel):
    retrieval_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
    )
    citation_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )


def _retrieval_score(trace: RetrievalTrace) -> float:
    if not trace.final_results:
        return 0.0

    top_score = max(
        result.score
        for result in trace.final_results
    )

    if trace.mode.value.startswith("hybrid"):
        # RRF scores are generally small, so scale them into a diagnostic
        # range. This is heuristic, not a calibrated probability.
        return max(
            0.0,
            min(
                1.0,
                top_score * trace.config.rrf_k_constant,
            ),
        )

    return max(0.0, min(1.0, top_score))


def _citation_coverage(
    answer: GeneratedAnswer,
    citation_checks: list[CitationCheck],
) -> float:
    if not answer.claims:
        return 0.0

    supported_claims = {
        check.claim_text
        for check in citation_checks
        if check.supported
    }

    return len(supported_claims) / len(answer.claims)


def score_confidence(
    trace: RetrievalTrace,
    answer: GeneratedAnswer,
    citation_checks: list[CitationCheck],
    thresholds: ConfidenceThresholds | None = None,
) -> ConfidenceReport:
    thresholds = thresholds or ConfidenceThresholds()

    retrieval_score = _retrieval_score(trace)
    citation_coverage = _citation_coverage(
        answer=answer,
        citation_checks=citation_checks,
    )
    answer_completeness = (
        1.0
        if answer.answer.strip() and not answer.abstained
        else 0.0
    )

    composite_score = (
        0.4 * retrieval_score
        + 0.4 * citation_coverage
        + 0.2 * answer_completeness
    )

    reasons: list[str] = []

    if retrieval_score < thresholds.retrieval_threshold:
        reasons.append("Retrieved evidence has low confidence.")

    if citation_coverage < thresholds.citation_threshold:
        reasons.append("Citation coverage is below the configured threshold.")

    if not answer.answer.strip():
        reasons.append("Generated answer is empty.")

    should_abstain = bool(reasons)

    return ConfidenceReport(
        retrieval_score=retrieval_score,
        citation_coverage=citation_coverage,
        answer_completeness=answer_completeness,
        composite_score=composite_score,
        should_abstain=should_abstain,
        reasons=reasons,
    )