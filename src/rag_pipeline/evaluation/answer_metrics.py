from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from rag_pipeline.evaluation.schema import EvaluationCase
from rag_pipeline.models import (
    CitationCheck,
    GeneratedAnswer,
)


class AnswerMetricResult(BaseModel):
    case_count: int = Field(ge=0)

    abstention_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    citation_support_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    citation_coverage: float = Field(
        ge=0.0,
        le=1.0,
    )

    lexical_answer_f1: float = Field(
        ge=0.0,
        le=1.0,
    )


def _tokens(text: str) -> list[str]:
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def _token_f1(
    prediction: str,
    reference: str,
) -> float:
    prediction_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)

    if not prediction_tokens or not reference_tokens:
        return 0.0

    prediction_counts: dict[str, int] = {}

    for token in prediction_tokens:
        prediction_counts[token] = (
            prediction_counts.get(token, 0) + 1
        )

    reference_counts: dict[str, int] = {}

    for token in reference_tokens:
        reference_counts[token] = (
            reference_counts.get(token, 0) + 1
        )

    overlap = 0

    for token, count in prediction_counts.items():
        overlap += min(
            count,
            reference_counts.get(token, 0),
        )

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


def compute_answer_metrics(
    cases: Sequence[EvaluationCase],
    answers: Mapping[str, GeneratedAnswer],
    citation_checks: Mapping[str, Sequence[CitationCheck]],
) -> AnswerMetricResult:
    """
    Compute answer and citation metrics.

    lexical_answer_f1 is only a baseline signal. It should not be treated
    as semantic correctness.
    """
    if not cases:
        raise ValueError("At least one evaluation case is required.")

    abstention_matches = 0
    total_citation_checks = 0
    supported_citations = 0
    total_claims = 0
    claims_with_supported_citation = 0
    lexical_scores: list[float] = []

    for case in cases:
        answer = answers.get(case.case_id)

        if answer is None:
            raise ValueError(
                f"Missing generated answer for {case.case_id}"
            )

        checks = list(
            citation_checks.get(
                case.case_id,
                [],
            )
        )

        if answer.abstained == case.expected_abstention:
            abstention_matches += 1

        total_citation_checks += len(checks)
        supported_citations += sum(
            1
            for check in checks
            if check.citation_exists and check.supported
        )

        total_claims += len(answer.claims)

        for claim in answer.claims:
            claim_checks = [
                check
                for check in checks
                if check.claim_text == claim.text
            ]

            if any(
                check.citation_exists and check.supported
                for check in claim_checks
            ):
                claims_with_supported_citation += 1

        if (
            case.reference_answer
            and not case.expected_abstention
        ):
            lexical_scores.append(
                _token_f1(
                    prediction=answer.answer,
                    reference=case.reference_answer,
                )
            )

    citation_support_rate = (
        supported_citations / total_citation_checks
        if total_citation_checks
        else 0.0
    )

    citation_coverage = (
        claims_with_supported_citation / total_claims
        if total_claims
        else 0.0
    )

    lexical_answer_f1 = (
        sum(lexical_scores) / len(lexical_scores)
        if lexical_scores
        else 0.0
    )

    return AnswerMetricResult(
        case_count=len(cases),
        abstention_accuracy=(
            abstention_matches / len(cases)
        ),
        citation_support_rate=citation_support_rate,
        citation_coverage=citation_coverage,
        lexical_answer_f1=lexical_answer_f1,
    )