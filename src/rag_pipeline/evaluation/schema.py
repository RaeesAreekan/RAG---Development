from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
import json
from pathlib import Path


QuestionType = Literal[
    "lookup",
    "exact-keyword",
    "exact-identifier",
    "multi-hop",
    "ambiguous",
    "unanswerable",
]


class EvaluationCase(BaseModel):
    case_id: str
    question: str
    question_type: QuestionType

    reference_answer: str | None = None

    # Chunk ID -> graded relevance from 0 to 3.
    # 0 = irrelevant
    # 1 = weakly relevant
    # 2 = useful evidence
    # 3 = essential evidence
    graded_chunk_judgments: dict[str, int] = Field(
        default_factory=dict,
    )

    acceptable_source_refs: list[str] = Field(
        default_factory=list,
    )

    # Used especially for multi-hop questions.
    required_evidence_units: list[str] = Field(
        default_factory=list,
    )

    expected_abstention: bool = False
    review_status: Literal[
        "draft",
        "llm-proposed",
        "human-approved",
    ] = "draft"

    reviewer_notes: str | None = None

    def validate_relevance_grades(self) -> "EvaluationCase":
        for chunk_id, grade in self.graded_chunk_judgments.items():
            if not 0 <= grade <= 3:
                raise ValueError(
                    f"Invalid relevance grade for {chunk_id}: {grade}"
                )

        return self
    def validate_for_benchmark(self) -> "EvaluationCase":
        """
        Validate that this case is safe to use in benchmark scoring.

        LLM-proposed and draft cases must not accidentally enter benchmark runs.
        """
        self.validate_relevance_grades()

        if self.review_status != "human-approved":
            raise ValueError(
                f"Case {self.case_id} is not human-approved: "
                f"{self.review_status}"
            )

        if self.expected_abstention:
            if self.graded_chunk_judgments:
                raise ValueError(
                    f"Unanswerable case {self.case_id} should not have "
                    "positive relevance judgments."
                )
            return self

        if not self.reference_answer:
            raise ValueError(
                f"Answerable case {self.case_id} needs a reference_answer."
            )

        if not self.graded_chunk_judgments:
            raise ValueError(
                f"Answerable case {self.case_id} needs human-graded chunks."
            )

        if not any(
            grade > 0
            for grade in self.graded_chunk_judgments.values()
        ):
            raise ValueError(
                f"Answerable case {self.case_id} needs at least one "
                "relevant chunk."
            )

        if not self.required_evidence_units:
            raise ValueError(
                f"Answerable case {self.case_id} needs required evidence units."
            )

        return self


class RetrievalMetricResult(BaseModel):
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    reciprocal_rank: float
    ndcg_at_k: dict[int, float]


class EvaluationDataset(BaseModel):
    dataset_version: str
    corpus_version: str
    cases: list[EvaluationCase]


def load_evaluation_cases(
    path: Path,
) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
                case = EvaluationCase.model_validate(payload)
                case.validate_relevance_grades()
                cases.append(case)
            except Exception as exc:
                raise ValueError(
                    f"Invalid evaluation case on line {line_number}"
                ) from exc

    if not cases:
        raise ValueError("Evaluation dataset is empty")

    return cases

def load_benchmark_cases(
    path: Path,
) -> list[EvaluationCase]:
    """
    Load only cases that are ready for benchmark scoring.
    """
    cases = load_evaluation_cases(path)

    seen_ids: set[str] = set()

    for case in cases:
        if case.case_id in seen_ids:
            raise ValueError(
                f"Duplicate evaluation case ID: {case.case_id}"
            )

        seen_ids.add(case.case_id)
        case.validate_for_benchmark()

    return cases