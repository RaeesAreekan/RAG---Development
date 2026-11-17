from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from rag_pipeline.evaluation.answer_metrics import AnswerMetricResult
from rag_pipeline.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
)
from rag_pipeline.evaluation.schema import RetrievalMetricResult
from rag_pipeline.generation.schemas import GroundedAnswerResult

from typing import Any

class EvaluationCaseResult(BaseModel):
    case_id: str
    retrieval_metrics: RetrievalMetricResult
    grounded_answer: GroundedAnswerResult


class EvaluationReport(BaseModel):
    dataset_path: str
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    case_count: int = Field(ge=0)
    retrieval_metrics: RetrievalMetricResult
    answer_metrics: AnswerMetricResult
    cases: list[EvaluationCaseResult] = Field(
        default_factory=list
    )

    run_metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def write_markdown(self, path: Path) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        lines = [
            "# RAG Evaluation Report",
            "",
            f"- Dataset: `{self.dataset_path}`",
            f"- Evaluated at: `{self.evaluated_at.isoformat()}`",
            f"- Cases: `{self.case_count}`",
            "",
            "## Retrieval metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            (
                f"| MRR | "
                f"{self.retrieval_metrics.reciprocal_rank:.3f} |"
            ),
        ]

        for k, value in (
            self.retrieval_metrics.recall_at_k.items()
        ):
            lines.append(
                f"| Recall@{k} | {value:.3f} |"
            )

        for k, value in (
            self.retrieval_metrics.precision_at_k.items()
        ):
            lines.append(
                f"| Precision@{k} | {value:.3f} |"
            )

        for k, value in (
            self.retrieval_metrics.ndcg_at_k.items()
        ):
            lines.append(
                f"| nDCG@{k} | {value:.3f} |"
            )

        lines.extend(
            [
                "",
                "## Answer metrics",
                "",
                "| Metric | Value |",
                "|---|---:|",
                (
                    f"| Abstention accuracy | "
                    f"{self.answer_metrics.abstention_accuracy:.3f} |"
                ),
                (
                    f"| Citation support rate | "
                    f"{self.answer_metrics.citation_support_rate:.3f} |"
                ),
                (
                    f"| Citation coverage | "
                    f"{self.answer_metrics.citation_coverage:.3f} |"
                ),
                (
                    f"| Lexical answer F1 | "
                    f"{self.answer_metrics.lexical_answer_f1:.3f} |"
                ),
                "",
                "## Per-case results",
                "",
            ]
        )

        for case in self.cases:
            answer = case.grounded_answer.answer

            lines.extend(
                [
                    f"### {case.case_id}",
                    "",
                    f"- Abstained: `{answer.abstained}`",
                    (
                        "- Confidence: "
                        f"`{case.grounded_answer.confidence.composite_score:.3f}`"
                    ),
                    (
                        "- Retrieval MRR: "
                        f"`{case.retrieval_metrics.reciprocal_rank:.3f}`"
                    ),
                    "",
                ]
            )
        if self.run_metadata:
            lines.extend(
                [
                    "",
                    "## Run metadata",
                    "",
                ]
            )

            for key, value in self.run_metadata.items():
                lines.append(
                    f"- {key}: `{value}`"
                )

        path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )