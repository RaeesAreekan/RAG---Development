from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field

from rag_pipeline.models import Chunk
from rag_pipeline.providers.base import GenerationProvider
from rag_pipeline.evaluation.schema import EvaluationCase


class AnnotationBatch(BaseModel):
    """
    Structured response expected from the annotation model.
    """

    cases: list[EvaluationCase] = Field(
        default_factory=list,
    )


def build_annotation_prompt(
    chunks: Sequence[Chunk],
    number_of_cases: int,
) -> str:
    """
    Build a prompt that asks an LLM to propose evaluation cases.

    The output is only a proposal. A human must review every case.
    """
    if not chunks:
        raise ValueError("At least one chunk is required.")

    if number_of_cases < 1:
        raise ValueError("number_of_cases must be positive.")

    evidence_blocks: list[str] = []

    for chunk in chunks:
        source_path = chunk.metadata.get(
            "source_path",
            "unknown-source",
        )

        evidence_blocks.append(
            "\n".join(
                [
                    f"CHUNK_ID: {chunk.chunk_id}",
                    f"SOURCE_PATH: {source_path}",
                    f"SECTION: {chunk.section_heading or ''}",
                    "TEXT:",
                    chunk.text,
                ]
            )
        )

    evidence = "\n\n---\n\n".join(evidence_blocks)

    return f"""
You are helping create a human-reviewed evaluation dataset for a RAG system.

Generate exactly {number_of_cases} diverse evaluation cases using only the
provided chunks.

Include a mixture of:

- lookup questions;
- exact keyword questions;
- exact identifier questions;
- multi-hop questions when the evidence supports them;
- ambiguous questions when the evidence is genuinely ambiguous;
- unanswerable questions when the requested information is absent.

Rules:

1. Do not invent facts.
2. Every answerable question must have a concise reference_answer.
3. Every answerable question must identify one or more relevant chunk IDs.
4. Use relevance grades:
   - 0 = irrelevant;
   - 1 = weakly relevant;
   - 2 = useful evidence;
   - 3 = essential evidence.
5. Every question must include required_evidence_units.
6. Unanswerable questions must set expected_abstention=true.
7. Set review_status to "llm-proposed".
8. The human reviewer will verify every question, answer, source, and grade.

Available evidence:

{evidence}
""".strip()


def propose_annotation_cases(
    chunks: Sequence[Chunk],
    provider: GenerationProvider,
    number_of_cases: int = 10,
) -> list[EvaluationCase]:
    """
    Generate LLM-proposed cases.

    These cases are deliberately marked as llm-proposed and cannot be used
    directly in benchmark scoring.
    """
    prompt = build_annotation_prompt(
        chunks=chunks,
        number_of_cases=number_of_cases,
    )

    response = provider.generate(
        prompt=prompt,
        response_schema=AnnotationBatch,
    )

    proposed_cases: list[EvaluationCase] = []

    for case in response.cases:
        case = case.model_copy(
            update={
                "review_status": "llm-proposed",
            }
        )

        case.validate_relevance_grades()
        proposed_cases.append(case)

    return proposed_cases


def write_annotation_queue(
    cases: Sequence[EvaluationCase],
    path: Path,
) -> None:
    """
    Write proposed cases to an editable JSONL review queue.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(
                json.dumps(
                    case.model_dump(mode="json"),
                    ensure_ascii=False,
                )
            )
            file.write("\n")