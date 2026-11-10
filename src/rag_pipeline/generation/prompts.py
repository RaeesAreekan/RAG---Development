from __future__ import annotations

from collections.abc import Sequence

from rag_pipeline.models import RetrievalResult


def build_grounded_prompt(
    question: str,
    evidence: Sequence[RetrievalResult],
) -> str:
    if not question.strip():
        raise ValueError("question must not be blank")

    if not evidence:
        raise ValueError("evidence must not be empty")

    evidence_blocks: list[str] = []

    for result in evidence:
        source = result.source_path or "unknown source"
        section = result.section_heading or "unknown section"

        evidence_blocks.append(
            "\n".join(
                [
                    f"[{result.chunk_id}]",
                    f"Source: {source}",
                    f"Section: {section}",
                    f"Content:",
                    result.text,
                ]
            )
        )

    context = "\n\n".join(evidence_blocks)

    return f"""
You are a grounded documentation assistant.

Answer the user's question using only the evidence blocks below.

Rules:
1. Do not use outside knowledge.
2. Every material claim must include one or more citation IDs.
3. Citation IDs must exactly match the IDs in the evidence.
4. If the evidence is insufficient, set "abstained" to true.
5. If abstained is true, explain why in "abstention_reason".
6. Do not invent source names, chunk IDs, or facts.
7. Return a structured response with:
   - answer
   - claims
   - abstained
   - abstention_reason
   - model_name

User question:
{question}

Evidence:
{context}
""".strip()