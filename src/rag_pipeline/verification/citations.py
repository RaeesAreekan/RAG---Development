from __future__ import annotations

from collections.abc import Sequence

from rag_pipeline.generation.schemas import CitationSupportResponse
from rag_pipeline.models import (
    CitationCheck,
    GeneratedAnswer,
    RetrievalResult,
)
from rag_pipeline.providers.base import GenerationProvider


def verify_citations(
    answer: GeneratedAnswer,
    evidence: Sequence[RetrievalResult],
    judge: GenerationProvider | None = None,
) -> list[CitationCheck]:
    """
    Verify claim-level citations.

    Without a judge, this performs deterministic citation existence checks.
    With a judge, it also performs semantic support checks.
    """
    evidence_by_id = {
        result.chunk_id: result
        for result in evidence
    }

    checks: list[CitationCheck] = []

    for claim in answer.claims:
        if not claim.citations:
            checks.append(
                CitationCheck(
                    claim_text=claim.text,
                    citation_id="<missing>",
                    citation_exists=False,
                    supported=False,
                    entailment_score=0.0,
                    reason="Claim does not contain a citation.",
                )
            )
            continue

        for citation_id in claim.citations:
            cited_chunk = evidence_by_id.get(citation_id)

            if cited_chunk is None:
                checks.append(
                    CitationCheck(
                        claim_text=claim.text,
                        citation_id=citation_id,
                        citation_exists=False,
                        supported=False,
                        entailment_score=0.0,
                        reason="Citation ID was not found in retrieved evidence.",
                    )
                )
                continue

            if judge is None:
                checks.append(
                    CitationCheck(
                        claim_text=claim.text,
                        citation_id=citation_id,
                        citation_exists=True,
                        supported=True,
                        entailment_score=1.0,
                        reason="Citation exists; semantic judge was not used.",
                    )
                )
                continue

            judge_prompt = f"""
Determine whether the evidence supports the claim.

Claim:
{claim.text}

Evidence:
{cited_chunk.text}

Return supported=true only when the evidence directly supports the claim.
""".strip()

            judgment = judge.generate(
                prompt=judge_prompt,
                response_schema=CitationSupportResponse,
            )

            checks.append(
                CitationCheck(
                    claim_text=claim.text,
                    citation_id=citation_id,
                    citation_exists=True,
                    supported=judgment.supported,
                    entailment_score=judgment.entailment_score,
                    reason=judgment.reason,
                )
            )

    return checks