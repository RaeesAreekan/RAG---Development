from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rag_pipeline.generation.schemas import GroundedAnswerResult
from rag_pipeline.retrieval.engine import RetrievalTrace


class CachedEvaluationCase(BaseModel):
    case_id: str
    question: str
    retrieval_trace: RetrievalTrace
    grounded_answer: GroundedAnswerResult


def make_cache_key(
    case_id: str,
    question: str,
    metadata: dict[str, Any],
) -> str:
    payload = {
        "case_id": case_id,
        "question": question,
        "metadata": metadata,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def load_cached_case(
    cache_dir: Path,
    cache_key: str,
) -> CachedEvaluationCase | None:
    path = cache_dir / f"{cache_key}.json"

    if not path.exists():
        return None

    return CachedEvaluationCase.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def save_cached_case(
    cache_dir: Path,
    cache_key: str,
    case: CachedEvaluationCase,
) -> None:
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = cache_dir / f"{cache_key}.json"

    path.write_text(
        case.model_dump_json(indent=2),
        encoding="utf-8",
    )