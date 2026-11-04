from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class IndexManifest(BaseModel):
    corpus_version: str
    dense_index_path: str
    sparse_index_path: str
    chunk_count: int = Field(ge=0)
    chunk_ids: list[str]
    embedding_model: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        return cls.model_validate_json(
            path.read_text(encoding="utf-8")
        )