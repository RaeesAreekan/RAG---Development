from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, repr=False)

    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o"
    evaluation_model: str = "gpt-4o-mini"

    corpus_path: Path = Path("data/corpus")
    processed_data_path: Path = Path("data/processed")
    index_path: Path = Path("data/indexes")

    dense_top_k: int = Field(default=10, ge=1)
    sparse_top_k: int = Field(default=10, ge=1)
    final_top_k: int = Field(default=5, ge=1)

    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    rrf_k_constant: int = Field(default=60, ge=1)

    abstention_retrieval_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
    )
    abstention_citation_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()