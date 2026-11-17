from __future__ import annotations

from rag_pipeline.config import Settings
from rag_pipeline.pipeline import build_indexes


def main() -> None:
    settings = Settings.from_env()

    manifest = build_indexes(
        corpus_root=settings.corpus_path,
        settings=settings,
    )

    print("Demo corpus indexed successfully.")
    print(f"Corpus version: {manifest.corpus_version}")
    print(f"Chunk count: {manifest.chunk_count}")
    print(f"Dense index: {manifest.dense_index_path}")
    print(f"Sparse index: {manifest.sparse_index_path}")


if __name__ == "__main__":
    main()