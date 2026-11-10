from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_pipeline.config import Settings
from rag_pipeline.pipeline import (
    answer_retrieval_query,
    build_indexes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG hybrid retrieval pipeline",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    build_parser = subparsers.add_parser(
        "build-index",
        help="Build dense and sparse indexes",
    )
    build_parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus"),
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Run hybrid retrieval",
    )
    query_parser.add_argument(
        "question",
        type=str,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()

    if args.command == "build-index":
        manifest = build_indexes(
            corpus_root=args.corpus,
            settings=settings,
        )

        print(manifest.model_dump_json(indent=2))
        return

    if args.command == "query":
        trace = answer_retrieval_query(
            query=args.question,
            settings=settings,
        )

        print(
            json.dumps(
                trace.model_dump(mode="json"),
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()