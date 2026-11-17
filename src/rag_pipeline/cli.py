from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_pipeline.config import Settings
from rag_pipeline.pipeline import (
    answer_retrieval_query,
    build_indexes,
    load_latest_manifest,
)

from rag_pipeline.evaluation.runner import run_evaluation
from rag_pipeline.generation.service import GenerationService
from rag_pipeline.providers.openai import OpenAIGenerationProvider

from rag_pipeline.retrieval.engine import RetrievalMode
from rag_pipeline.retrieval.rerank import KeywordOverlapReranker

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
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run the evaluation suite",
    )

    evaluate_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evals/golden.jsonl"),
    )

    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evals/reports"),
    )

    evaluate_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/evals/cache"),
    )

    evaluate_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable evaluation caching",
    )

    compare_parser = subparsers.add_parser(
        "compare-eval",
        help="Compare retrieval strategies",
    )

    compare_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evals/golden.jsonl"),
    )

    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evals/comparison"),
    )

    compare_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/evals/cache"),
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
    if args.command == "evaluate":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for live evaluation."
            )

        generation_provider = OpenAIGenerationProvider(
            api_key=settings.openai_api_key,
            model=settings.evaluation_model,
        )

        generation_service = GenerationService(
            provider=generation_provider,
        )

        def retrieve(question: str): # type: ignore
            return answer_retrieval_query(
                query=question,
                settings=settings,
            )

        def answer(question: str, trace):
            return generation_service.answer(
                question=question,
                retrieval=trace,
            )

        report = run_evaluation(
            dataset=args.dataset,
            retriever=retrieve,
            answerer=answer,
            cache_dir=(
                None
                if args.no_cache
                else args.cache_dir
            ),
            run_metadata={
                "corpus_version": (
                    load_latest_manifest(settings)
                    .corpus_version
                ),
                "retrieval_mode": "hybrid",
                "embedding_model": settings.embedding_model,
                "generation_model": settings.generation_model,
                "evaluation_model": settings.evaluation_model,
                "dense_top_k": settings.dense_top_k,
                "sparse_top_k": settings.sparse_top_k,
                "final_top_k": settings.final_top_k,
                "dense_weight": settings.dense_weight,
                "sparse_weight": settings.sparse_weight,
            },
        )

        json_path = args.output_dir / "evaluation.json"
        markdown_path = args.output_dir / "evaluation.md"

        report.write_json(json_path)
        report.write_markdown(markdown_path)

        print(
            f"Evaluation complete.\n"
            f"JSON report: {json_path}\n"
            f"Markdown report: {markdown_path}"
        )

        return
    if args.command == "compare-eval":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for comparison evaluation."
            )

        generation_provider = OpenAIGenerationProvider(
            api_key=settings.openai_api_key,
            model=settings.evaluation_model,
        )

        generation_service = GenerationService(
            provider=generation_provider,
        )

        variants = {
            "dense": {
                "mode": RetrievalMode.DENSE,
                "reranker": None,
            },
            "sparse": {
                "mode": RetrievalMode.SPARSE,
                "reranker": None,
            },
            "hybrid": {
                "mode": RetrievalMode.HYBRID,
                "reranker": None,
            },
            "hybrid-reranked": {
                "mode": RetrievalMode.HYBRID_RERANKED,
                "reranker": KeywordOverlapReranker(),
            },
        }

        comparison: dict[str, dict] = {}

        for variant_name, variant_config in variants.items():
            mode = variant_config["mode"]
            reranker = variant_config["reranker"]

            def retrieve(
                question: str,
                selected_mode=mode,
                selected_reranker=reranker,
            ):
                return answer_retrieval_query(
                    query=question,
                    settings=settings,
                    mode=selected_mode,
                    reranker=selected_reranker,
                )

            def answer(question: str, trace):
                return generation_service.answer(
                    question=question,
                    retrieval=trace,
                )

            report = run_evaluation(
                dataset=args.dataset,
                retriever=retrieve,
                answerer=answer,
                cache_dir=args.cache_dir / variant_name,
                run_metadata={
                    "corpus_version": (
                        load_latest_manifest(settings)
                        .corpus_version
                    ),
                    "retrieval_mode": variant_name,
                    "embedding_model": settings.embedding_model,
                    "generation_model": settings.generation_model,
                    "evaluation_model": settings.evaluation_model,
                    "dense_top_k": settings.dense_top_k,
                    "sparse_top_k": settings.sparse_top_k,
                    "final_top_k": settings.final_top_k,
                },
            )

            variant_output_dir = (
                args.output_dir / variant_name
            )

            report.write_json(
                variant_output_dir / "evaluation.json"
            )

            report.write_markdown(
                variant_output_dir / "evaluation.md"
            )

            comparison[variant_name] = {
                "retrieval_metrics": (
                    report.retrieval_metrics.model_dump()
                ),
                "answer_metrics": (
                    report.answer_metrics.model_dump()
                ),
                "run_metadata": report.run_metadata,
            }

        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        comparison_path = (
            args.output_dir / "comparison.json"
        )

        comparison_path.write_text(
            json.dumps(
                comparison,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"Comparison complete.\n"
            f"Report: {comparison_path}"
        )

        return


if __name__ == "__main__":
    main()