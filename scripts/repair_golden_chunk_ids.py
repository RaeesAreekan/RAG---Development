from __future__ import annotations

import json
import re
from pathlib import Path

from rag_pipeline.chunking.base import ChunkingConfig
from rag_pipeline.chunking.structure import StructureAwareChunker
from rag_pipeline.ingestion.loaders import load_document
from rag_pipeline.ingestion.manifest import build_corpus_manifest


ROOT = Path(".").resolve()
CORPUS_ROOT = ROOT / "data" / "corpus"
INPUT_PATH = ROOT / "data" / "evals" / "golden.jsonl"
OUTPUT_PATH = ROOT / "data" / "evals" / "golden.repaired.jsonl"


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def build_current_chunk_map() -> dict[tuple[str, str], list[str]]:
    corpus = build_corpus_manifest(CORPUS_ROOT)

    documents = []

    for path in sorted(CORPUS_ROOT.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".md",
            ".markdown",
            ".txt",
            ".html",
            ".htm",
            ".pdf",
        }:
            continue

        documents.extend(
            load_document(path.resolve())
        )

    chunker = StructureAwareChunker()
    chunk_map: dict[tuple[str, str], list[str]] = {}

    config = ChunkingConfig(
        corpus_version=corpus.version_id,
        max_chars=800,
        overlap_chars=120,
    )

    for document in documents:
        chunks = chunker.chunk(
            document=document,
            config=config,
        )

        for chunk in chunks:
            relative_source = (
                Path(document.source_path)
                .resolve()
                .relative_to(CORPUS_ROOT)
                .as_posix()
            )

            section_slug = slugify(
                chunk.section_heading or ""
            )

            key = (
                relative_source,
                section_slug,
            )

            chunk_map.setdefault(
                key,
                [],
            ).append(chunk.chunk_id)

    return chunk_map


def main() -> None:
    chunk_map = build_current_chunk_map()

    repaired_cases = []

    lines = INPUT_PATH.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line.strip():
            continue

        case = json.loads(line)

        if not case["graded_chunk_judgments"]:
            repaired_cases.append(case)
            continue

        source_refs = case.get(
            "acceptable_source_refs",
            [],
        )

        old_grades = list(
            case["graded_chunk_judgments"].values()
        )

        new_judgments: dict[str, int] = {}

        for index, source_ref in enumerate(source_refs):
            if "#" not in source_ref:
                raise ValueError(
                    f"Line {line_number}: source reference "
                    f"has no section anchor: {source_ref}"
                )

            source_path, anchor = source_ref.split(
                "#",
                maxsplit=1,
            )

            key = (
                source_path,
                anchor,
            )

            current_chunk_ids = chunk_map.get(key)

            if not current_chunk_ids:
                raise ValueError(
                    f"Line {line_number}: no current chunk found "
                    f"for {source_ref}"
                )

            if index < len(old_grades):
                grade = old_grades[index]
            else:
                grade = 3

            for chunk_id in current_chunk_ids:
                new_judgments[chunk_id] = grade

        case["graded_chunk_judgments"] = new_judgments
        repaired_cases.append(case)

    OUTPUT_PATH.write_text(
        "\n".join(
            json.dumps(
                case,
                ensure_ascii=False,
            )
            for case in repaired_cases
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote repaired dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()