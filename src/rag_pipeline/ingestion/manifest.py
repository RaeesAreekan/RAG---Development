from __future__ import annotations

import hashlib
from pathlib import Path

from rag_pipeline.models import CorpusVersion


SUPPORTED_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".html",
    ".htm",
    ".pdf",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def build_corpus_manifest(root: Path) -> CorpusVersion:
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")

    file_hashes: dict[str, str] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = path.relative_to(root).as_posix()
        file_hashes[relative_path] = sha256_file(path)

    if not file_hashes:
        raise ValueError(f"No supported documents found in corpus: {root}")

    manifest_text = "\n".join(
        f"{relative_path}:{file_hashes[relative_path]}"
        for relative_path in sorted(file_hashes)
    )

    version_hash = hashlib.sha256(
        manifest_text.encode("utf-8")
    ).hexdigest()[:16]

    return CorpusVersion(
        version_id=f"corpus-{version_hash}",
        file_hashes=file_hashes,
    )