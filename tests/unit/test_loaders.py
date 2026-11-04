from pathlib import Path

from rag_pipeline.ingestion.loaders import load_document

from rag_pipeline.ingestion.manifest import build_corpus_manifest
FIXTURE_DIR = Path("tests/fixtures/docs")


def test_load_markdown_document() -> None:
    documents = load_document(FIXTURE_DIR / "example.md")

    assert len(documents) == 1
    assert documents[0].source_path.endswith("example.md")
    assert documents[0].title == "Authentication"
    assert len(documents[0].sections) == 2
    assert "API_KEY" in documents[0].content


def test_load_text_document() -> None:
    documents = load_document(FIXTURE_DIR / "example.txt")

    assert len(documents) == 1
    assert "retry configuration" in documents[0].content


def test_load_html_document() -> None:
    documents = load_document(FIXTURE_DIR / "example.html")

    assert len(documents) == 1
    assert documents[0].title == "Example HTML Documentation"
    assert "production configuration" in documents[0].content


def test_unsupported_extension_raises_error(tmp_path: Path) -> None:
    unsupported = tmp_path / "example.csv"
    unsupported.write_text("a,b,c", encoding="utf-8")

    try:
        load_document(unsupported)
    except ValueError as exc:
        assert "Unsupported document extension" in str(exc)
    else:
        raise AssertionError("Expected unsupported extension error")

def test_build_corpus_manifest_is_deterministic(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    (corpus_dir / "a.md").write_text(
        "# Document A\n\nContent A",
        encoding="utf-8",
    )
    (corpus_dir / "b.txt").write_text(
        "Content B",
        encoding="utf-8",
    )

    first = build_corpus_manifest(corpus_dir)
    second = build_corpus_manifest(corpus_dir)

    assert first.version_id == second.version_id
    assert set(first.file_hashes) == {"a.md", "b.txt"}


def test_manifest_ignores_unsupported_files(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    (corpus_dir / "document.md").write_text(
        "Supported document",
        encoding="utf-8",
    )
    (corpus_dir / "ignored.csv").write_text(
        "a,b,c",
        encoding="utf-8",
    )

    manifest = build_corpus_manifest(corpus_dir)

    assert "document.md" in manifest.file_hashes
    assert "ignored.csv" not in manifest.file_hashes