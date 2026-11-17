from pathlib import Path

from rag_pipeline.config import Settings
from rag_pipeline.pipeline import (
    answer_retrieval_query,
    build_indexes,
)


def test_demo_corpus_can_be_indexed_and_queried(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    document = corpus / "sample.md"
    document.write_text(
        (
            "# Authentication\n\n"
            "Requests are authenticated using API keys.\n\n"
            "## Testing\n\n"
            "Dependencies can be replaced using "
            "app.dependency_overrides."
        ),
        encoding="utf-8",
    )

    settings = Settings(
        openai_api_key=None,
        corpus_path=corpus,
        index_path=tmp_path / "indexes",
    )

    manifest = build_indexes(
        corpus_root=corpus,
        settings=settings,
    )

    assert manifest.chunk_count > 0
    assert manifest.corpus_version

    trace = answer_retrieval_query(
        query="How are requests authenticated?",
        settings=settings,
    )

    assert trace.index_version == (
        manifest.corpus_version
    )
    assert trace.final_results
    assert any(
        "API keys" in result.text
        for result in trace.final_results
    )