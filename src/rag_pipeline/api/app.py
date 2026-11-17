from __future__ import annotations
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from rag_pipeline.api.dependencies import (
    RAGService,
    get_rag_service,
    get_settings,
)
from rag_pipeline.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RetrievalSummary,
    DocumentListResponse,
    DocumentSummary,
)
from rag_pipeline.config import Settings
from rag_pipeline.pipeline import load_latest_manifest

from rag_pipeline.ingestion.loaders import load_document
from rag_pipeline.ingestion.manifest import (
    SUPPORTED_EXTENSIONS,
    build_corpus_manifest,
)

app = FastAPI(
    title="RAG Hybrid Search API",
    version="0.1.0",
    description=(
        "OpenAI-first RAG API with hybrid retrieval, "
        "grounded answers, citations, confidence, and abstention."
    ),
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    try:
        manifest = load_latest_manifest(settings)

    except FileNotFoundError:
        return HealthResponse(
            status="degraded",
            index_loaded=False,
            corpus_version=None,
        )

    return HealthResponse(
        status="ok",
        index_loaded=True,
        corpus_version=manifest.corpus_version,
    )


@app.post(
    "/v1/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    try:
        retrieval, answer = service.ask(
            request.question
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The retrieval index is not available.",
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return AskResponse(
        question=request.question,
        answer=answer,
        retrieval=RetrievalSummary(
            mode=retrieval.mode,
            index_version=retrieval.index_version,
            results=retrieval.final_results,
        ),
    )


@app.get(
    "/v1/documents",
    response_model=DocumentListResponse,
)
def list_documents(
    settings: Settings = Depends(get_settings),
) -> DocumentListResponse:
    corpus_root = settings.corpus_path.resolve()

    if not corpus_root.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Corpus directory does not exist: "
                f"{corpus_root}"
            ),
        )

    try:
        corpus = build_corpus_manifest(
            corpus_root
        )

        documents = []

        for path in sorted(
            corpus_root.rglob("*")
        ):
            if not path.is_file():
                continue

            if path.suffix.lower() not in (
                SUPPORTED_EXTENSIONS
            ):
                continue

            loaded_documents = load_document(
                path.resolve()
            )

            for document in loaded_documents:
                relative_path = (
                    Path(document.source_path)
                    .resolve()
                    .relative_to(corpus_root)
                    .as_posix()
                )

                documents.append(
                    DocumentSummary(
                        document_id=document.document_id,
                        source_path=relative_path,
                        title=document.title,
                        sections=[
                            section.heading
                            for section in document.sections
                            if section.heading
                        ],
                    )
                )

        return DocumentListResponse(
            corpus_version=corpus.version_id,
            documents=documents,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc