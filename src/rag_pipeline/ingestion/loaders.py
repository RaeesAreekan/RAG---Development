from __future__ import annotations

import hashlib
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from rag_pipeline.ingestion.normalize import (
    clean_text,
    extract_html_title,
    extract_markdown_sections,
)
from rag_pipeline.models import Document, DocumentSection


def _document_id(path: Path) -> str:
    digest = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()
    return f"doc-{digest[:16]}"


def _build_document(
    path: Path,
    content: str,
    title: str | None = None,
    sections: list[DocumentSection] | None = None,
) -> Document:
    normalized_content = clean_text(content)

    if not normalized_content:
        raise ValueError(f"Document is empty after normalization: {path}")

    return Document(
        document_id=_document_id(path),
        source_path=path.as_posix(),
        title=title,
        content=normalized_content,
        sections=sections or [],
    )


def _load_markdown_or_text(path: Path) -> Document:
    content = path.read_text(encoding="utf-8")
    normalized = clean_text(content)

    sections = (
        extract_markdown_sections(normalized)
        if path.suffix.lower() in {".md", ".markdown"}
        else []
    )

    title = sections[0].heading if sections else path.stem

    return _build_document(
        path=path,
        content=normalized,
        title=title,
        sections=sections,
    )


def _load_html(path: Path) -> Document:
    raw_html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    title = extract_html_title(raw_html)

    heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
    heading_text = heading.get_text(" ", strip=True) if heading else None

    text = soup.get_text("\n", strip=True)
    normalized = clean_text(text)

    sections = (
        [DocumentSection(heading=heading_text, text=normalized)]
        if normalized
        else []
    )

    return _build_document(
        path=path,
        content=normalized,
        title=title,
        sections=sections,
    )


def _load_pdf(path: Path) -> Document:
    reader = PdfReader(str(path))

    page_texts: list[str] = []
    sections: list[DocumentSection] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")

        if not text:
            continue

        page_texts.append(text)
        sections.append(
            DocumentSection(
                heading=None,
                text=text,
                page_number=page_number,
            )
        )

    return _build_document(
        path=path,
        content="\n\n".join(page_texts),
        title=path.stem,
        sections=sections,
    )


def load_document(path: Path) -> list[Document]:
    """Load one supported file into normalized document objects."""
    suffix = path.suffix.lower()

    if suffix in {".md", ".markdown", ".txt"}:
        return [_load_markdown_or_text(path)]

    if suffix in {".html", ".htm"}:
        return [_load_html(path)]

    if suffix == ".pdf":
        return [_load_pdf(path)]

    raise ValueError(f"Unsupported document extension: {suffix}")