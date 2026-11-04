from __future__ import annotations

import re

from rag_pipeline.models import DocumentSection


def clean_text(text: str) -> str:
    """Normalize line endings and excessive blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.rstrip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    blank_seen = False

    for line in lines:
        if line.strip():
            cleaned_lines.append(line)
            blank_seen = False
        elif not blank_seen:
            cleaned_lines.append("")
            blank_seen = True

    return "\n".join(cleaned_lines).strip()


def extract_markdown_sections(text: str) -> list[DocumentSection]:
    """Extract Markdown sections while preserving heading hierarchy."""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    sections: list[DocumentSection] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush_section() -> None:
        section_text = clean_text("\n".join(current_lines))

        if section_text:
            sections.append(
                DocumentSection(
                    heading=current_heading,
                    text=section_text,
                )
            )

    for line in text.splitlines():
        match = heading_pattern.match(line)

        if match:
            flush_section()
            current_heading = match.group(2)
            current_lines = []
        else:
            current_lines.append(line)

    flush_section()

    return sections


def extract_html_title(text: str) -> str | None:
    match = re.search(
        r"<title[^>]*>(.*?)</title>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return None

    return clean_text(match.group(1))