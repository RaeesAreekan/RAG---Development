from rag_pipeline.ingestion.normalize import (
    clean_text,
    extract_html_title,
    extract_markdown_sections,
)


def test_clean_text_normalizes_line_endings_and_blank_lines() -> None:
    result = clean_text("hello\r\n\r\n\r\nworld\r\n")

    assert result == "hello\n\nworld"


def test_extract_markdown_sections_preserves_headings() -> None:
    text = """# Authentication

Use the API key.

## Configuration

Timeout defaults to 30 seconds.
"""

    sections = extract_markdown_sections(text)

    assert len(sections) == 2
    assert sections[0].heading == "Authentication"
    assert "API key" in sections[0].text
    assert sections[1].heading == "Configuration"


def test_extract_html_title() -> None:
    html = "<title>Example Documentation</title>"

    assert extract_html_title(html) == "Example Documentation"