from __future__ import annotations

from rag_pipeline.chunking.base import ChunkingConfig
from rag_pipeline.models import Chunk, Document, DocumentSection, make_chunk_id


class StructureAwareChunker:
    """Prefer document sections, splitting oversized sections when necessary."""

    def chunk(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        if not document.sections:
            return self._chunk_without_sections(document, config)

        chunks: list[Chunk] = []

        for section in document.sections:
            section_chunks = self._split_section(section, config)

            for section_text in section_chunks:
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(
                            config.corpus_version,
                            document.document_id,
                            len(chunks),
                        ),
                        document_id=document.document_id,
                        text=section_text,
                        chunk_index=len(chunks),
                        section_heading=section.heading,
                        page_number=section.page_number,
                        strategy="structure-aware",
                        char_count=len(section_text),
                        metadata={
                            "section_heading": section.heading,
                        },
                    )
                )

        return chunks

    def _split_section(
        self,
        section: DocumentSection,
        config: ChunkingConfig,
    ) -> list[str]:
        text = section.text.strip()

        if len(text) <= config.max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        step = config.max_chars - config.overlap_chars

        while start < len(text):
            end = min(start + config.max_chars, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(chunk_text)

            if end >= len(text):
                break

            start += step

        return chunks

    def _chunk_without_sections(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        text = document.content
        chunks: list[Chunk] = []

        start = 0
        step = config.max_chars - config.overlap_chars

        while start < len(text):
            end = min(start + config.max_chars, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(
                            config.corpus_version,
                            document.document_id,
                            len(chunks),
                        ),
                        document_id=document.document_id,
                        text=chunk_text,
                        chunk_index=len(chunks),
                        strategy="structure-aware",
                        char_count=len(chunk_text),
                    )
                )

            if end >= len(text):
                break

            start += step

        return chunks