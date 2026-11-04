from __future__ import annotations

from rag_pipeline.chunking.base import Chunker, ChunkingConfig
from rag_pipeline.models import Chunk, Document, make_chunk_id


class FixedChunker:
    """Split normalized document text into overlapping character windows."""

    def chunk(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        if config.overlap_chars >= config.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")

        text = document.content
        chunks: list[Chunk] = []

        start = 0
        chunk_index = 0
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
                            chunk_index,
                        ),
                        document_id=document.document_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        strategy="fixed",
                        char_count=len(chunk_text),
                        metadata={
                            "start_char": start,
                            "end_char": end,
                        },
                    )
                )

                chunk_index += 1

            if end >= len(text):
                break

            start += step

        return chunks