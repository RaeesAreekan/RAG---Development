from pathlib import Path
from typing import Protocol

from rag_pipeline.models import Document


class DocumentLoader(Protocol):
    def supported_extensions(self) -> set[str]:
        ...

    def load(self, path: Path) -> list[Document]:
        ...