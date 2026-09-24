import hashlib
import re

from ragbench.models import Chunk, DocumentInput


class WordChunker:
    """Dependency-free word-window chunker with stable chunk identifiers."""

    def __init__(self, chunk_size: int = 120, overlap: int = 20) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: DocumentInput) -> list[Chunk]:
        words = re.findall(r"\S+", document.text.strip())
        if not words:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        for position, start in enumerate(range(0, len(words), step)):
            window = words[start : start + self.chunk_size]
            if not window:
                break
            text = " ".join(window)
            identity = f"{document.document_id}:{position}:{text}"
            if document.page_number is not None:
                identity = f"{document.document_id}:page-{document.page_number}:{position}:{text}"
            digest = hashlib.sha1(identity.encode(), usedforsecurity=False).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}-{digest}",
                    document_id=document.document_id,
                    title=document.title,
                    text=text,
                    source=document.source,
                    position=position,
                    page_number=document.page_number,
                )
            )
            if start + self.chunk_size >= len(words):
                break
        return chunks
