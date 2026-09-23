from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from ragbench.embeddings import Embedder
from ragbench.models import Chunk, SearchResult


class VectorStore(Protocol):
    @property
    def size(self) -> int: ...

    def add(self, chunks: list[Chunk]) -> int: ...

    def search(self, query: str, k: int = 3) -> list[SearchResult]: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...


@dataclass
class _IndexedChunk:
    chunk: Chunk
    vector: np.ndarray


class InMemoryVectorStore:
    """Small transparent vector store used as the baseline benchmark implementation."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._items: list[_IndexedChunk] = []

    @property
    def size(self) -> int:
        return len(self._items)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        self._items.extend(
            _IndexedChunk(chunk=chunk, vector=np.asarray(vector, dtype=np.float32))
            for chunk, vector in zip(chunks, vectors, strict=True)
        )
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        if not self._items:
            return []
        query_vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector)) or 1.0
        scored: list[SearchResult] = []
        for item in self._items:
            item_norm = float(np.linalg.norm(item.vector)) or 1.0
            score = float(np.dot(query_vector, item.vector) / (query_norm * item_norm))
            scored.append(SearchResult(chunk=item.chunk, score=score))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:k]

    def clear(self) -> None:
        self._items.clear()

    def close(self) -> None:
        return None


class QdrantVectorStore:
    """Persistent local Qdrant store using cosine distance and document payloads."""

    def __init__(self, embedder: Embedder, path: str, collection_name: str) -> None:
        self.embedder = embedder
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))
        self.collection_name = collection_name

    @property
    def size(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return int(self.client.count(self.collection_name, exact=True).count)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]),
                    distance=models.Distance.COSINE,
                ),
            )
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload={"chunk": chunk.model_dump(mode="json")},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        if not self.client.collection_exists(self.collection_name):
            return []
        query_vector = self.embedder.embed([query])[0]
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        return [
            SearchResult(
                chunk=Chunk.model_validate(point.payload["chunk"]),
                score=float(point.score),
            )
            for point in response.points
            if point.payload and "chunk" in point.payload
        ]

    def clear(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def close(self) -> None:
        self.client.close()
