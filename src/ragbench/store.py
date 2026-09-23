import math
import re
from collections import Counter
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
    def backend_name(self) -> str: ...

    @property
    def size(self) -> int: ...

    def add(self, chunks: list[Chunk]) -> int: ...

    def search(self, query: str, k: int = 3) -> list[SearchResult]: ...

    def all_chunks(self) -> list[Chunk]: ...

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
    def backend_name(self) -> str:
        return "memory-vector"

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

    def all_chunks(self) -> list[Chunk]:
        return [item.chunk for item in self._items]

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
    def backend_name(self) -> str:
        return "qdrant-vector"

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

    def all_chunks(self) -> list[Chunk]:
        if not self.client.collection_exists(self.collection_name):
            return []
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(
                Chunk.model_validate(point.payload["chunk"])
                for point in points
                if point.payload and "chunk" in point.payload
            )
            if offset is None:
                break
        return chunks

    def clear(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def close(self) -> None:
        self.client.close()


class HybridVectorStore:
    """Fuse BM25 lexical ranks with an existing vector store using weighted RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        rrf_k: int = 60,
        lexical_weight: float = 1.25,
        vector_weight: float = 1.0,
    ) -> None:
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self._chunks = {chunk.chunk_id: chunk for chunk in vector_store.all_chunks()}

    @property
    def backend_name(self) -> str:
        return f"{self.vector_store.backend_name}+bm25-rrf"

    @property
    def size(self) -> int:
        return self.vector_store.size

    def add(self, chunks: list[Chunk]) -> int:
        added = self.vector_store.add(chunks)
        self._chunks.update((chunk.chunk_id, chunk) for chunk in chunks)
        return added

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _lexical_search(self, query: str) -> list[SearchResult]:
        chunks = list(self._chunks.values())
        if not chunks:
            return []
        tokenized = [self._tokens(chunk.text) for chunk in chunks]
        query_terms = Counter(self._tokens(query))
        average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
        document_frequency = Counter(
            token for tokens in tokenized for token in set(tokens) if token in query_terms
        )
        scored: list[SearchResult] = []
        for chunk, tokens in zip(chunks, tokenized, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for token, query_frequency in query_terms.items():
                frequency = frequencies[token]
                if frequency == 0:
                    continue
                document_count = len(chunks)
                frequency_count = document_frequency[token]
                inverse_document_frequency = math.log(
                    1 + (document_count - frequency_count + 0.5) / (frequency_count + 0.5)
                )
                length_normalization = 1.5 * (
                    1 - 0.75 + 0.75 * len(tokens) / average_length
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * 2.5
                    / (frequency + length_normalization)
                    * query_frequency
                )
            if score > 0:
                scored.append(SearchResult(chunk=chunk, score=score))
        return sorted(scored, key=lambda result: result.score, reverse=True)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        candidate_limit = max(k * 4, 10)
        vector_results = self.vector_store.search(query, k=candidate_limit)
        lexical_results = self._lexical_search(query)
        fused_scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for weight, results in (
            (self.vector_weight, vector_results),
            (self.lexical_weight, lexical_results),
        ):
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.chunk_id
                chunks[chunk_id] = result.chunk
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + weight / (
                    self.rrf_k + rank
                )
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]
        return [
            SearchResult(chunk=chunks[chunk_id], score=fused_scores[chunk_id])
            for chunk_id in ranked_ids
        ]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def clear(self) -> None:
        self.vector_store.clear()
        self._chunks.clear()

    def close(self) -> None:
        self.vector_store.close()
