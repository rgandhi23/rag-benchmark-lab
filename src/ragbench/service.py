from time import perf_counter

from ragbench.chunking import WordChunker
from ragbench.generation import Generator
from ragbench.models import Citation, DocumentInput, IngestResponse, QueryResponse
from ragbench.store import VectorStore


class RAGService:
    def __init__(
        self,
        chunker: WordChunker,
        store: VectorStore,
        generator: Generator,
        default_top_k: int = 3,
    ) -> None:
        self.chunker = chunker
        self.store = store
        self.generator = generator
        self.default_top_k = default_top_k

    def ingest(self, documents: list[DocumentInput]) -> IngestResponse:
        chunks = [chunk for document in documents for chunk in self.chunker.split(document)]
        added = self.store.add(chunks)
        return IngestResponse(
            documents_added=len(documents),
            chunks_added=added,
            total_chunks=self.store.size,
        )

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        started = perf_counter()
        results = self.store.search(question, k=top_k or self.default_top_k)
        answer = self.generator.answer(question, results)
        citations = [
            Citation(
                citation_id=index,
                document_id=result.chunk.document_id,
                title=result.chunk.title,
                source=result.chunk.source,
                excerpt=result.chunk.text[:320],
                score=round(result.score, 4),
            )
            for index, result in enumerate(results, start=1)
        ]
        return QueryResponse(
            answer=answer,
            citations=citations,
            provider=self.generator.provider_name,
            latency_ms=round((perf_counter() - started) * 1_000, 2),
        )
