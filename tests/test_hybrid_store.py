from ragbench.chunking import WordChunker
from ragbench.embeddings import HashingEmbedder
from ragbench.models import DocumentInput
from ragbench.store import HybridVectorStore, InMemoryVectorStore


def test_hybrid_search_uses_lexical_and_vector_ranks() -> None:
    chunks = WordChunker(chunk_size=120, overlap=20).split(
        DocumentInput(
            document_id="atlas",
            title="Atlas",
            text=(
                "Atlas production dashboards track p50 and p95 latency, error rate, "
                "and empty-result rate."
            ),
        )
    )
    chunks += WordChunker(chunk_size=120, overlap=20).split(
        DocumentInput(
            document_id="beacon",
            title="Beacon",
            text="Beacon production monitoring tracks model latency and prediction drift.",
        )
    )
    store = HybridVectorStore(InMemoryVectorStore(HashingEmbedder()))
    store.add(chunks)

    results = store.search("What does the Atlas production dashboard track?", k=1)

    assert results[0].chunk.document_id == "atlas"
    assert store.backend_name == "memory-vector+bm25-rrf"
