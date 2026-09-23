from ragbench.embeddings import HashingEmbedder
from ragbench.models import Chunk
from ragbench.store import QdrantVectorStore


def test_qdrant_store_persists_between_instances(tmp_path) -> None:
    path = tmp_path / "qdrant"
    chunk = Chunk(
        chunk_id="atlas-1",
        document_id="atlas",
        title="Atlas",
        text="Batching and caching reduced retrieval latency.",
        position=0,
    )
    first = QdrantVectorStore(HashingEmbedder(), str(path), "test_chunks")
    first.add([chunk])
    first.close()

    second = QdrantVectorStore(HashingEmbedder(), str(path), "test_chunks")
    results = second.search("retrieval latency", k=1)

    assert second.size == 1
    assert results[0].chunk.document_id == "atlas"
    second.close()
