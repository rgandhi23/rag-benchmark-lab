from ragbench.chunking import WordChunker
from ragbench.embeddings import HashingEmbedder
from ragbench.evaluation import evaluate_recall
from ragbench.generation import DemoGenerator
from ragbench.models import DocumentInput, EvaluationExample
from ragbench.service import RAGService
from ragbench.store import InMemoryVectorStore


def make_service() -> RAGService:
    return RAGService(
        WordChunker(chunk_size=40, overlap=5),
        InMemoryVectorStore(HashingEmbedder()),
        DemoGenerator(),
    )


def test_query_returns_ranked_citation() -> None:
    service = make_service()
    service.ingest(
        [
            DocumentInput(
                document_id="latency",
                title="Latency",
                text="Embedding request batching and caching reduced search latency.",
            ),
            DocumentInput(
                document_id="security",
                title="Security",
                text="Role based access control protects confidential documents.",
            ),
        ]
    )

    response = service.query("How was search latency reduced?", top_k=1)

    assert response.provider == "demo"
    assert response.citations[0].document_id == "latency"
    assert response.citations[0].score > 0


def test_recall_at_k() -> None:
    service = make_service()
    service.ingest(
        [
            DocumentInput(document_id="a", title="A", text="GPU inference optimization"),
            DocumentInput(document_id="b", title="B", text="Healthcare data validation"),
        ]
    )

    result = evaluate_recall(
        service.store,
        [EvaluationExample(question="GPU optimization", relevant_document_ids=["a"])],
        k=1,
    )

    assert result.hit_rate_at_k == 1.0
    assert result.precision_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.mean_reciprocal_rank == 1.0
