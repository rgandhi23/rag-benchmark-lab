import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse

from ragbench.chunking import WordChunker
from ragbench.config import Settings, get_settings
from ragbench.embeddings import HashingEmbedder, OpenAIEmbedder
from ragbench.evaluation import evaluate_recall
from ragbench.experiments import BenchmarkRunRepository
from ragbench.generation import DemoGenerator, OpenAIGenerator
from ragbench.models import (
    BenchmarkRun,
    DocumentInput,
    EvaluationRequest,
    EvaluationResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from ragbench.service import RAGService
from ragbench.store import InMemoryVectorStore, QdrantVectorStore


def build_service(settings: Settings) -> RAGService:
    if settings.provider == "openai":
        assert settings.openai_api_key is not None
        embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
        generator = OpenAIGenerator(settings.openai_api_key, settings.chat_model)
    else:
        embedder = HashingEmbedder()
        generator = DemoGenerator()
    if settings.store_backend == "qdrant":
        store = QdrantVectorStore(
            embedder,
            path=settings.qdrant_path,
            collection_name=settings.qdrant_collection,
        )
    else:
        store = InMemoryVectorStore(embedder)
    return RAGService(
        chunker=WordChunker(settings.chunk_size, settings.chunk_overlap),
        store=store,
        generator=generator,
        default_top_k=settings.top_k,
    )


def load_sample_documents() -> list[DocumentInput]:
    sample_path = Path(__file__).resolve().parents[2] / "data" / "sample" / "documents.json"
    if not sample_path.exists():
        return []
    return [DocumentInput.model_validate(item) for item in json.loads(sample_path.read_text())]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    service = build_service(settings)
    samples = load_sample_documents()
    if samples:
        service.ingest(samples)
    app.state.rag_service = service
    app.state.settings = settings
    app.state.run_repository = BenchmarkRunRepository(settings.experiment_db_path)
    try:
        yield
    finally:
        service.store.close()
        app.state.run_repository.close()


app = FastAPI(
    title="RAG Benchmark Lab",
    version="0.1.0",
    description="Measure retrieval quality and inspect every citation behind an answer.",
    lifespan=lifespan,
)


def get_service() -> RAGService:
    return app.state.rag_service


ServiceDependency = Annotated[RAGService, Depends(get_service)]


def get_run_repository() -> BenchmarkRunRepository:
    return app.state.run_repository


RunRepositoryDependency = Annotated[BenchmarkRunRepository, Depends(get_run_repository)]


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    dashboard_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(dashboard_path)


@app.get("/health")
def health(service: ServiceDependency) -> dict[str, str | int]:
    return {
        "status": "ok",
        "provider": service.generator.provider_name,
        "indexed_chunks": service.store.size,
    }


@app.post("/v1/documents", response_model=IngestResponse)
def ingest_documents(
    documents: list[DocumentInput], service: ServiceDependency
) -> IngestResponse:
    if not documents:
        raise HTTPException(status_code=400, detail="At least one document is required")
    return service.ingest(documents)


@app.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest, service: ServiceDependency) -> QueryResponse:
    return service.query(request.question, request.top_k)


@app.post("/v1/evaluate", response_model=EvaluationResponse)
def evaluate(
    request: EvaluationRequest,
    service: ServiceDependency,
    repository: RunRepositoryDependency,
) -> EvaluationResponse:
    result = evaluate_recall(service.store, request.examples, request.k, request.name)
    settings: Settings = app.state.settings
    result.run_id = repository.save(
        result,
        provider=service.generator.provider_name,
        store_backend=settings.store_backend,
    )
    return result


@app.get("/v1/runs", response_model=list[BenchmarkRun])
def list_runs(repository: RunRepositoryDependency, limit: int = 25) -> list[BenchmarkRun]:
    return repository.list(limit=max(1, min(limit, 100)))
