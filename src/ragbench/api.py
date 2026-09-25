import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ragbench import __version__
from ragbench.chunking import WordChunker
from ragbench.config import Settings, get_settings
from ragbench.embeddings import HashingEmbedder, OpenAIEmbedder
from ragbench.evaluation import evaluate_recall
from ragbench.experiments import BenchmarkRunRepository
from ragbench.generation import DemoGenerator, OpenAIGenerator
from ragbench.ingestion import PDFIngestionError, extract_pdf_documents
from ragbench.models import (
    BenchmarkRun,
    ComponentHealth,
    DocumentInput,
    EvaluationExample,
    EvaluationRequest,
    EvaluationResponse,
    FileIngestResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StorageStatus,
)
from ragbench.observability import configure_observability
from ragbench.service import RAGService
from ragbench.store import HybridVectorStore, InMemoryVectorStore, QdrantVectorStore


def build_service(settings: Settings) -> RAGService:
    if settings.provider == "openai":
        assert settings.openai_api_key is not None
        embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
        generator = OpenAIGenerator(settings.openai_api_key, settings.chat_model)
    else:
        embedder = HashingEmbedder()
        generator = DemoGenerator()
    if settings.store_backend == "qdrant":
        vector_store = QdrantVectorStore(
            embedder,
            path=settings.qdrant_path,
            collection_name=settings.qdrant_collection,
        )
    else:
        vector_store = InMemoryVectorStore(embedder)
    store = HybridVectorStore(vector_store) if settings.retrieval_mode == "hybrid" else vector_store
    return RAGService(
        chunker=WordChunker(settings.chunk_size, settings.chunk_overlap),
        store=store,
        generator=generator,
        default_top_k=settings.top_k,
    )


def load_sample_documents() -> list[DocumentInput]:
    sample_path = files("ragbench").joinpath("data/documents.json")
    return [DocumentInput.model_validate(item) for item in json.loads(sample_path.read_text())]


def load_sample_evaluation_set() -> list[EvaluationExample]:
    sample_path = files("ragbench").joinpath("data/evaluation.json")
    return [EvaluationExample.model_validate(item) for item in json.loads(sample_path.read_text())]


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
    version=__version__,
    description="Measure retrieval quality and inspect every citation behind an answer.",
    lifespan=lifespan,
)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).resolve().parent / "static"),
    name="static",
)
metrics_registry = configure_observability(app)


def get_service() -> RAGService:
    return app.state.rag_service


ServiceDependency = Annotated[RAGService, Depends(get_service)]


def get_run_repository() -> BenchmarkRunRepository:
    return app.state.run_repository


RunRepositoryDependency = Annotated[BenchmarkRunRepository, Depends(get_run_repository)]
PDFUpload = Annotated[UploadFile, File()]
OptionalFormField = Annotated[str | None, Form()]


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    dashboard_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(dashboard_path)


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def _component_check(callback: Callable[[], None]) -> ComponentHealth:
    started = perf_counter()
    callback()
    return ComponentHealth(status="ok", latency_ms=round((perf_counter() - started) * 1_000, 2))


def readiness_status(
    service: RAGService,
    repository: BenchmarkRunRepository,
) -> HealthResponse:
    checks = {
        "vector_store": _component_check(service.store.ping),
        "experiment_store": _component_check(repository.ping),
    }
    return HealthResponse(
        status="ready",
        version=__version__,
        provider=service.generator.provider_name,
        retrieval_mode=app.state.settings.retrieval_mode,
        indexed_chunks=service.store.size,
        checks=checks,
    )


@app.get("/health", response_model=HealthResponse)
@app.get("/health/ready", response_model=HealthResponse)
def readiness(
    service: ServiceDependency,
    repository: RunRepositoryDependency,
) -> HealthResponse:
    try:
        return readiness_status(service, repository)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="A persistence dependency is unavailable",
        ) from exc


@app.get("/v1/system/storage", response_model=StorageStatus)
def storage_status(
    service: ServiceDependency,
    repository: RunRepositoryDependency,
) -> StorageStatus:
    settings = app.state.settings
    return StorageStatus(
        backend=service.store.backend_name,
        persistent=settings.store_backend == "qdrant",
        indexed_chunks=service.store.size,
        benchmark_runs=repository.count(),
        collection=settings.qdrant_collection,
    )


@app.get("/metrics", include_in_schema=False)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        metrics_registry.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/v1/documents", response_model=IngestResponse)
def ingest_documents(
    documents: list[DocumentInput], service: ServiceDependency
) -> IngestResponse:
    if not documents:
        raise HTTPException(status_code=400, detail="At least one document is required")
    return service.ingest(documents)


@app.post("/v1/documents/pdf", response_model=FileIngestResponse)
async def ingest_pdf(
    service: ServiceDependency,
    file: PDFUpload,
    document_id: OptionalFormField = None,
    title: OptionalFormField = None,
) -> FileIngestResponse:
    settings = app.state.settings
    filename = file.filename or "upload.pdf"
    allowed_content_types = {"application/pdf", "application/octet-stream"}
    if file.content_type not in allowed_content_types and not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF uploads are supported",
        )
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PDF exceeds the configured upload limit",
        )
    try:
        documents = extract_pdf_documents(
            filename,
            content,
            document_id=document_id,
            title=title,
        )
    except PDFIngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = service.ingest(documents)
    return FileIngestResponse(
        **result.model_dump(),
        filename=filename,
        pages_processed=len(documents),
    )


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
    result.run_id = repository.save(
        result,
        provider=service.generator.provider_name,
        store_backend=service.store.backend_name,
    )
    return result


@app.get("/v1/evaluation-set", response_model=list[EvaluationExample])
def sample_evaluation_set() -> list[EvaluationExample]:
    return load_sample_evaluation_set()


@app.get("/v1/runs", response_model=list[BenchmarkRun])
def list_runs(repository: RunRepositoryDependency, limit: int = 25) -> list[BenchmarkRun]:
    return repository.list(limit=max(1, min(limit, 100)))
