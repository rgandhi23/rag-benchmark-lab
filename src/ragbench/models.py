from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)
    source: str | None = None
    page_number: int | None = Field(default=None, ge=1)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    source: str | None = None
    position: int
    page_number: int | None = None


class SearchResult(BaseModel):
    chunk: Chunk
    score: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    citation_id: int
    document_id: str
    title: str
    source: str | None
    excerpt: str
    score: float
    page_number: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    provider: str
    latency_ms: float


class IngestResponse(BaseModel):
    documents_added: int
    chunks_added: int
    total_chunks: int


class FileIngestResponse(IngestResponse):
    filename: str
    pages_processed: int


class EvaluationExample(BaseModel):
    question: str
    relevant_document_ids: list[str] = Field(min_length=1)


class EvaluationRequest(BaseModel):
    name: str = Field(default="ad-hoc", min_length=1, max_length=200)
    examples: list[EvaluationExample] = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=20)


class EvaluationCaseResult(BaseModel):
    question: str
    relevant_document_ids: list[str]
    retrieved_document_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    latency_ms: float


class EvaluationResponse(BaseModel):
    run_id: int | None = None
    name: str = "ad-hoc"
    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    average_latency_ms: float
    cases: list[EvaluationCaseResult]
    k: int


class BenchmarkRun(BaseModel):
    run_id: int
    name: str
    created_at: str
    provider: str
    store_backend: str
    k: int
    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    average_latency_ms: float


class ComponentHealth(BaseModel):
    status: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    retrieval_mode: str
    indexed_chunks: int
    checks: dict[str, ComponentHealth]


class StorageStatus(BaseModel):
    backend: str
    persistent: bool
    indexed_chunks: int
    benchmark_runs: int
    collection: str
