from fastapi.testclient import TestClient

from ragbench.api import app


def test_health_and_query() -> None:
    with TestClient(app) as client:
        dashboard = client.get("/")
        health = client.get("/health")
        evaluation_set = client.get("/v1/evaluation-set")
        response = client.post(
            "/v1/query", json={"question": "What security controls does Harbor use?", "top_k": 2}
        )

    assert dashboard.status_code == 200
    assert "Retrieval quality you can inspect" in dashboard.text
    assert "How one question travels through the pipeline" in dashboard.text
    assert health.status_code == 200
    assert health.json()["version"] == "0.3.1"
    assert health.json()["indexed_chunks"] == 3
    assert health.json()["retrieval_mode"] == "vector"
    assert health.json()["status"] == "ready"
    assert set(health.json()["checks"]) == {"vector_store", "experiment_store"}
    assert evaluation_set.status_code == 200
    assert len(evaluation_set.json()) == 9
    assert response.status_code == 200
    assert response.json()["citations"][0]["document_id"] == "harbor"


def test_liveness_and_storage_diagnostics() -> None:
    with TestClient(app) as client:
        live = client.get("/health/live")
        storage = client.get("/v1/system/storage")

    assert live.status_code == 200
    assert live.json() == {"status": "ok", "version": "0.3.1"}
    assert storage.status_code == 200
    assert storage.json()["backend"] == "memory-vector"
    assert storage.json()["persistent"] is False
    assert storage.json()["indexed_chunks"] == 3


def test_readiness_returns_503_when_persistence_fails(monkeypatch) -> None:
    def fail_ping() -> None:
        raise RuntimeError("storage unavailable")

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.rag_service.store, "ping", fail_ping)
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "A persistence dependency is unavailable"


def test_request_id_and_prometheus_metrics() -> None:
    with TestClient(app) as client:
        health = client.get("/health/live", headers={"X-Request-ID": "test-request-123"})
        metrics = client.get("/metrics")

    assert health.headers["X-Request-ID"] == "test-request-123"
    assert metrics.status_code == 200
    assert "ragbench_http_requests_total" in metrics.text
    assert 'method="GET",path="/health/live"' in metrics.text


def test_evaluation_is_persisted() -> None:
    payload = {
        "name": "api-test",
        "k": 3,
        "examples": [
            {
                "question": "What improved Atlas search latency?",
                "relevant_document_ids": ["atlas"],
            }
        ],
    }
    with TestClient(app) as client:
        evaluation = client.post("/v1/evaluate", json=payload)
        runs = client.get("/v1/runs")

    assert evaluation.status_code == 200
    assert evaluation.json()["run_id"] is not None
    assert evaluation.json()["mean_reciprocal_rank"] == 1.0
    assert runs.status_code == 200
    assert runs.json()[0]["name"] == "api-test"


def test_pdf_endpoint_rejects_invalid_content() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/documents/pdf",
            files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not a valid PDF"
