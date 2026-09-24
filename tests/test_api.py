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
    assert health.json()["version"] == "0.2.0"
    assert health.json()["indexed_chunks"] == 3
    assert health.json()["retrieval_mode"] == "vector"
    assert evaluation_set.status_code == 200
    assert len(evaluation_set.json()) == 9
    assert response.status_code == 200
    assert response.json()["citations"][0]["document_id"] == "harbor"


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
