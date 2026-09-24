import json

import pytest

from ragbench.cli import main
from ragbench.config import get_settings


def test_doctor_explains_active_components(capsys) -> None:
    get_settings.cache_clear()

    main(["doctor"])

    report = json.loads(capsys.readouterr().out)
    assert report["provider"] == "demo"
    assert report["store_backend"] == "memory"
    assert report["retrieval_mode"] == "vector"
    assert report["persistent_vectors"] is False
    assert report["api_key_configured"] is False


def test_query_command_prints_citations(capsys) -> None:
    get_settings.cache_clear()

    main(["query", "What improved Atlas search latency?", "--top-k", "1"])

    response = json.loads(capsys.readouterr().out)
    assert response["citations"][0]["document_id"] == "atlas"


def test_evaluation_gate_exits_when_quality_regresses(capsys) -> None:
    get_settings.cache_clear()

    with pytest.raises(SystemExit) as raised:
        main(["evaluate", "--min-mrr", "1.01"])

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert "Evaluation gate failed" in captured.err
    assert json.loads(captured.out)["mean_reciprocal_rank"] <= 1.0
