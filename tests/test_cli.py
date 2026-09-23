import json

from ragbench.cli import main
from ragbench.config import get_settings


def test_doctor_explains_active_components(capsys) -> None:
    get_settings.cache_clear()

    main(["doctor"])

    report = json.loads(capsys.readouterr().out)
    assert report["provider"] == "demo"
    assert report["store_backend"] == "memory"
    assert report["persistent_vectors"] is False
    assert report["api_key_configured"] is False


def test_query_command_prints_citations(capsys) -> None:
    get_settings.cache_clear()

    main(["query", "What improved Atlas search latency?", "--top-k", "1"])

    response = json.loads(capsys.readouterr().out)
    assert response["citations"][0]["document_id"] == "atlas"
