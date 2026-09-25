import json
import tarfile

import pytest

from ragbench.backup import BackupError, create_backup
from ragbench.config import Settings
from ragbench.experiments import BenchmarkRunRepository


def test_backup_archives_vectors_database_and_manifest(tmp_path) -> None:
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    (qdrant_path / "storage.sqlite").write_bytes(b"vector-state")
    database_path = tmp_path / "experiments.sqlite3"
    repository = BenchmarkRunRepository(str(database_path))
    repository.close()
    output = tmp_path / "backup.tar.gz"
    settings = Settings(
        store_backend="qdrant",
        qdrant_path=str(qdrant_path),
        experiment_db_path=str(database_path),
    )

    create_backup(settings, output)

    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
        manifest_file = archive.extractfile("manifest.json")
        assert manifest_file is not None
        manifest = json.load(manifest_file)
    assert "qdrant/storage.sqlite" in names
    assert "experiments.sqlite3" in names
    assert manifest["app_version"] == "0.3.0"
    assert manifest["contains_vectors"] is True


def test_backup_requires_persistent_backend(tmp_path) -> None:
    settings = Settings(store_backend="memory", experiment_db_path=":memory:")

    with pytest.raises(BackupError, match="persistent Qdrant"):
        create_backup(settings, tmp_path / "backup.tar.gz")
