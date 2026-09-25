import json
import sqlite3
import tarfile
import tempfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from ragbench import __version__
from ragbench.config import Settings


class BackupError(RuntimeError):
    """Raised when there is no persistent state available to back up."""


def create_backup(settings: Settings, output: Path) -> Path:
    """Create a portable archive of local Qdrant vectors and SQLite experiment history."""
    if settings.store_backend != "qdrant":
        raise BackupError("Backups require the persistent Qdrant backend")

    qdrant_path = Path(settings.qdrant_path)
    database_path = Path(settings.experiment_db_path)
    if not qdrant_path.exists() and not database_path.exists():
        raise BackupError("No persistent vector or experiment data was found")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "app_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "collection": settings.qdrant_collection,
        "contains_vectors": qdrant_path.exists(),
        "contains_experiments": database_path.exists(),
    }

    with tempfile.TemporaryDirectory(prefix="ragbench-backup-") as temporary_directory:
        sqlite_snapshot = Path(temporary_directory) / "experiments.sqlite3"
        if database_path.exists():
            source = sqlite3.connect(database_path)
            destination = sqlite3.connect(sqlite_snapshot)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()

        with tarfile.open(output, "w:gz") as archive:
            if qdrant_path.exists():
                archive.add(qdrant_path, arcname="qdrant", recursive=True)
            if sqlite_snapshot.exists():
                archive.add(sqlite_snapshot, arcname="experiments.sqlite3")
            payload = json.dumps(manifest, indent=2).encode()
            metadata = tarfile.TarInfo("manifest.json")
            metadata.size = len(payload)
            metadata.mtime = int(datetime.now(UTC).timestamp())
            archive.addfile(metadata, fileobj=BytesIO(payload))
    return output
