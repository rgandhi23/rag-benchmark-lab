import json
import sqlite3
from pathlib import Path
from threading import Lock

from ragbench.models import BenchmarkRun, EvaluationResponse


class BenchmarkRunRepository:
    """SQLite repository for comparing benchmark results across configurations."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if path != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = Lock()
        self._initialize()

    def _initialize(self) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    store_backend TEXT NOT NULL,
                    k INTEGER NOT NULL,
                    hit_rate_at_k REAL NOT NULL,
                    precision_at_k REAL NOT NULL,
                    recall_at_k REAL NOT NULL,
                    mean_reciprocal_rank REAL NOT NULL,
                    average_latency_ms REAL NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )

    def save(self, result: EvaluationResponse, provider: str, store_backend: str) -> int:
        with self.lock, self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO benchmark_runs (
                    name, provider, store_backend, k, hit_rate_at_k, precision_at_k,
                    recall_at_k, mean_reciprocal_rank, average_latency_ms, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.name,
                    provider,
                    store_backend,
                    result.k,
                    result.hit_rate_at_k,
                    result.precision_at_k,
                    result.recall_at_k,
                    result.mean_reciprocal_rank,
                    result.average_latency_ms,
                    json.dumps([case.model_dump() for case in result.cases]),
                ),
            )
            return int(cursor.lastrowid)

    def list(self, limit: int = 25) -> list[BenchmarkRun]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT run_id, name, created_at, provider, store_backend, k,
                       hit_rate_at_k, precision_at_k, recall_at_k,
                       mean_reciprocal_rank, average_latency_ms
                FROM benchmark_runs
                ORDER BY run_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [BenchmarkRun.model_validate(dict(row)) for row in rows]

    def ping(self) -> None:
        with self.lock:
            self.connection.execute("SELECT 1").fetchone()

    def count(self) -> int:
        with self.lock:
            row = self.connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()
        return int(row[0])

    def close(self) -> None:
        self.connection.close()
