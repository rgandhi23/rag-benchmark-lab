import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response


@dataclass
class RequestMetric:
    requests: int = 0
    errors: int = 0
    latency_seconds: float = 0.0


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._metrics: dict[tuple[str, str], RequestMetric] = defaultdict(RequestMetric)

    def record(self, method: str, path: str, status_code: int, latency_seconds: float) -> None:
        with self._lock:
            metric = self._metrics[(method, path)]
            metric.requests += 1
            metric.errors += int(status_code >= 500)
            metric.latency_seconds += latency_seconds

    def render(self) -> str:
        lines = [
            "# HELP ragbench_http_requests_total HTTP requests processed.",
            "# TYPE ragbench_http_requests_total counter",
        ]
        with self._lock:
            items = sorted(self._metrics.items())
            for (method, path), metric in items:
                labels = f'method="{method}",path="{path}"'
                lines.append(f"ragbench_http_requests_total{{{labels}}} {metric.requests}")
            lines.extend(
                [
                    "# HELP ragbench_http_errors_total HTTP 5xx responses.",
                    "# TYPE ragbench_http_errors_total counter",
                ]
            )
            for (method, path), metric in items:
                labels = f'method="{method}",path="{path}"'
                lines.append(f"ragbench_http_errors_total{{{labels}}} {metric.errors}")
            lines.extend(
                [
                    "# HELP ragbench_http_request_duration_seconds_total "
                    "Cumulative request latency.",
                    "# TYPE ragbench_http_request_duration_seconds_total counter",
                ]
            )
            for (method, path), metric in items:
                labels = f'method="{method}",path="{path}"'
                lines.append(
                    "ragbench_http_request_duration_seconds_total"
                    f"{{{labels}}} {metric.latency_seconds:.6f}"
                )
        return "\n".join(lines) + "\n"


def configure_observability(app: FastAPI) -> MetricsRegistry:
    registry = MetricsRegistry()
    logger = logging.getLogger("ragbench.requests")

    @app.middleware("http")
    async def request_telemetry(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid4())
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            elapsed = perf_counter() - started
            registry.record(request.method, request.url.path, status_code, elapsed)
            logger.exception(
                json.dumps(
                    {
                        "event": "request_completed",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "latency_ms": round(elapsed * 1_000, 2),
                    }
                )
            )
            raise
        elapsed = perf_counter() - started
        registry.record(request.method, request.url.path, status_code, elapsed)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            json.dumps(
                {
                    "event": "request_completed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round(elapsed * 1_000, 2),
                }
            )
        )
        return response

    return registry
