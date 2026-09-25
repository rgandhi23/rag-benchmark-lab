# Changelog

All notable project changes are documented here.

## 0.3.0 - 2026-09-25

### Added

- Separate liveness and dependency-aware readiness endpoints.
- Storage diagnostics for vectors and benchmark history.
- JavaScript health polling and operational dashboard indicators.
- Portable Qdrant and SQLite backup archives with versioned manifests.
- Kubernetes Deployment, Service, and persistent volume example.
- Wheel-build verification that confirms JavaScript assets are packaged.

### Changed

- Compose now applies CPU, memory, process-count, and graceful-shutdown limits.
- Container health checks use the readiness endpoint.

## 0.2.0 - 2026-09-24

### Added

- PDF ingestion with page-aware citations and configurable upload limits.
- Multi-stage, non-root Docker image with an HTTP health check.
- Hardened Compose deployment with persistent `/data` storage.
- GitHub Container Registry build and publishing workflow.
- Configurable CLI quality gates for MRR, recall, hit rate, and retrieval latency.
- Deployment guide and portfolio release checklist.

### Fixed

- Removed an invalid Docker build step that referenced a nonexistent top-level `data` directory.
