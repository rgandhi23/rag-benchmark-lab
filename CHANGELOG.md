# Changelog

All notable project changes are documented here.

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
