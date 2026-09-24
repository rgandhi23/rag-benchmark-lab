# Deployment Guide

RAG Benchmark Lab ships as a non-root, health-checked container with persistent storage.
Demo mode is the safest default and requires no external credentials.

## Local Docker deployment

```bash
cp .env.example .env.local
docker compose up --build --detach
docker compose ps
curl --fail http://localhost:8000/health
```

Open `http://localhost:8000`. Qdrant vectors and SQLite benchmark history are stored in the
named `ragbench-data` volume and survive container replacement.

Stop the service without deleting its data:

```bash
docker compose down
```

Delete the persisted data only when intentionally starting over:

```bash
docker compose down --volumes
```

## Runtime configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAGBENCH_PORT` | `8000` | Host port used by Compose |
| `RAGBENCH_PROVIDER` | `demo` | `demo` or `openai` |
| `RAGBENCH_RETRIEVAL_MODE` | `vector` | `vector` or `hybrid` |
| `RAGBENCH_STORE_BACKEND` | `qdrant` | `qdrant` or `memory` |
| `RAGBENCH_MAX_UPLOAD_BYTES` | `10485760` | Maximum PDF upload size |

When using OpenAI mode, inject `OPENAI_API_KEY` through the deployment platform's secret
manager. Do not bake `.env.local` into an image or commit it to Git.

## Container security

The Compose configuration:

- runs the application as an unprivileged `ragbench` user;
- removes Linux capabilities and prevents privilege escalation;
- mounts the root filesystem read-only;
- provides a size-limited temporary filesystem;
- writes application state only to `/data`;
- performs an HTTP health check every 30 seconds.

## GitHub Container Registry

The Container workflow builds every pull request. Pushes to `main` and tags matching `v*`
publish images to `ghcr.io/<owner>/<repository>`. GitHub's automatically provided token is used;
no registry password needs to be stored in the repository.

Create a release after CI is green:

```bash
git tag v0.2.0
git push origin v0.2.0
```

## Production notes

Local embedded Qdrant is designed for a single application process. Before horizontally scaling,
move vectors to a Qdrant server or another shared vector database and move experiment history to
PostgreSQL. Terminate TLS at a trusted reverse proxy or managed platform and restrict upload size
there as well as in the application.
