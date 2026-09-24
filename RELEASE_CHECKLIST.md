# Portfolio Release Checklist

## Quality

- [ ] `make lint` passes.
- [ ] `make test` passes with coverage reported.
- [ ] `make docker-build` succeeds from a clean checkout.
- [ ] `/health`, `/v1/query`, `/v1/evaluate`, and PDF rejection paths are verified.
- [ ] Vector and hybrid benchmark results are recorded with the same evaluation set.

## Security

- [ ] `.env.local` is ignored and no credential appears in Git history.
- [ ] The container runs as a non-root user.
- [ ] Upload limits are configured at both the API and hosting layer.
- [ ] Only public or synthetic documents are included in the repository.

## Portfolio presentation

- [ ] The README screenshot matches the current dashboard.
- [ ] The case study explains the baseline, intervention, measured improvement, and tradeoff.
- [ ] The architecture diagram matches the deployed system.
- [ ] The demo script works from a fresh clone.
- [ ] Repository description, topics, license, and social preview are configured.

## Release

- [ ] CI is green on `main`.
- [ ] The version and changelog are updated.
- [ ] A `v*` tag builds and publishes the GHCR image.
- [ ] The deployed `/health` endpoint reports healthy.
- [ ] The public URL is added to the README and repository About section.
