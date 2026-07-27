# backend/ -- API Layer

**Status: a minimal, real implementation exists (`app.py`). The full
production layer described below (auth, datastores, background workers)
remains not yet implemented.**

## What exists

`app.py`: a FastAPI application wrapping `reasoning/ttrn/pipeline.py`'s
`TTRNPipeline` and `digital_twin/`'s scenario library directly -- no mock
data, no cached responses, every endpoint computes a real result live.

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Health check (Render's `healthCheckPath`) |
| `GET` | `/` | Service metadata |
| `GET` | `/scenarios` | List all 18 Digital Twin scenarios (Section 15.1's ten canonical + baseline + 2 composites + Phase 8's 5 firmware scenarios) |
| `POST` | `/scenarios/{name}/run?seed=0` | Run one scenario through Algorithm 1 (bare pipeline, no optional collaborators) and return the full per-query-time trajectory as `digital_twin.dataset.MissionRecord` rows |
| `GET` | `/docs` | Auto-generated OpenAPI/Swagger UI (FastAPI default) |

## Run locally

```bash
pip install -e ".[backend]"
uvicorn backend.app:app --reload
# then open http://127.0.0.1:8000/docs
```

## What was intentionally left out

Per the project's original deployment requirements, a full production
layer would additionally have authentication, PostgreSQL/Neo4j/Redis
persistence, Celery background workers, and WebSocket streaming. None of
that exists: this backend is stateless (every request builds a fresh
`MissionDigitalTwin`/`TTRNPipeline`, matching how `examples/run_ttrn_pipeline.py`
and the test suite already use these classes) and has no datastore,
because nothing in this repository yet needs one -- a Digital Twin
scenario run completes in a few milliseconds
(`paper/artifact/benchmark-appendix.md`), so there is no async/background
workload to justify Celery, and there is no user data to justify a
database or auth layer yet. Building those out is real future work, not
started here; see `.env.example` for the environment-variable shape
already reserved for it (`DATABASE_URL`, `NEO4J_*`, `REDIS_URL`,
`CHRONOS_SECRET_KEY`) and `docs/code-quality-audit.md` for why
`pydantic` (used by this backend for response schemas) and `PyYAML`
(used to validate `deployment/render.yaml`, not loaded by this app at
runtime) are declared dependencies.

## Tests

`tests/backend/test_app.py` -- every endpoint, including a
parametrized run of all 18 scenarios through the live API and a
same-seed-twice determinism check.
