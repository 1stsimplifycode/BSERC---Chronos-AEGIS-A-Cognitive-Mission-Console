# deployment/ -- Render Deployment Configuration

**Status: configuration is real and locally verified; no live deployment
exists yet.**

## What exists

`render.yaml`: a Render Blueprint for `backend/app.py`'s minimal FastAPI
service -- Python (no Docker needed; Render's native Python runtime
builds via `pip install -e '.[backend]'` and runs
`uvicorn backend.app:app --host 0.0.0.0 --port $PORT`), a free-tier web
service, and `/health` wired as the health-check path. No Dockerfile
exists in `docker/` because Render's native Python runtime is sufficient
for this service -- see `docker/README.md`.

Locally verified during Phase 9's deployment-preparation work: the exact
`startCommand` above was run directly
(`uvicorn backend.app:app --host 127.0.0.1 --port 8123`) and served real
requests to `/health` and `/scenarios` before being stopped.

## What does not exist yet

**No live Render deployment, and no URL.** The repository itself is now
on GitHub (https://github.com/1stsimplifycode/chronos-aegis), so that
half of the prerequisite is done. What remains needs the repository
owner's own Render account -- either (a) connecting the GitHub repo in
the Render dashboard ("New Blueprint Instance" -\> point at this repo), or
(b) authenticated use of the Render CLI/API. Neither is something an
assistant can do without the repository owner's own account access --
see `docs/reproducibility-audit.md`'s "no step requires ... an account/
API key" statement, which describes the *test/reproducibility* workflow,
not deployment; deployment is the one part of this project that
inherently needs an external, credentialed action.

A real URL will be added here only after an actual deployment has been
created -- never fabricated ahead of time.

## To actually deploy (repository owner's own action)

1. In the Render dashboard: New -\> Blueprint -\> select
   `1stsimplifycode/chronos-aegis`. Render will read
   `deployment/render.yaml` (see that file's own header for the
   root-vs-subdirectory blueprint-path note).
2. Confirm the build succeeds and `/health` returns `{"status": "ok"}`.
3. Add the resulting URL to `.env.example`'s `RENDER_EXTERNAL_URL`
   documentation and to this file.

## Environment variables

`.env.example` at the repo root documents every variable the eventual
full production platform would use; `render.yaml` currently sets only
`CHRONOS_CORS_ORIGINS` (the one this minimal backend actually reads --
`backend/app.py`'s CORS middleware). The database/auth variables remain
unused until `backend/`'s fuller production layer is built.
