# frontend/ -- Mission Console

**Status: a minimal, real static dashboard exists. It is not the
React/Next.js/TypeScript/Tailwind "Enterprise Dashboard" this file
previously described as a future possibility (Mission Overview, TRMG
visualization, Argument Graph, Cyber Propagation, etc., one view per CMI
component) -- that remains not started.**

## What exists

A build-step-free, dependency-free static page (`index.html` +
`styles.css` + `app.js`, no npm, no bundler, no CDN script) that talks to
`backend/app.py`'s real API over `fetch()`:

- Lists all 18 Digital Twin scenarios from `GET /scenarios`.
- Runs a selected scenario live via `POST /scenarios/{name}/run?seed=`
  and renders the full per-query-time trajectory: an outcome summary, a
  canvas line chart of `trust_mean` and `p_success` over time, and a
  table of every `MissionRecord` field the API returns.
- Polls `GET /health` for a live backend-status indicator.

No value on the page is fabricated, cached, or hand-authored -- every
number comes straight from the same API response `tests/backend/test_app.py`
exercises.

## Run it

`backend/app.py` mounts this directory at `/ui`:

```bash
pip install -e ".[backend]"
uvicorn backend.app:app --reload
# open http://127.0.0.1:8000/ui
```

## What was intentionally left out

No build tooling (React/Vite/webpack), no client-side routing, no state
management library, no charting library -- the page is small enough
(three files, no dependency) that none of that is justified yet. The
canvas chart is hand-drawn with the 2D Canvas API rather than a charting
library for the same reason `backend/`'s README gives for skipping a
database: nothing here yet needs the complexity. The originally-envisioned
per-component views (TRMG visualization, Argument Graph, Trust Evolution,
Cyber Propagation, Counterfactual Viewer, Evaluation dashboard) are only
possible for API endpoints that exist -- today that's the Digital Twin
scenario runner only, so that is the only view built. The others remain
real future work, not started here.

## Tests

`tests/backend/test_app.py::test_ui_serves_the_static_console` and
`::test_ui_serves_app_js` confirm the static mount actually serves this
directory's files through the live FastAPI app (not just that the files
exist on disk).
