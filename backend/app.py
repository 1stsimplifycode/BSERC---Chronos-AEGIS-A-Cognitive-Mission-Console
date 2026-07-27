"""Chronos-AEGIS backend: a minimal, real FastAPI wrapper over the
reasoning pipeline (Algorithm 1, Section 5) and the Mission Digital Twin
(Section 15), for a live, runnable demonstration.

This is deliberately *not* the full production API layer this package's
own README envisioned in Phase 1 (auth, PostgreSQL/Neo4j/Redis, Celery
background workers, WebSocket streaming) -- none of that exists here.
What exists is real: every endpoint below calls directly into the same
library code `examples/run_ttrn_pipeline*.py` and `digital_twin/` already
exercise, with no mock data and no fabricated response. See
`backend/README.md` for the full scope statement.

`frontend/` (a static, build-step-free HTML/CSS/JS console) is mounted
at `/ui` below and talks to this same API -- see `frontend/README.md`.

Run locally with:
    uvicorn backend.app:app --reload
    # then open http://127.0.0.1:8000/ui
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from digital_twin.dataset import MissionRecord, build_dataset
from digital_twin.integration import run_scenario_with_pipeline
from digital_twin.scenarios import ALL_SCENARIO_BUILDERS, ScenarioSpec
from digital_twin.twin import MissionDigitalTwin
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline

app = FastAPI(
    title="Chronos-AEGIS API",
    description=(
        "A minimal live demonstration of the Chronos-AEGIS Cognitive "
        "Mission Intelligence reasoning pipeline (manuscript Sections "
        "5 and 15). No authentication, no persistence, no fabricated "
        "data -- every response is computed live from the same library "
        "code the test suite and examples exercise. See "
        "docs/traceability-matrix.md in the source repository for what "
        "is and is not experimentally validated."
    ),
    version="0.1.0",
)

_allowed_origins = [o for o in os.environ.get("CHRONOS_CORS_ORIGINS", "*").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_SCENARIO_SPECS: dict[str, ScenarioSpec] = {b().name: b() for b in ALL_SCENARIO_BUILDERS}
"""Built once at import time -- ScenarioSpec construction is pure and
cheap (digital_twin/scenarios.py's own docstring documents this), so
there is no per-request rebuild cost and no shared mutable state risk."""


class ScenarioInfo(BaseModel):
    """A scenario's own metadata -- not its event sequence, which is not
    JSON-serializable and not needed by a caller deciding what to run."""

    name: str
    description: str
    ground_truth_outcome: str
    intervention_nodes: tuple[str, ...]
    query_times: tuple[float, ...]


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Render's health-check endpoint."""
    return {"status": "ok"}


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "name": "chronos-aegis",
        "description": "Cognitive Mission Intelligence reasoning pipeline -- live demo API",
        "docs": "/docs",
        "scenarios": "/scenarios",
        "ui": "/ui",
    }


@app.get("/scenarios", response_model=list[ScenarioInfo], tags=["digital-twin"])
def list_scenarios() -> list[ScenarioInfo]:
    """Every Digital Twin scenario (Section 15.1's ten canonical
    scenarios, the normal-mission baseline, two composites, and Phase
    8's five firmware scenarios -- 18 total)."""
    return [
        ScenarioInfo(
            name=spec.name,
            description=spec.description,
            ground_truth_outcome=spec.ground_truth_outcome.value,
            intervention_nodes=spec.intervention_nodes,
            query_times=spec.query_times,
        )
        for spec in _SCENARIO_SPECS.values()
    ]


@app.post("/scenarios/{name}/run", response_model=list[MissionRecord], tags=["digital-twin"])
def run_scenario(name: str, seed: int = 0) -> list[MissionRecord]:
    """Run one scenario through Algorithm 1 (the bare pipeline -- no
    optional Episodic Memory/Argumentation/Causal Engine/Scheduler
    collaborators, matching `evaluation.batch.default_pipeline_factory`)
    and return the full per-query-time trajectory, live and real -- not
    a cached or pre-computed result. Deterministic for a given `seed`
    (see docs/reproducibility-guide.md in the source repository).
    """
    spec = _SCENARIO_SPECS.get(name)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown scenario {name!r}; see GET /scenarios for valid names",
        )

    twin = MissionDigitalTwin(spec.name, seed=seed)
    gnn = MissionAwareGNN(embedding_dim=16, feature_dim=8, seed=seed)
    pipeline = TTRNPipeline(twin.graph, gnn, k_hop=6)
    query_points = run_scenario_with_pipeline(twin, spec, pipeline)
    return list(build_dataset(spec.name, seed, query_points))


_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=_FRONTEND_DIR, html=True), name="ui")
