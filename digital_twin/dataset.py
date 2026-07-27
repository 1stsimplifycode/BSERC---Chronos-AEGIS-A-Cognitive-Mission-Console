"""Dataset generation (Section 15: "This becomes the reproducibility
dataset."): flattens `digital_twin.integration.TwinQueryPoint`s into
per-query-time records and exports them as CSV or JSON via the standard
library only.

This module does not re-derive anything from `MissionDigitalTwin` itself
-- every field it writes was already captured on `TwinQueryPoint` at the
correct simulation time by `run_scenario_with_pipeline`
(`digital_twin/integration.py`); see that module's docstring for why
recomputing mission-state or cyber-metric snapshots here, after the twin
has finished running, would be wrong (it would describe the twin's final
state at every row instead of each row's own query time).

Parquet is not implemented: this repository has no pandas/pyarrow
dependency anywhere else in the codebase, and adding one solely for this
optional export path would be a new dependency the phase brief explicitly
makes optional ("Parquet (optional if already used)"). Documented, not
silently skipped.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from digital_twin.integration import TwinQueryPoint


@dataclass(frozen=True)
class MissionRecord:
    """One flattened row of the reproducibility dataset: timestamp,
    mission state, trust, scheduler output, cyber metrics, memory
    retrieval, causal outcome, and argumentation result, per Section 15's
    Dataset Generation requirement."""

    scenario_name: str
    seed: int
    timestamp: float
    mission_id: str
    mission_phase: str
    mission_completed: bool
    mission_outcome: str | None
    trust_mean: float
    trust_variance: float
    p_success: float
    expected_delay: float
    cyber_impact: float
    comm_reliability: float
    operational_confidence: float
    mdri_value: float
    mdri_dependency_node_count: int
    cyber_incident_count: int
    memory_episodes_retrieved: int
    argumentation_claim: str | None
    argumentation_escalated: bool
    argumentation_accepted_agent_count: int
    causal_result_count: int
    causal_summary: str
    scheduler_action: str | None
    scheduler_feasible: bool | None
    scheduler_summary: str | None
    warning_count: int


def build_record(scenario_name: str, seed: int, query_point: TwinQueryPoint) -> MissionRecord:
    """Flatten one `TwinQueryPoint` into a `MissionRecord`."""
    result = query_point.result
    reconciled = result.reconciled_decision
    causal = result.causal_context
    scheduling = result.scheduling_result

    return MissionRecord(
        scenario_name=scenario_name,
        seed=seed,
        timestamp=query_point.time,
        mission_id=result.state.mission_id,
        mission_phase=query_point.mission_phase,
        mission_completed=query_point.mission_completed,
        mission_outcome=(
            query_point.mission_outcome.value if query_point.mission_outcome is not None else None
        ),
        trust_mean=result.posterior.mean,
        trust_variance=result.posterior.variance,
        p_success=result.predictions.p_success,
        expected_delay=result.predictions.expected_delay,
        cyber_impact=result.predictions.cyber_impact,
        comm_reliability=result.predictions.comm_reliability,
        operational_confidence=result.predictions.operational_confidence,
        mdri_value=query_point.mdri_value,
        mdri_dependency_node_count=query_point.mdri_dependency_node_count,
        cyber_incident_count=query_point.cyber_incident_count,
        memory_episodes_retrieved=len(result.state.retrieved_memory),
        argumentation_claim=reconciled.claim if reconciled is not None else None,
        argumentation_escalated=reconciled.escalated if reconciled is not None else False,
        argumentation_accepted_agent_count=(
            len(reconciled.extension_hypotheses) if reconciled is not None else 0
        ),
        causal_result_count=len(causal.results) if causal is not None else 0,
        causal_summary=causal.summary if causal is not None else "",
        scheduler_action=scheduling.action if scheduling is not None else None,
        scheduler_feasible=scheduling.feasible if scheduling is not None else None,
        scheduler_summary=scheduling.summary if scheduling is not None else None,
        warning_count=len(result.warnings),
    )


def build_dataset(
    scenario_name: str, seed: int, query_points: tuple[TwinQueryPoint, ...]
) -> tuple[MissionRecord, ...]:
    """Flatten every `TwinQueryPoint` from one scenario run into records,
    in query-time order."""
    return tuple(build_record(scenario_name, seed, qp) for qp in query_points)


def to_json(records: tuple[MissionRecord, ...]) -> str:
    return json.dumps([asdict(r) for r in records], indent=2)


def write_json(records: tuple[MissionRecord, ...], path: str | Path) -> None:
    Path(path).write_text(to_json(records), encoding="utf-8")


def write_csv(records: tuple[MissionRecord, ...], path: str | Path) -> None:
    """Write `records` as CSV. An empty `records` tuple still produces a
    valid (headerless) empty file rather than raising, since "no rows"
    is a legitimate, honest outcome for e.g. a scenario with no query
    times configured."""
    if not records:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(asdict(records[0]).keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
