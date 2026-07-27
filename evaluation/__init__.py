"""The Evaluation Framework (manuscript Section 17), built on top of the
Mission Digital Twin (`digital_twin/`, Section 15) and `TTRNPipeline`
(`reasoning/ttrn/pipeline.py`, Section 5).

evaluation.batch        Batch execution: scenario/seed/parameter sweeps
                         over the Digital Twin + TTRNPipeline
evaluation.aggregate     Per-scenario metric aggregation + JSON summaries
evaluation.statistics    Descriptive stats, Cohen's d, Holm-Bonferroni,
                         and (when scipy is installed) paired significance
                         tests (Section 17.6)

See `evaluation/README.md` for what this does and does not claim: this
package runs the real, implemented reasoning pipeline over real simulated
mission data and reports whatever it genuinely computes -- it does not
report, and this repository does not contain anywhere, a claimed
evaluation *result* for any of the manuscript's RQ1-RQ13.
"""

from evaluation.aggregate import ScenarioSummary, summarize_scenario, summarize_sweep
from evaluation.batch import RunResult, default_pipeline_factory, run_single, run_sweep
from evaluation.statistics import (
    DescriptiveStats,
    PairedComparisonResult,
    cohens_d,
    descriptive_stats,
    holm_bonferroni,
    paired_comparison,
)

__all__ = [
    "ScenarioSummary",
    "summarize_scenario",
    "summarize_sweep",
    "RunResult",
    "default_pipeline_factory",
    "run_single",
    "run_sweep",
    "DescriptiveStats",
    "PairedComparisonResult",
    "cohens_d",
    "descriptive_stats",
    "holm_bonferroni",
    "paired_comparison",
]
