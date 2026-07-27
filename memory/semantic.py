"""Semantic Memory M_s (manuscript Section 8.2.2) and its distillation
step (Algorithm 2, lines 6-8).

"M_s holds general, mission-invariant knowledge... entries in M_s are not
indexed by a specific past mission but represent knowledge distilled across
many missions... performed by the consolidation step of Section 8.3 when a
pattern recurs across a threshold number of episodic entries."

Algorithm 2:
    6: if a pattern in s has recurred across theta_pattern entries of M_e
       with consistent o then
    7:     Distill the recurring (s -> o) pattern into M_s as a new or
           updated semantic rule
    8: end if

This module distills strictly on "consistent o": a candidate pattern
(grouped by dependency_identity + topology_class) is only distilled into a
rule if every one of its >= theta_pattern supporting episodes shares the
*same* outcome -- not merely a majority -- since the manuscript's wording
is "consistent", not "predominant".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from memory.episodic import EpisodicEntry
from memory.outcome import MissionOutcome


@dataclass(frozen=True)
class SemanticRule:
    """A distilled (s -> o) pattern (Algorithm 2, line 7)."""

    dependency_identity: str | None
    topology_class: str | None
    predicted_outcome: MissionOutcome
    support_count: int


class SemanticMemory:
    """M_s: a store of SemanticRules, keyed by the same (dependency_identity,
    topology_class) grouping Episodic Memory indexes on."""

    def __init__(self) -> None:
        self._rules: dict[tuple[str | None, str | None], SemanticRule] = {}

    def distill(
        self, entries: Iterable[EpisodicEntry], theta_pattern: int
    ) -> tuple[SemanticRule, ...]:
        """Algorithm 2, lines 6-8: group entries by (dependency_identity,
        topology_class); for any group with >= theta_pattern entries all
        sharing the same outcome, create or update a SemanticRule. Returns
        the rules that were created or updated by this call.
        """
        groups: dict[tuple[str | None, str | None], list[EpisodicEntry]] = defaultdict(list)
        for entry in entries:
            key = (entry.features.dependency_identity, entry.features.topology_class)
            groups[key].append(entry)

        updated: list[SemanticRule] = []
        for key, group in groups.items():
            if len(group) < theta_pattern:
                continue
            outcomes = {e.outcome for e in group}
            if len(outcomes) != 1:
                continue  # not consistent
            rule = SemanticRule(
                dependency_identity=key[0],
                topology_class=key[1],
                predicted_outcome=next(iter(outcomes)),
                support_count=len(group),
            )
            self._rules[key] = rule
            updated.append(rule)
        return tuple(updated)

    def query(
        self, dependency_identity: str | None, topology_class: str | None
    ) -> SemanticRule | None:
        return self._rules.get((dependency_identity, topology_class))

    def all_rules(self) -> tuple[SemanticRule, ...]:
        return tuple(self._rules.values())
