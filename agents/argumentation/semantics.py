"""Dung-style abstract argumentation semantics over the Argumentation Graph
(manuscript Section 9.2-9.4).

Algorithm 4, line 4: "Extension(AG) <- compute the grounded (or preferred,
if the grounded extension is empty) extension of AG."

Standard Dung (1995) definitions, specialized to a finite agent-id set:
  - S is conflict-free if no a, b in S have (a, b) in Attacks.
  - S defends a if every attacker of a is itself attacked by some member of S.
  - S is admissible if S is conflict-free and S defends every member of S.
  - The grounded extension is the least fixed point of the characteristic
    function F(S) = {a : S defends a}, starting from S = empty set.
  - A preferred extension is a set-inclusion-maximal admissible set.

Complexity (Section 9.5): "since |A| = 9 is fixed and small, this cost is
not a practical bottleneck at the scale specified here." `preferred_extensions`
brute-forces over all 2^|agent_ids| subsets, which is exactly the naive
algorithm Section 9.5 is describing as acceptable at this scale -- it is not
intended for a much larger agent set.
"""

from __future__ import annotations

from itertools import combinations


def is_conflict_free(subset: frozenset[str], attacks: frozenset[tuple[str, str]]) -> bool:
    return not any(a in subset and b in subset for a, b in attacks)


def defends(subset: frozenset[str], agent_id: str, attacks: frozenset[tuple[str, str]]) -> bool:
    """Whether `subset` defends `agent_id`: every attacker of `agent_id` is
    itself attacked by some member of `subset`."""
    attackers = {a for a, b in attacks if b == agent_id}
    return all(
        any((defender, attacker) in attacks for defender in subset) for attacker in attackers
    )


def characteristic_function(
    subset: frozenset[str], agent_ids: frozenset[str], attacks: frozenset[tuple[str, str]]
) -> frozenset[str]:
    """F(S) = {a in H : S defends a}."""
    return frozenset(a for a in agent_ids if defends(subset, a, attacks))


def grounded_extension(
    agent_ids: frozenset[str], attacks: frozenset[tuple[str, str]]
) -> frozenset[str]:
    """The least fixed point of F, starting from the empty set. F is
    monotonic over a finite domain, so this terminates in at most
    |agent_ids| iterations."""
    current: frozenset[str] = frozenset()
    for _ in range(len(agent_ids) + 1):
        nxt = characteristic_function(current, agent_ids, attacks)
        if nxt == current:
            return current
        current = nxt
    return current  # pragma: no cover -- unreachable given the iteration bound above


def is_admissible(
    subset: frozenset[str], agent_ids: frozenset[str], attacks: frozenset[tuple[str, str]]
) -> bool:
    if not is_conflict_free(subset, attacks):
        return False
    defended = characteristic_function(subset, agent_ids, attacks)
    return subset <= defended


def preferred_extensions(
    agent_ids: frozenset[str], attacks: frozenset[tuple[str, str]]
) -> tuple[frozenset[str], ...]:
    """All set-inclusion-maximal admissible sets, via brute-force
    enumeration (see module docstring for why this is faithful to Section
    9.5's complexity remark at |A| = 9)."""
    ordered = sorted(agent_ids)
    admissible_sets: list[frozenset[str]] = []
    for size in range(len(ordered), -1, -1):
        for combo in combinations(ordered, size):
            candidate = frozenset(combo)
            if any(candidate < existing for existing in admissible_sets):
                continue  # already dominated by a known-maximal superset
            if is_admissible(candidate, agent_ids, attacks):
                admissible_sets.append(candidate)

    maximal = [
        s for s in admissible_sets if not any(s < other for other in admissible_sets if s != other)
    ]
    return tuple(maximal)
