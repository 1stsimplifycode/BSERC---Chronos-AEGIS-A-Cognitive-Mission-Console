"""Structural Causal Model (manuscript Section 10.2).

"We define a structural causal model over the mission-relevant subset of
the TRMG as a triple M = (U, V, F), following Pearl's formalism: U is a set
of exogenous (background) variables representing factors outside the
model's control (for example, unmodeled environmental noise); V is a set
of endogenous variables corresponding to TRMG node attributes relevant to
the mission decision (aircraft-subsystem health, communication
availability, weather state, cyber-alert status, mission trust T); and F =
{f_v : v in V} is a set of structural functions, each expressing v as a
deterministic function of its direct causes in V union U, v = f_v(pa(v),
u_v), where pa(v) denotes v's parents in the induced causal graph. The
causal graph itself -- which TRMG edges are treated as causal parent
relationships rather than merely correlational associations -- is a subset
of the TRMG's dependency edges d (Section 6.1), restricted to those for
which a structural function f_v has been specified or learned; we do not
assume every TRMG edge is causal, and Section 10.4 discusses how this
subset is determined."

IMPLEMENTATION SCOPE. Section 10.4 states plainly that causal discovery
("determining which TRMG dependency edges are causal") is "an open
problem, not a solved one" for this work, to be addressed empirically once
Section 17's evaluation protocol runs; until then "the causal graph...
should be understood as partially specified by domain knowledge... rather
than fully learned from data." Consistent with that, this module does NOT
attempt to discover structural functions from data. `StructuralCausalModel`
is a container a caller populates with domain-supplied structural
functions (`register`) -- exactly the "engineer-supplied statement" kind of
annotation Section 10.4 describes as the interim approach. Nothing here
invents a specific physical model (e.g. "communication failure reduces
trust by 0.3") on the manuscript's behalf.

`do()` is implemented via the standard graph-mutilation semantics --
replace f_X with the constant x, leave every other structural function
unchanged, then evaluate forward -- which is exact given a fully specified
M. Section 10.3 additionally distinguishes the *observational* conditional
P(T | X=x) from the interventional P(T | do(X=x)), which in the general
case (recovering an interventional distribution from purely observational
data) requires Pearl's abduction step over a distribution on U that the
manuscript does not specify. Algorithm 5 (Section 10.5), the only
algorithmic specification in Section 10, only ever calls for the
interventional (do) computation -- so that is what this module implements;
the fuller observational/interventional/counterfactual three-way machinery
of Section 10.3 remains a documented gap, not silently approximated.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

StructuralFunction = Callable[[dict[str, float], float], float]
"""f_v(pa(v), u_v) -> v: parent values keyed by variable id, plus this
variable's own exogenous term, produce this variable's value."""


@dataclass(frozen=True)
class _Equation:
    parents: tuple[str, ...]
    function: StructuralFunction
    exogenous: float


class StructuralCausalModel:
    """M = (U, V, F) (Section 10.2), restricted to variables a caller has
    explicitly registered a structural function for -- an unregistered
    TRMG node/edge is simply outside V, not assumed acausal or causal."""

    def __init__(self) -> None:
        self._equations: dict[str, _Equation] = {}

    def register(
        self,
        variable: str,
        parents: tuple[str, ...],
        function: StructuralFunction,
        exogenous: float = 0.0,
    ) -> None:
        """f_v = function, pa(v) = parents, u_v = exogenous. Registering v
        with a nonempty `parents` is what marks every (parent, v) edge as
        causal (Section 10.2: "restricted to those [edges] for which a
        structural function f_v has been specified")."""
        if variable in parents:
            raise ValueError(f"variable '{variable}' cannot be its own parent")
        self._equations[variable] = _Equation(
            parents=parents, function=function, exogenous=exogenous
        )

    @property
    def variables(self) -> frozenset[str]:
        return frozenset(self._equations)

    def is_causal_edge(self, source: str, target: str) -> bool:
        """Whether (source, target) is marked causal in M: target has a
        registered structural function with source among its parents."""
        equation = self._equations.get(target)
        return equation is not None and source in equation.parents

    def parents_of(self, variable: str) -> tuple[str, ...]:
        equation = self._equations.get(variable)
        return equation.parents if equation is not None else ()

    def descendants(self, variable: str) -> frozenset[str]:
        """Every variable reachable from `variable` by following parent ->
        child edges (Algorithm 5, line 4: "Recompute descendants of X")."""
        children: dict[str, list[str]] = {v: [] for v in self._equations}
        for v, eq in self._equations.items():
            for p in eq.parents:
                children.setdefault(p, []).append(v)

        seen: set[str] = set()
        queue: deque[str] = deque(children.get(variable, []))
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(children.get(node, []))
        return frozenset(seen)

    def _topological_order(self) -> list[str]:
        in_degree = {v: len(eq.parents) for v, eq in self._equations.items()}
        # Only count parents that are themselves registered variables;
        # unregistered parents are treated as free (already available) inputs.
        for v, eq in self._equations.items():
            in_degree[v] = sum(1 for p in eq.parents if p in self._equations)

        ready = deque(v for v, d in in_degree.items() if d == 0)
        order: list[str] = []
        remaining = dict(in_degree)
        children: dict[str, list[str]] = {v: [] for v in self._equations}
        for v, eq in self._equations.items():
            for p in eq.parents:
                if p in self._equations:
                    children[p].append(v)

        while ready:
            node = ready.popleft()
            order.append(node)
            for child in children[node]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    ready.append(child)

        if len(order) != len(self._equations):
            raise ValueError(
                "StructuralCausalModel contains a cycle among registered variables; "
                "this evaluator only supports acyclic structural causal models."
            )
        return order

    def evaluate(
        self, overrides: dict[str, float] | None = None, inputs: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Evaluate every registered variable in topological order.

        `overrides` implements do(): a variable in `overrides` takes that
        constant value directly, its own structural function is not
        called (graph mutilation, Section 10.2/10.5). `inputs` supplies
        values for parents that are referenced but not themselves
        registered in M (e.g. a raw TRMG-derived reading feeding into a
        registered variable's structural function) -- distinct from
        `overrides` because an input is observed data, not an intervention.
        """
        overrides = overrides or {}
        inputs = inputs or {}
        values: dict[str, float] = {}

        for variable in self._topological_order():
            if variable in overrides:
                values[variable] = overrides[variable]
                continue
            equation = self._equations[variable]
            parent_values: dict[str, float] = {}
            for parent in equation.parents:
                if parent in values:
                    parent_values[parent] = values[parent]
                elif parent in inputs:
                    parent_values[parent] = inputs[parent]
                else:
                    raise KeyError(
                        f"variable '{variable}' depends on parent '{parent}', which is "
                        f"neither a registered variable evaluated so far nor supplied in `inputs`"
                    )
            values[variable] = equation.function(parent_values, equation.exogenous)

        return values

    def do(
        self, variable: str, value: float, inputs: dict[str, float] | None = None
    ) -> dict[str, float]:
        """do(X=x): evaluate M with `variable` mutilated to the constant
        `value`, everything else computed normally (Section 10.2, Algorithm
        5 lines 3-4)."""
        return self.evaluate(overrides={variable: value}, inputs=inputs)
