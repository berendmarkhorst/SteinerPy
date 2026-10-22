"""Select and construct existing strategies from normalized problem data."""

from dataclasses import replace

from . import objects
from ._composition import GroupComposition
from ._feature_composition import FeatureComposition
from ._problem_validation import validate_problem

_IMPLEMENTATIONS = {
    "steiner": objects.SteinerProblem,
    "directed": objects.DirectedSteinerProblem,
    "group": objects.GroupSteinerProblem,
    "directed_group": objects.DirectedGroupSteinerProblem,
    "partial_terminal": objects.PartialTerminalSteinerProblem,
    "full_terminal": objects.FullTerminalSteinerProblem,
    "hop_constrained": objects.HopConstrainedSteinerProblem,
    "prize_collecting": objects.PrizeCollectingProblem,
    "directed_prize_collecting": objects.DirectedPrizeCollectingProblem,
    "node_weighted": objects.NodeWeightedSteinerProblem,
    "max_weight_connected": objects.MaxWeightConnectedSubgraph,
    "budgeted_max_weight_connected": objects.BudgetedMaxWeightConnectedSubgraph,
    "rectilinear": objects.RectilinearSteinerProblem,
}

VARIANTS = tuple(_IMPLEMENTATIONS)


def _classic_cost_only(spec):
    """Classic edge minimization also permits constant mandatory-node costs."""
    return (
        spec.requested_variant in {None, "steiner"}
        and not spec.graph.is_directed()
        # The local repeated comparison regressed on forests; keep their
        # existing formulation until a measured selection rule supports them.
        and len(spec.terminal_groups) == 1
        and not spec.groups
        and {n for n, cost in spec.node_costs.items() if cost}
        <= set(spec.terminal_groups[0])
        and not spec.partial_terminals
        and all(
            getattr(spec, name) is None
            for name in (
                "root",
                "node_prizes",
                "node_weights",
                "budget",
                "node_budget",
                "penalty_cost",
                "penalty_budget",
                "max_degree",
                "hop_limit",
            )
        )
    )


def _select_variant(spec):
    """Resolve supported features without changing the normalized specification."""
    if spec.requested_variant is not None:
        return spec.requested_variant
    if spec.groups:
        return "directed_group" if spec.graph.is_directed() else "group"
    if spec.graph.is_directed():
        return (
            "hop_constrained" if "hop_limit" in spec.supplied_features else "directed"
        )
    return "steiner"


def build_problem(spec):
    """Adapt a normalized specification to a legacy or composed implementation."""
    variant = _select_variant(spec)
    if variant == "rectilinear":
        implementation = _IMPLEMENTATIONS[variant]
        options = validate_problem(spec, variant, implementation)
        return spec, variant, implementation(**options)
    feature_mode = spec.requested_variant in {
        None,
        "steiner",
        "directed",
        "group",
        "directed_group",
        "hop_constrained",
    } and (
        bool(
            spec.supplied_features.intersection(
                {
                    "node_prizes",
                    "node_weights",
                    "node_budget",
                    "partial_terminals",
                    "penalty_cost",
                    "penalty_budget",
                }
            )
        )
        or (
            not spec.graph.is_directed()
            and (
                spec.root is not None
                or (
                    spec.hop_limit is not None
                    and not spec.groups
                    and "node_costs" not in spec.supplied_features
                )
            )
        )
    )
    if spec.requested_variant in {"partial_terminal", "full_terminal"} and (
        spec.groups
        or "node_costs" in spec.supplied_features
        or any(
            getattr(spec, name) is not None
            for name in (
                "budget",
                "node_budget",
                "hop_limit",
                "node_costs",
                "node_prizes",
                "node_weights",
                "penalty_cost",
                "penalty_budget",
                "max_degree",
            )
        )
    ):
        feature_mode = True
        if (
            spec.requested_variant == "partial_terminal"
            and spec.partial_terminals is None
        ):
            raise TypeError("partial_terminal requires partial_terminals=.")
        if spec.requested_variant == "full_terminal":
            spec = replace(
                spec,
                partial_terminals=tuple(
                    dict.fromkeys(
                        n for g in spec.terminal_groups + spec.groups for n in g
                    )
                ),
            )
    composed = (
        "node_costs" in spec.supplied_features
        and variant != "budgeted_max_weight_connected"
    ) or bool(
        spec.groups
        and (
            spec.terminal_groups
            or (
                not spec.graph.is_directed()
                and spec.supplied_features.intersection(
                    {"hop_limit", "max_degree", "budget"}
                )
            )
        )
    )
    composed |= len(spec.terminal_groups) > 1 and (
        spec.hop_limit is not None or spec.graph.is_directed()
    )
    composed |= feature_mode
    composed |= bool(
        spec.groups
        and (
            spec.options.get("preprocess", False)
            or spec.options.get("enumeration_safe", False)
        )
    )
    if composed:
        feature_mode |= bool(
            spec.options.get("preprocess", False)
            or spec.options.get("enumeration_safe", False)
        )
        if spec.partial_terminals is not None:
            spec = replace(spec, partial_terminals=tuple(spec.partial_terminals))
        validate_problem(spec, None)
        # Preserve the composition path's snapshot of original graph and costs.
        spec = replace(
            spec,
            graph=spec.graph.copy(),
            node_costs=dict(spec.node_costs or {}),
            node_prizes=(
                dict(spec.node_prizes) if spec.node_prizes is not None else None
            ),
            node_weights=(
                dict(spec.node_weights) if spec.node_weights is not None else None
            ),
            partial_terminals=tuple(spec.partial_terminals or ()),
        )
        if _classic_cost_only(spec):
            # Composition defaults to no reductions. Enumeration still uses the
            # original specification so reductions cannot erase tied optima.
            return (
                spec,
                None,
                objects.SteinerProblem(
                    spec.graph,
                    [list(group) for group in spec.terminal_groups],
                    weight=spec.weight,
                    preprocess=spec.options.get("preprocess", False),
                    enumeration_safe=False,
                ),
            )
        strategy = (
            FeatureComposition
            if feature_mode or len(spec.terminal_groups) > 1
            else GroupComposition
        )
        return spec, None, strategy(spec)

    implementation = _IMPLEMENTATIONS[variant]
    options = validate_problem(spec, variant, implementation)
    instance = implementation(**options)
    if variant in {"group", "directed_group"} or (
        spec.budget is not None
        and variant in {"steiner", "directed", "hop_constrained"}
    ):
        spec = replace(spec, graph=spec.graph.copy())
    return spec, variant, instance
