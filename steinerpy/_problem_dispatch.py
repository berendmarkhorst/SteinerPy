"""Select and construct existing strategies from normalized problem data."""

from dataclasses import replace

from . import objects
from ._composition import GroupComposition
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
    composed = (
        "node_costs" in spec.supplied_features
        and variant != "budgeted_max_weight_connected"
    ) or bool(spec.groups and spec.terminal_groups)
    if composed:
        validate_problem(spec, None)
        # Preserve the composition path's snapshot of original graph and costs.
        spec = replace(
            spec, graph=spec.graph.copy(), node_costs=dict(spec.node_costs or {})
        )
        return spec, None, GroupComposition(spec)

    implementation = _IMPLEMENTATIONS[variant]
    options = validate_problem(spec, variant, implementation)
    return spec, variant, implementation(**options)
