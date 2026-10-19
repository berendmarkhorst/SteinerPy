"""Select and construct existing strategies from normalized problem data."""

import inspect
from dataclasses import replace

from . import objects
from ._composition import GroupComposition, validate_composition

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

_COMMON_OPTIONS = {
    "weight",
    "preprocess",
    "max_degree",
    "budget",
    "dual_ascent",
    "da_reduce",
    "heavy",
    "special_distance",
    "long_edge",
    "replace_nodes",
    "contract_terminals",
    "bound_based",
    "enumeration_safe",
    "primal_local_search",
    "implied_profit",
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
    options = spec.implementation_options()
    composed = (
        "node_costs" in spec.supplied_features
        and variant != "budgeted_max_weight_connected"
    ) or bool(spec.groups and spec.terminal_groups)
    if composed:
        if spec.requested_variant is not None and variant != "group":
            raise TypeError(
                "groups/node_costs is incompatible with variant={!r}.".format(variant)
            )
        validate_composition(spec)
        # Preserve the composition path's snapshot of original graph and costs.
        spec = replace(
            spec, graph=spec.graph.copy(), node_costs=dict(spec.node_costs or {})
        )
        return spec, None, GroupComposition(spec)

    implementation = _IMPLEMENTATIONS[variant]
    # SteinerProblem inherits its constructor; inspect.signature resolves it.
    parameters = inspect.signature(implementation).parameters
    allowed = set(parameters) - {"kwargs"}
    allowed.update(_COMMON_OPTIONS)
    if issubclass(implementation, objects.PrizeCollectingProblem):
        allowed.update({"pc_transform", "pc_reduce"})
    if implementation in (
        objects.MaxWeightConnectedSubgraph,
        objects.BudgetedMaxWeightConnectedSubgraph,
    ):
        allowed.add("penalty_budget")
    if spec.groups:
        options["groups"] = [list(group) for group in spec.groups]
    if spec.terminal_groups:
        if variant in {"directed", "hop_constrained"}:
            if len(spec.terminal_groups) != 1:
                raise ValueError("Directed terminals must describe one connected tree.")
            options["terminals"] = list(spec.terminal_groups[0])
        else:
            options["terminal_groups"] = [list(group) for group in spec.terminal_groups]
    if variant in {"directed", "directed_group", "hop_constrained"}:
        if options.get("root") is None:
            raise TypeError("Directed problems require root=.")
    if variant in {"group", "directed_group", "node_weighted"}:
        if any(
            options.get(key) is not None
            for key in ("max_degree", "budget", "hop_limit")
        ):
            raise TypeError(
                "groups/node weights cannot yet be combined with degree/hop limits or budgets."
            )
    if variant == "rectilinear":
        if spec.graph is not None:
            raise TypeError("rectilinear takes points=, not graph.")
    else:
        if spec.graph is None:
            raise TypeError("{} requires graph.".format(variant))
        options["graph"] = spec.graph
    unknown = set(options) - allowed
    if unknown:
        raise TypeError(
            "Unsupported option(s) for {}: {}".format(
                variant, ", ".join(sorted(unknown))
            )
        )
    # Validate required arguments before entering a graph transformation.
    inspect.signature(implementation).bind(**options)
    if variant == "node_weighted" and "preprocess" in options:
        if options.pop("preprocess"):
            raise ValueError("node_weighted requires preprocess=False.")
    return spec, variant, implementation(**options)
