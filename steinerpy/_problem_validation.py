"""Validate normalized problem features before transformations or construction."""

import inspect
import math
from collections.abc import Mapping

import networkx as nx

from . import objects

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


def validate_problem(spec, variant, implementation=None):
    """Check a selected strategy and return its validated constructor arguments.

    ``variant=None`` denotes the group/node-cost composition. Implementations
    are inspected, never constructed, so validation cannot transform the graph.
    """
    spec.validate_nodes()
    if variant is None:
        if spec.requested_variant not in (None, "group"):
            raise TypeError(
                "groups/node_costs is incompatible with variant={!r}.".format(
                    spec.requested_variant
                )
            )
        _validate_composition(spec)
        return {}

    options = spec.implementation_options()
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
    return options


def _validate_composition(spec):
    """Reject features whose original-space meaning is not implemented yet."""
    if spec.graph.is_directed() or spec.graph.is_multigraph():
        raise ValueError(
            "Group/node-cost composition requires a simple undirected graph."
        )
    if nx.number_of_selfloops(spec.graph):
        raise ValueError("Group/node-cost composition does not support self-loops.")
    if len(spec.terminal_groups) > 1:
        raise ValueError(
            "Forest terminals cannot yet be combined with groups/node_costs."
        )
    if not spec.groups:
        raise ValueError("node_costs requires groups in the modular interface for now.")
    options = spec.implementation_options()
    options.pop("node_costs", None)
    unknown = set(options) - {"weight", "preprocess"}
    if unknown:
        raise TypeError(
            "Unsupported option(s) for group/node-cost composition: {}".format(
                ", ".join(sorted(unknown))
            )
        )
    if options.get("preprocess", False):
        raise ValueError("Group/node-cost composition requires preprocess=False.")
    if spec.node_costs is not None and not isinstance(spec.node_costs, Mapping):
        raise TypeError("node_costs must map original nodes to non-negative costs.")
    for node, cost in (spec.node_costs or {}).items():
        if node not in spec.graph:
            raise ValueError("Node cost vertex {!r} is not in the graph.".format(node))
        _validate_cost(cost, "node {!r}".format(node))
    for u, v, data in spec.graph.edges(data=True):
        _validate_cost(data.get(spec.weight, 1), "edge {!r}".format((u, v)))


def _validate_cost(cost, location, nonnegative=True):
    try:
        valid = math.isfinite(cost) and (not nonnegative or cost >= 0)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(
            "Costs must be {}; invalid cost at {}.".format(
                "finite and non-negative" if nonnegative else "finite", location
            )
        )
