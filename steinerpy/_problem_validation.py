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
    if variant in {"group", "node_weighted"}:
        modifiers = ("budget", "max_degree", "hop_limit")
        if any(options.get(key) is not None for key in modifiers):
            raise TypeError(
                "groups/node weights cannot yet be combined with these constraints: "
                "budget/degree/hop limits require directed groups."
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
    _validate_legacy_features(spec, variant, implementation, options)
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
    unknown = set(options) - {
        "weight",
        "preprocess",
        "hop_limit",
        "max_degree",
        "budget",
    }
    if unknown:
        raise TypeError(
            "Unsupported option(s) for group/node-cost composition: {}".format(
                ", ".join(sorted(unknown))
            )
        )
    if options.get("preprocess", False):
        raise ValueError("Group/node-cost composition requires preprocess=False.")
    objects._validate_group_limit(spec.hop_limit, "hop_limit")
    objects._validate_group_limit(spec.max_degree, "max_degree")
    objects._validate_group_budget(spec.budget)
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


def _validate_legacy_features(spec, variant, implementation, options):
    """Reject unsupported feature interactions before legacy constructors run."""
    if variant == "group":
        _validate_composition(spec)
    if variant in {"directed_group", "node_weighted"} and spec.graph.is_multigraph():
        raise ValueError(f"{variant} requires a simple graph.")
    if variant == "directed_group":
        objects._validate_group_limit(options.get("hop_limit"), "hop_limit")
        objects._validate_group_limit(options.get("max_degree"), "max_degree")
        objects._validate_group_budget(options.get("budget"))
    if variant == "steiner" and options.get("preprocess", True):
        if spec.graph.is_directed():
            raise ValueError(
                "Graph preprocessing is not supported for directed graphs. "
                "Use preprocess=False."
            )
        if options.get("enumeration_safe", False) and any(
            data.get(spec.weight, 1) <= 0 for _, _, data in spec.graph.edges(data=True)
        ):
            raise ValueError(
                "enumeration_safe=True requires strictly positive edge weights."
            )
    directed = {
        "directed",
        "directed_group",
        "hop_constrained",
        "directed_prize_collecting",
    }
    if variant in directed and not spec.graph.is_directed():
        raise ValueError(
            "{} requires a directed graph (nx.DiGraph).".format(implementation.__name__)
        )
    if (
        variant in {"group", "node_weighted", "partial_terminal", "full_terminal"}
        and spec.graph.is_directed()
    ):
        raise ValueError(
            "{} requires an undirected graph.".format(implementation.__name__)
        )

    if variant in {"partial_terminal", "full_terminal"} and spec.budget is not None:
        raise ValueError(
            "Edge budgets cannot yet be combined with terminal-leaf transformations: "
            "transformed edge costs do not represent the original budget."
        )
    if (
        variant == "group"
        and options.get("enumeration_safe", False)
        and options.get("preprocess", True)
    ):
        raise ValueError(
            "Group preprocessing cannot use enumeration_safe=True: "
            "the transformation introduces zero-cost edges. Use preprocess=False."
        )
    if variant == "directed_prize_collecting":
        if not spec.graph:
            raise ValueError("graph must contain at least one node.")
        if options.get("pc_reduce", False):
            raise ValueError("pc_reduce is not supported for directed graphs.")
        for name in ("budget", "max_degree", "hop_limit", "penalty_budget"):
            if options.get(name) is not None:
                raise ValueError(
                    "{} is not supported by DirectedPrizeCollectingProblem.".format(
                        name
                    )
                )
        if not options.get("pc_transform", True):
            raise ValueError("Directed prize collecting requires pc_transform=True.")

    if variant == "budgeted_max_weight_connected":
        for name in ("budget", "penalty_budget"):
            if options.get(name) is not None:
                raise ValueError(
                    "{} is not supported by the node-budgeted maximum-weight variant; "
                    "use node_budget for its vertex-cost budget.".format(name)
                )
        for name in ("pc_transform", "pc_reduce"):
            if options.get(name, False):
                raise ValueError(
                    "{} is not supported by the node-budgeted maximum-weight variant.".format(
                        name
                    )
                )

    if variant in {"prize_collecting", "max_weight_connected"} and options.get(
        "pc_transform", False
    ):
        if spec.graph.is_directed():
            raise ValueError(
                "This prize transformation requires an undirected graph; "
                "use variant='directed_prize_collecting' for directed prizes."
            )
        incompatible = any(
            options.get(name) is not None
            for name in ("budget", "max_degree", "penalty_budget")
        )
        if variant == "prize_collecting":
            incompatible |= (
                len(spec.terminal_groups) != 1 or options.get("penalty_cost", 1000) != 0
            )
        if incompatible:
            raise ValueError(
                "pc_transform=True requires a classic prize-collecting objective or "
                "maximum-weight objective without degree limits or budgets."
            )

    if "pc_reduce" in options:
        value = options["pc_reduce"]
        if not (
            value in (False, None)
            or value is True
            or value in ("pcd", "pcd+trd", "pcd+trd+nodes")
        ):
            raise ValueError(
                "Unknown pc_reduce={!r}; choose False, True, 'pcd', 'pcd+trd', "
                "or 'pcd+trd+nodes'.".format(value)
            )

    for name in ("node_weights", "node_prizes", "node_costs"):
        if name not in spec.supplied_features:
            continue
        values = getattr(spec, name)
        if not isinstance(values, Mapping):
            raise TypeError(
                "{} must map original nodes to costs or prizes.".format(name)
            )
        nonnegative = name != "node_weights" or variant not in {
            "max_weight_connected",
            "budgeted_max_weight_connected",
        }
        for node, value in values.items():
            if node not in spec.graph:
                raise ValueError(
                    "{} vertex {!r} is not in the graph.".format(name, node)
                )
            _validate_cost(value, "{} node {!r}".format(name, node), nonnegative)

    # MWCSPB uses edge attributes only as topology, so signed costs remain valid.
    if spec.graph is not None and implementation._requires_nonnegative_edge_costs:
        for u, v, data in spec.graph.edges(data=True):
            _validate_cost(data.get(spec.weight, 1), "edge {!r}".format((u, v)))
