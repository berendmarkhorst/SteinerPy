"""A single, composition-based entry point for SteinerPy problem variants."""

import inspect

import networkx as nx

from . import objects
from ._problem_spec import ProblemSpec, choice_sets, terminal_sets
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


class Problem:
    """Configure any supported variant without choosing a problem subclass.

    With no ``variant``, infer the implementation from the graph and features.
    ``terminals=[a, b]`` requires a connected tree; nested lists specify forest
    requirements, each connected internally, with sharing allowed. ``groups``
    selects at least one member per group in one connected tree. Tuple node
    labels remain atomic. ``terminal_groups`` is the legacy forest alias.

    Existing implementations serve as composed strategies, retaining their
    formulations, preprocessing restrictions and solution types. See the
    variant guide and legacy class docstrings for definitions and references.

    Examples::

        Problem(graph, terminals=[0, 3], max_degree=2)
        Problem(digraph, root=0, terminals=[3])
        Problem(graph, groups=[[0, 1], [3, 4]], node_costs={1: 2})
        Problem(variant="rectilinear", points=[(0, 0), (1, 1)])
    """

    variants = tuple(_IMPLEMENTATIONS)

    def __init__(
        self,
        graph=None,
        terminal_groups=None,
        *,
        terminals=None,
        variant=None,
        **options,
    ):
        """Build a configured problem, rejecting unknown or misplaced options.

        A graph is required except for ``rectilinear``, which takes ``points``.
        Solver options such as ``time_limit`` belong to :meth:`get_solution`.
        """
        if variant is not None and (
            not isinstance(variant, str) or variant not in _IMPLEMENTATIONS
        ):
            raise ValueError(
                "Unknown variant {!r}; choose one of {}.".format(
                    variant, ", ".join(self.variants)
                )
            )
        if terminals is not None and terminal_groups is not None:
            raise TypeError("Supply terminals or terminal_groups, not both.")
        if graph is not None and not isinstance(graph, nx.Graph):
            raise TypeError("graph must be a NetworkX graph.")
        explicit = variant is not None
        if variant is None:
            if graph is None:
                raise TypeError("Problem requires graph.")
            if "groups" in options:
                variant = "directed_group" if graph.is_directed() else "group"
            elif graph.is_directed():
                variant = "hop_constrained" if "hop_limit" in options else "directed"
            else:
                variant = "steiner"
        if variant != "rectilinear" and graph is None:
            raise TypeError("{} requires graph.".format(variant))

        normalized = terminal_sets(terminals) if terminals is not None else ()
        if terminal_groups is not None:
            # Legacy forest syntax accepts any iterable of terminal iterables.
            normalized = choice_sets(terminal_groups)
        groups = choice_sets(options["groups"]) if "groups" in options else ()
        spec = ProblemSpec(
            graph,
            normalized,
            groups,
            options.get("node_costs"),
            options.get("weight", "weight"),
        )
        if graph is not None:
            spec.validate_nodes()
            if (
                "root" in options
                and options["root"] is not None
                and options["root"] not in graph
            ):
                raise ValueError(
                    "root {!r} is not in the graph.".format(options["root"])
                )
        self._spec = spec

        composed = (
            "node_costs" in options and variant != "budgeted_max_weight_connected"
        ) or bool(groups and normalized)
        if composed:
            if explicit and variant != "group":
                raise TypeError(
                    "groups/node_costs is incompatible with variant={!r}.".format(
                        variant
                    )
                )
            composition_options = {
                key: value
                for key, value in options.items()
                if key not in {"groups", "node_costs"}
            }
            validate_composition(spec, composition_options)
            # Snapshot original data so later caller mutations cannot change a solve.
            spec = ProblemSpec(
                graph.copy(),
                normalized,
                groups,
                dict(spec.node_costs or {}),
                spec.weight,
            )
            self._spec = spec
            self._implementation = GroupComposition(spec)
            self.variant = None
            return

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
        if groups:
            options["groups"] = [list(group) for group in groups]
        if normalized:
            if variant in {"directed", "hop_constrained"}:
                if len(normalized) != 1:
                    raise ValueError(
                        "Directed terminals must describe one connected tree."
                    )
                options["terminals"] = list(normalized[0])
            else:
                options["terminal_groups"] = [list(group) for group in normalized]
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
            if graph is not None:
                raise TypeError("rectilinear takes points=, not graph.")
        else:
            if graph is None:
                raise TypeError("{} requires graph.".format(variant))
            options["graph"] = graph
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
        self._implementation = implementation(**options)
        self.variant = variant

    def get_solution(self, *args, **kwargs):
        """Solve using the selected variant's solver options and solution type."""
        return self._implementation.get_solution(*args, **kwargs)

    def get_optimal_solutions(self, *args, **kwargs):
        """Enumerate optima, retaining the selected variant's restrictions."""
        return self._implementation.get_optimal_solutions(*args, **kwargs)

    @property
    def implementation(self):
        """The selected legacy object or private composition strategy.

        Inspect or modify model attributes here, for example
        ``problem.implementation.cut_stats``. Keeping access explicit avoids
        attribute assignments on a wrapper silently diverging from the solver.
        """
        return self._implementation
