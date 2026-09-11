"""A single, composition-based entry point for SteinerPy problem variants."""

import inspect

from . import objects

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

    ``variant`` selects the mathematical problem; remaining keyword arguments
    use the corresponding legacy constructor's names. ``terminal_groups`` means
    sets whose members must all be connected, whereas ``groups`` means sets
    from each of which at least one member must be selected. These arguments
    are deliberately not interchangeable.

    Existing implementations serve as composed strategies, retaining their
    formulations, preprocessing restrictions and solution types. See the
    variant guide and legacy class docstrings for definitions and references.

    Examples::

        Problem(graph, [[0, 3]], max_degree=2)
        Problem(digraph, variant="directed", root=0, terminals=[3])
        Problem(graph, variant="group", groups=[[0, 1], [3, 4]])
        Problem(variant="rectilinear", points=[(0, 0), (1, 1)])
    """

    variants = tuple(_IMPLEMENTATIONS)

    def __init__(
        self, graph=None, terminal_groups=None, *, variant="steiner", **options
    ):
        """Build a configured problem, rejecting unknown or misplaced options.

        A graph is required except for ``rectilinear``, which takes ``points``.
        Solver options such as ``time_limit`` belong to :meth:`get_solution`.
        """
        if not isinstance(variant, str) or variant not in _IMPLEMENTATIONS:
            raise ValueError(
                "Unknown variant {!r}; choose one of {}.".format(
                    variant, ", ".join(self.variants)
                )
            )
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
        if terminal_groups is not None:
            options["terminal_groups"] = terminal_groups
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
        """The configured legacy implementation for advanced model access.

        Inspect or modify model attributes here, for example
        ``problem.implementation.cut_stats``. Keeping access explicit avoids
        attribute assignments on a wrapper silently diverging from the solver.
        """
        return self._implementation
