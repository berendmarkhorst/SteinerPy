"""A single, composition-based entry point for SteinerPy problem variants."""

from ._problem_spec import normalize_problem
from ._problem_dispatch import VARIANTS, build_problem


class Problem:
    """Configure any supported variant without choosing a problem subclass.

    With no ``variant``, infer the implementation from the graph and features.
    ``terminals=[a, b]`` requires a connected tree; nested lists specify forest
    requirements, each connected internally, with sharing allowed. ``groups``
    selects at least one member per group in one connected tree. Tuple node
    labels remain atomic. ``terminal_groups`` is the legacy forest alias.

    Feature combinations accept ``node_prizes`` (minimize edge/node costs plus
    omitted prizes) or signed ``node_weights`` (maximize selected node weights;
    edge/node costs then serve only as budget resources). ``node_budget`` limits
    ``node_costs`` separately from the edge ``budget``. With a reward objective,
    terminal/group requirements stay mandatory unless ``penalty_cost`` makes
    them optional; ``penalty_budget`` limits missed requirements. Without rewards
    or penalties, an edge budget maximizes coverage, then minimizes cost.
    ``partial_terminals`` limits listed terminals/group members to degree one.
    Composed ``preprocess=True`` removes only edges forbidden by resource bounds,
    preserving all optima. Enumeration returns distinct inclusion-minimal original
    node/edge selections; its time limit applies per solve, as in the legacy API.

    Existing implementations serve as composed strategies, retaining their
    formulations, preprocessing restrictions and solution types. See the
    variant guide and legacy class docstrings for definitions and references.

    Examples::

        Problem(graph, terminals=[0, 3], max_degree=2)
        Problem(digraph, root=0, terminals=[3])
        Problem(graph, groups=[[0, 1], [3, 4]], node_costs={1: 2})
        Problem(variant="rectilinear", points=[(0, 0), (1, 1)])
    """

    variants = VARIANTS

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
        spec = normalize_problem(
            graph, terminal_groups, terminals, variant, options, self.variants
        )
        self._spec, self.variant, self._implementation = build_problem(spec)

    def get_solution(self, *args, **kwargs):
        """Solve using the selected variant's solver options and solution type."""
        if self._uses_classic_composition():
            return self._classic_composed_solution(*args, **kwargs)
        return self._implementation.get_solution(*args, **kwargs)

    def _classic_composed_solution(
        self, time_limit=300, log_file="", solver="highs", threads=None, exact=True
    ):
        from ._feature_composition import FeatureComposition

        if not exact:
            raise NotImplementedError(
                "Feature composition does not support heuristic-only solving."
            )
        # Composition's fourth positional argument is threads; the classic
        # signature uses that position for dual_ascent. Forward by name.
        solution = self._implementation.get_solution(
            time_limit=time_limit,
            log_file=log_file,
            solver=solver,
            threads=threads,
            exact=exact,
        )
        nodes = {n for group in self._spec.terminal_groups for n in group}
        nodes.update(n for edge in solution.edges for n in edge)
        strategy = FeatureComposition(self._spec)
        result = strategy.decode(
            solution.edges, [n for n in self._spec.graph if n in nodes]
        )
        return strategy.solution(result, solution.gap, solution.runtime)

    def _uses_classic_composition(self):
        from .objects import SteinerProblem

        return self.variant is None and isinstance(self._implementation, SteinerProblem)

    def get_optimal_solutions(self, *args, **kwargs):
        """Enumerate optima, retaining the selected variant's restrictions."""
        if (
            self._uses_classic_composition()
            or self.variant in {"group", "directed_group"}
            or (
                self._spec.budget is not None
                and self.variant in {"steiner", "directed", "hop_constrained"}
            )
        ):
            from ._feature_composition import FeatureComposition

            return FeatureComposition(self._spec).get_optimal_solutions(*args, **kwargs)
        return self._implementation.get_optimal_solutions(*args, **kwargs)

    @property
    def implementation(self):
        """The selected legacy object or private composition strategy.

        Inspect or modify model attributes here, for example
        ``problem.implementation.cut_stats``. Keeping access explicit avoids
        attribute assignments on a wrapper silently diverging from the solver.
        """
        return self._implementation
