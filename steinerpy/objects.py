import networkx as nx
import highspy as hp
import warnings
from typing import List, Tuple, Dict, Optional
import logging
from .mathematical_model import (
    build_model, run_model,
    build_model_gurobi, run_model_gurobi,
    build_prize_collecting_model, run_prize_collecting_model,
    build_budget_model, run_budget_model,
)
from .graph_reducer import preprocess_graph, reduction_stats, map_solution_to_original, ReductionTracker

logger = logging.getLogger(__name__)


def node_split_graph(
    graph: nx.Graph,
    terminal_groups: List[List],
    node_weights: Dict,
    weight: str = "weight",
) -> Tuple[nx.DiGraph, List[List], Dict]:
    """
    Transform a node-weighted graph into an edge-weighted *directed* graph by
    splitting each node v into v_in and v_out connected by a directed edge whose
    weight equals node_weights[v].  Original edges become directed cross-edges
    between the split nodes in both directions (to preserve undirected semantics).

    :param graph: original undirected graph.
    :param terminal_groups: nested list of terminals.
    :param node_weights: dict mapping node -> node cost/weight.
    :param weight: edge-attribute name for weights.
    :return: (new_graph, new_terminal_groups, node_map)
             where node_map maps every split-node name back to its original node.
    """
    new_graph = nx.DiGraph()
    node_map: Dict[str, object] = {}

    for v in graph.nodes():
        v_in, v_out = f"{v}_in", f"{v}_out"
        node_map[v_in] = v
        node_map[v_out] = v
        new_graph.add_edge(v_in, v_out, **{weight: node_weights.get(v, 0)})

    for u, v, data in graph.edges(data=True):
        edge_cost = data.get(weight, 0)
        # Both directions to preserve undirected semantics
        new_graph.add_edge(f"{u}_out", f"{v}_in", **{weight: edge_cost})
        new_graph.add_edge(f"{v}_out", f"{u}_in", **{weight: edge_cost})

    new_terminal_groups = [[f"{t}_in" for t in group] for group in terminal_groups]
    return new_graph, new_terminal_groups, node_map


class BaseSteinerProblem:
    def __init__(self, graph: nx.Graph, terminal_groups: List[List], weight="weight", preprocess=True, **kwargs):
        """
        Initialize the SteinerProblem (can be tree or forest).
        :param graph: networkx graph (Graph or DiGraph).
        :param terminal_groups: nested list of terminals.
        :param weight: edge attribute specified by this string as the edge weight.
        """
        self.original_graph = graph
        self.preprocess = preprocess
        # Opt-in dual-ascent bound reduction (removes provably non-optimal edges
        # then cascades the degree reductions). Requires preprocessing, an
        # undirected graph, and no budget/degree modifier.
        self.da_reduce = kwargs.get('da_reduce', False)
        # Opt-in *heavy* graph reductions (sound edge-deletion tests from the
        # Steiner reduction literature): the Special Distance / bottleneck
        # Steiner distance test (Rehfeldt & Koch 2023, Thm 1; Steiner tree only)
        # and the long-edge / alternative-path test (tree and forest). ``heavy``
        # turns on whichever applies; ``special_distance`` / ``long_edge`` allow
        # fine-grained control. Like ``da_reduce`` they require preprocessing, an
        # undirected graph, and no budget/degree modifier (those variants do not
        # minimise edge cost, so a "non-optimal edge" deletion would be unsound).
        self.heavy_reduce = kwargs.get('heavy', False)
        sd_opt = kwargs.get('special_distance', self.heavy_reduce)
        le_opt = kwargs.get('long_edge', self.heavy_reduce)

        if preprocess:
            if isinstance(graph, nx.DiGraph):
                raise ValueError("Graph preprocessing is not supported for directed graphs. Use preprocess=False.")
            _heavy_ok = kwargs.get('budget') is None and kwargs.get('max_degree') is None
            self.graph, self.reduction_tracker = preprocess_graph(
                graph, terminal_groups, weight,
                special_distance=bool(sd_opt) and _heavy_ok,
                long_edge=bool(le_opt) and _heavy_ok,
            )
            if self.da_reduce and kwargs.get('budget') is None and kwargs.get('max_degree') is None:
                from .dual_ascent import reduce_graph_with_dual_ascent
                self.graph = reduce_graph_with_dual_ascent(
                    self.graph, terminal_groups, weight, self.reduction_tracker)
            stats = reduction_stats(self.original_graph, self.graph)
            logger.info(f"Graph reduced: {stats['nodes_removed']} nodes ({stats['node_reduction_percent']:.1f}%), "
                  f"{stats['edges_removed']} edges ({stats['edge_reduction_percent']:.1f}%) removed")
        else:
            self.graph = graph
            self.reduction_tracker = ReductionTracker()

        self.terminal_groups = terminal_groups
        self.weight = weight
        self.edges = list(self.graph.edges())
        # For directed graphs arcs are one-directional; for undirected both directions
        if isinstance(self.graph, nx.DiGraph):
            self.arcs = self.edges
        else:
            self.arcs = self.edges + [(v, u) for (u, v) in self.edges]
        self.nodes = list(self.graph.nodes())
        self.steiner_points = set(self.nodes) - set([t for group in terminal_groups for t in group])
        self.roots = [group[0] for group in self.terminal_groups]

        # Extract global modifiers like max_degree or budget from kwargs
        self.max_degree = kwargs.get('max_degree', None)
        self.budget = kwargs.get('budget', None)
        # Opt-in dual-ascent accelerator (lower bound + primal heuristic +
        # reduced-cost variable fixing). Off by default; see steinerpy.dual_ascent.
        self.dual_ascent = kwargs.get('dual_ascent', False)


    def _da_eligible(self) -> bool:
        """Whether the dual-ascent accelerator is sound for this problem.

        Eligible for plain Steiner tree/forest (undirected) and directed Steiner
        arborescence (DiGraph). Excluded when a budget or degree constraint is
        active, since the dual-ascent primal heuristic ignores those and its
        upper bound would no longer be valid for reduced-cost fixing.
        """
        if self.budget is not None:
            return False
        if getattr(self, 'max_degree', None) is not None:
            return False
        return True

    def _solution_from_da(self, da, t0, gap) -> 'Solution':
        """Build a :class:`Solution` from a dual-ascent result (no ILP).

        Shared by the proven-optimal early-exit (``gap == 0.0``) and the
        heuristic-only mode (``gap`` = the proven optimality gap).  Maps the
        primal back to the original graph when preprocessing was used.
        """
        import time as _time
        if self.preprocess:
            original = map_solution_to_original(
                da.primal_edges, self.reduction_tracker, self.graph)
        else:
            original = da.primal_edges
        return Solution(
            gap=gap, runtime=_time.time() - t0, objective=da.upper_bound,
            selected_edges=da.primal_edges, original_selected_edges=original,
            was_preprocessed=self.preprocess,
        )

    def _heuristic_solution(self) -> 'Solution':
        """Return the dual-ascent primal directly, with no ILP.

        Genuinely heuristic (the primal may be sub-optimal), but the returned
        ``Solution.gap`` is a *valid* optimality gap: ``gap == 0.0`` certifies
        the primal is provably optimal, and a positive gap bounds how far it
        could be from optimum — neither of which a pure heuristic (e.g.
        ``networkx.steiner_tree``) provides.
        """
        import time as _time, math as _math
        if not self._da_eligible():
            raise NotImplementedError(
                "heuristic mode (exact=False) supports plain Steiner tree/forest "
                "and directed problems only (not budget- or degree-constrained "
                "variants)."
            )
        from .dual_ascent import dual_ascent as _run_da
        t0 = _time.time()
        da = _run_da(self, self.weight)
        if not da.feasible or _math.isinf(da.upper_bound):
            raise RuntimeError(
                "dual-ascent heuristic found no feasible solution; the terminals "
                "may be disconnected."
            )
        if _math.isinf(da.lower_bound):
            gap = _math.inf
        else:
            gap = (da.upper_bound - da.lower_bound) / max(1.0, abs(da.upper_bound))
        return self._solution_from_da(da, t0, gap)

    def __repr__(self):
        return f"Problem with a graph of {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges and {self.terminal_groups} as terminal groups."

    # ------------------------------------------------------------------
    # Biconnected-component decomposition (thesis Ch. 2.6) — single group
    # ------------------------------------------------------------------

    def _decompose_enabled(self) -> bool:
        import os
        val = os.environ.get("STEINERPY_DECOMPOSE")
        return bool(val) and val not in ("0", "false", "False")

    def _decomposable(self) -> bool:
        """Plain single-group min-cost Steiner tree on an undirected graph.

        The decomposition is exactness-preserving only for the plain edge-cost
        objective; budget / degree / directed variants couple across blocks or do
        not minimise edge cost, so they are excluded.
        """
        if isinstance(self.graph, nx.DiGraph):
            return False
        if self.budget is not None or getattr(self, 'max_degree', None) is not None:
            return False
        if len(self.terminal_groups) != 1:
            return False
        return len(self.terminal_groups[0]) >= 2

    def _block_demanded_terminals(self, blocks, terminals):
        """Per-block demanded terminal sets for single-group decomposition.

        Builds the block-cut tree, roots it, and for every block returns the set
        of vertices it must connect: its own terminals plus each incident cut
        vertex whose *away* side (through the block-cut tree) still contains a
        terminal.  A block with < 2 demanded vertices contributes nothing.
        """
        art = set(nx.articulation_points(self.graph))
        # Block-cut tree: block nodes ('B', i) and cut-vertex nodes ('C', c).
        T = nx.Graph()
        node_block = {}  # non-cut vertex -> its (unique) block index
        for i, B in enumerate(blocks):
            bn = ('B', i)
            T.add_node(bn)
            for v in B:
                if v in art:
                    T.add_edge(bn, ('C', v))
                else:
                    node_block[v] = i

        def loc(t):
            return ('C', t) if t in art else ('B', node_block[t])

        own_count = {}
        for t in terminals:
            n = loc(t)
            own_count[n] = own_count.get(n, 0) + 1
        total = len(terminals)

        # Root the tree; compute parent pointers and subtree terminal counts.
        if T.number_of_nodes() == 0:
            return {}
        root = next(iter(T.nodes()))
        parent = {root: None}
        order = [root]
        stack = [root]
        while stack:
            u = stack.pop()
            for v in T.neighbors(u):
                if v not in parent:
                    parent[v] = u
                    order.append(v)
                    stack.append(v)
        subtree = {n: own_count.get(n, 0) for n in T.nodes()}
        for n in reversed(order):
            p = parent[n]
            if p is not None:
                subtree[p] += subtree[n]

        demands = {}
        for i, B in enumerate(blocks):
            bn = ('B', i)
            dem = set(t for t in terminals if t in B)  # own (incl. cut-vertex) terminals
            for cn in T.neighbors(bn):
                c = cn[1]
                if parent.get(cn) == bn:
                    away = subtree[cn]            # cut vertex is a child -> its subtree
                else:
                    away = total - subtree[bn]    # cut vertex is the parent -> the rest
                if away > 0:
                    dem.add(c)
            demands[i] = dem
        return demands

    def _decompose_single_group(self, time_limit, log_file, solver, dual_ascent, threads):
        """Solve a single-group Steiner tree by biconnected-component blocks.

        Returns a :class:`Solution` (union of per-block optimal trees) or ``None``
        when decomposition does not apply / does not help, in which case the
        caller solves monolithically.
        """
        import time as _time
        G = self.graph
        terminals = set(self.terminal_groups[0])
        blocks = [b for b in nx.biconnected_components(G)]
        if len(blocks) <= 1:
            return None  # 2-connected: nothing to gain

        demands = self._block_demanded_terminals(blocks, terminals)

        t0 = _time.time()
        union = {}
        worst_gap = 0.0
        for i, B in enumerate(blocks):
            dem = demands.get(i, set())
            if len(dem) < 2:
                continue  # this block connects nothing on its own
            subG = nx.Graph()
            subG.add_nodes_from(B)
            for (u, v) in G.subgraph(B).edges():
                subG.add_edge(u, v, **{self.weight: G.edges[u, v][self.weight]})
            sub = SteinerProblem(subG, [sorted(dem, key=lambda n: str(n))],
                                 weight=self.weight, preprocess=False)
            sub_sol = sub.get_solution(
                time_limit=time_limit, log_file=log_file, solver=solver,
                dual_ascent=dual_ascent, exact=True, threads=threads,
                decompose=False,
            )
            if sub_sol.gap is not None:
                worst_gap = max(worst_gap, sub_sol.gap)
            for (u, v) in sub_sol.selected_edges:
                union[frozenset((u, v))] = (u, v)

        selected_edges = list(union.values())
        objective = sum(G.edges[e][self.weight] for e in selected_edges)
        if self.preprocess:
            original = map_solution_to_original(selected_edges, self.reduction_tracker, self.graph)
        else:
            original = selected_edges
        return Solution(
            gap=worst_gap, runtime=_time.time() - t0, objective=objective,
            selected_edges=selected_edges, original_selected_edges=original,
            was_preprocessed=self.preprocess,
        )

    def get_solution(self, time_limit: float = 300, log_file: str = "", solver: str = "highs",
                     dual_ascent: bool = None, exact: bool = True, threads: int = None,
                     decompose: bool = None) -> 'Solution':
        """
        Get the solution of the Steiner Problem.

        When ``budget`` was supplied at construction time the solver maximises
        the number of connected terminals subject to the budget constraint and
        returns a :class:`BudgetSolution`.  Otherwise the solver minimises the
        total edge cost and returns a plain :class:`Solution`.

        Optional modifiers set at construction time (``max_degree``, ``budget``)
        are applied automatically regardless of which problem class is used.

        :param time_limit: time limit in seconds.
        :param log_file: path to the log file.
        :param solver: which MIP solver to use – ``"highs"`` (default) or
            ``"gurobi"``.  When ``"gurobi"`` is chosen the cut-based
            formulation is solved with Gurobi lazy-cut callbacks, which
            requires *gurobipy* and a valid Gurobi license to be installed.
        :param exact: when ``True`` (default) solve to optimality.  When
            ``False`` run *heuristic-only* mode: return the dual-ascent primal
            with no ILP — much faster, and the returned ``Solution.gap`` is a
            valid optimality gap (``0.0`` ⇒ provably optimal).  Supported for
            plain Steiner tree/forest and directed problems only; raises
            ``NotImplementedError`` for budget/degree-constrained variants.
        :return: :class:`Solution` (or :class:`BudgetSolution` when a budget is set).
        :raises ValueError: if an unknown solver name is provided.
        :raises ImportError: if ``solver="gurobi"`` but gurobipy is not installed.
        """
        solver = solver.lower()
        if solver not in ("highs", "gurobi"):
            raise ValueError(
                f"Unknown solver '{solver}'. Choose 'highs' or 'gurobi'."
            )

        # Heuristic-only mode: return the dual-ascent primal with no ILP. Much
        # faster (no MIP), and unlike a pure heuristic it carries a proven
        # optimality gap (gap == 0.0 ⇒ provably optimal).
        if not exact:
            return self._heuristic_solution()

        if self.budget is not None:
            # Budget-constrained path: only HiGHS is supported for this variant
            if solver == "gurobi":
                raise NotImplementedError(
                    "The budget-constrained variant does not yet support solver='gurobi'. "
                    "Use solver='highs' instead."
                )
            # Budget-constrained: maximise connected terminals
            model, x, y1, y2, z, f, penalty_vars = build_budget_model(
                self, time_limit=time_limit, logfile=log_file, threads=threads
            )
            gap, runtime, connected_count, selected_edges, penalties = run_budget_model(
                model, self, x, penalty_vars
            )

            if self.preprocess:
                original_selected_edges = map_solution_to_original(
                    selected_edges, self.reduction_tracker, self.graph
                )
            else:
                original_selected_edges = selected_edges

            total_terminals = sum(len(g) for g in self.terminal_groups)

            return BudgetSolution(
                gap=gap,
                runtime=runtime,
                objective=connected_count,
                selected_edges=selected_edges,
                original_selected_edges=original_selected_edges,
                was_preprocessed=self.preprocess,
                connected_terminals=connected_count,
                total_terminals=total_terminals,
                penalties=penalties,
            )

        # Optional biconnected-component decomposition (thesis Ch. 2.6): split a
        # single-group Steiner tree at its articulation points, solve each block
        # independently (smaller MIPs) and recombine. Exactness-preserving; opt-in
        # via decompose=True or the STEINERPY_DECOMPOSE env var.
        decompose_on = self._decompose_enabled() if decompose is None else decompose
        if decompose_on and self._decomposable():
            dec = self._decompose_single_group(
                time_limit, log_file, solver, dual_ascent, threads
            )
            if dec is not None:
                return dec

        # Optional dual-ascent accelerator: lower bound + primal heuristic +
        # reduced-cost variable fixing. Early-exits when proven optimal.
        use_da = self.dual_ascent if dual_ascent is None else dual_ascent
        fixing = None
        da_primal = None
        da_cuts = None
        da_ub = None
        if use_da and self._da_eligible():
            import time as _time
            import math as _math
            from .dual_ascent import (
                dual_ascent as _run_da, reduced_cost_fixing, steiner_cuts,
            )
            _t0 = _time.time()
            da = _run_da(self, self.weight)
            if da.feasible and not _math.isinf(da.lower_bound) and not _math.isinf(da.upper_bound):
                if abs(da.upper_bound - da.lower_bound) <= 1e-6 * max(1.0, abs(da.upper_bound)):
                    # Proven optimal by dual ascent — skip the ILP entirely.
                    return self._solution_from_da(da, _t0, 0.0)
                fixing = reduced_cost_fixing(self, da)
                da_primal = da.primal_edges
                # Warm-start the ILP cut loop with the Steiner cuts found during
                # dual ascent, and supply the primal value as an objective cutoff.
                da_cuts = steiner_cuts(da, self.roots)
                da_ub = da.upper_bound

        if solver == "gurobi":
            model, x, y1, y2, z = build_model_gurobi(self, time_limit=time_limit, logfile=log_file, threads=threads)
            if da_cuts is not None:
                from .dual_ascent import (
                    apply_fixes_gurobi, seed_cuts_gurobi, set_gurobi_cutoff,
                )
                seed_cuts_gurobi(model, y2, z, da_cuts)
                apply_fixes_gurobi(model, x, y1, y2, fixing)
                set_gurobi_cutoff(model, da_ub)
            gap, runtime, objective, selected_edges = run_model_gurobi(model, self, x, y2, z)
            if da_cuts is not None and _math.isinf(objective):
                # The dual-ascent acceleration (cutoff / fixing / seeded cuts)
                # over-constrained the model into infeasibility although the
                # instance is feasible; re-solve from a clean model.
                model, x, y1, y2, z = build_model_gurobi(self, time_limit=time_limit, logfile=log_file, threads=threads)
                gap, runtime, objective, selected_edges = run_model_gurobi(model, self, x, y2, z)
        else:
            model, x, y1, y2, z = build_model(self, time_limit=time_limit, logfile=log_file, threads=threads)
            if da_cuts is not None:
                from .dual_ascent import (
                    apply_fixes_highs, set_highs_warm_start,
                    seed_cuts_highs, set_highs_cutoff,
                )
                seed_cuts_highs(model, y2, z, da_cuts)
                apply_fixes_highs(model, x, y1, y2, fixing)
                set_highs_warm_start(model, x, da_primal)
                set_highs_cutoff(model, da_ub)
            gap, runtime, objective, selected_edges = run_model(model, self, x, y2, z)
            if da_cuts is not None and _math.isinf(objective):
                # The acceleration over-constrained a feasible instance into
                # infeasibility; re-solve from a clean, un-accelerated model.
                model, x, y1, y2, z = build_model(self, time_limit=time_limit, logfile=log_file, threads=threads)
                gap, runtime, objective, selected_edges = run_model(model, self, x, y2, z)

        # Map solution back to original graph if preprocessing was used
        if self.preprocess:
            original_selected_edges = map_solution_to_original(selected_edges, self.reduction_tracker, self.graph)
        else:
            original_selected_edges = selected_edges

        solution = Solution(
            gap=gap,
            runtime=runtime,
            objective=objective,
            selected_edges=selected_edges,
            original_selected_edges=original_selected_edges,
            was_preprocessed=self.preprocess,
        )

        return solution

class Solution:
    def __init__(self, gap: float, runtime: float, objective: float, 
                 selected_edges: List[Tuple], original_selected_edges: List[Tuple] = None,
                 was_preprocessed: bool = False):
        self.gap = gap
        self.runtime = runtime
        self.objective = objective
        self.selected_edges = selected_edges  # Edges in the (possibly reduced) graph
        self.original_selected_edges = original_selected_edges or selected_edges  # Edges in the original graph
        self.was_preprocessed = was_preprocessed
    
    @property
    def edges(self):
        """Return the edges in the original graph."""
        return self.original_selected_edges


class SteinerProblem(BaseSteinerProblem):
    pass

class PrizeCollectingProblem(SteinerProblem):  # Inherit from SteinerProblem instead of BaseSteinerProblem
    def __init__(self, graph, terminal_groups, node_prizes, penalty_cost=1000, penalty_budget=None, **kwargs):
        """
        Prize Collecting Steiner Problem - extends regular Steiner problem.
        :param node_prizes: dict mapping node -> prize value
        :param penalty_cost: cost per unconnected terminal
        :param penalty_budget: maximum total penalty allowed (optional)
        """
        # Prize-Collecting Steiner Tree is incompatible with graph reduction:
        # degree-1 removal and degree-2 contraction discard non-terminal nodes
        # together with their prizes, which corrupts the objective. Force it off
        # (also covers subclasses such as MaxWeightConnectedSubgraph).
        if kwargs.get('preprocess', False):
            warnings.warn(
                "Graph preprocessing is not supported for PrizeCollectingProblem "
                "(and subclasses such as MaxWeightConnectedSubgraph): degree-1/degree-2 "
                "reductions discard non-terminal node prizes and corrupt the objective. "
                "Forcing preprocess=False.",
                UserWarning,
                stacklevel=2,
            )
        kwargs['preprocess'] = False

        # Opt-in accelerators (mirror the dual_ascent=/da_reduce=/heavy= pattern):
        #   pc_transform — solve via the classic PCSTP/MWCSP -> SAP transformation
        #     and the existing dual-ascent + directed-cut machinery (exact path).
        #   pc_reduce    — prize-safe edge deletion (prize-constrained distance).
        # Both default off; the default path remains the penalty/Big-M flow ILP.
        self.pc_transform = kwargs.pop('pc_transform', False)
        self.pc_reduce = kwargs.pop('pc_reduce', False)

        # Initialize base Steiner problem first
        super().__init__(graph, terminal_groups, **kwargs)

        # Add prize collecting specific attributes
        self.node_prizes = node_prizes
        self.penalty_cost = penalty_cost
        self.penalty_budget = penalty_budget

        # For a native classic PCSTP (penalty_cost == 0) the stored graph *is* the
        # PCSTP graph, so the prize-safe PCD reduction can shrink it in place,
        # accelerating both the penalty ILP and the transform path. MWCS uses its
        # own transformed graph (penalty_cost != 0) and reduces inside
        # _build_pc_transform instead.
        self._pc_reduced = False
        if self.pc_reduce and self.penalty_cost == 0 and self._pc_eligible():
            from .pc_reductions import reduce_pcstp_graph
            self.graph = reduce_pcstp_graph(self.graph, self.node_prizes, self.weight)
            self.edges = list(self.graph.edges())
            self.arcs = self.edges + [(v, u) for (u, v) in self.edges]
            self.nodes = list(self.graph.nodes())
            all_terms = {t for grp in self.terminal_groups for t in grp}
            self.steiner_points = set(self.nodes) - all_terms
            self._pc_reduced = True
        
    # ------------------------------------------------------------------
    # Opt-in SAP-transformation fast path (classic forgo-prize PCSTP / MWCSP)
    # ------------------------------------------------------------------

    def _pc_eligible(self) -> bool:
        """Whether the classic PCSTP -> SAP transformation is *sound* here.

        The transformation and the dual-ascent bound model the classic
        forgo-prize PCSTP objective ``sum c(e) + sum_{v not in S} p(v)``. That is
        equivalent to SteinerPy's penalty model only when there are no extra
        modifiers and ``penalty_cost == 0`` (the prize is the only incentive). The
        MWCS subclass overrides this with its own exact mapping.
        """
        if getattr(self, 'penalty_budget', None) is not None:
            return False
        if getattr(self, 'budget', None) is not None:
            return False
        if getattr(self, 'max_degree', None) is not None:
            return False
        if len(self.terminal_groups) != 1:
            return False
        if self.penalty_cost != 0:
            return False
        return True

    def _classic_pcstp(self):
        """Return ``(graph, prizes, mwcsp_const)`` for the classic PCSTP form.

        Native PCSTP uses the stored graph and prizes (``mwcsp_const`` is None).
        :class:`MaxWeightConnectedSubgraph` overrides this to return the
        transformed PCSTP graph and the constant needed to recover the MWCSP
        weight.
        """
        return self.graph, self.node_prizes, None

    def _build_pc_transform(self):
        """Build the PCSTP -> SAP transform context (optionally PCD-reduced)."""
        from .pc_transform import transform_pcstp_to_sap
        graph, prizes, mwcsp_const = self._classic_pcstp()
        if self.pc_reduce and not self._pc_reduced:
            from .pc_reductions import reduce_pcstp_graph
            graph = reduce_pcstp_graph(graph, prizes, self.weight)
        ctx = transform_pcstp_to_sap(graph, prizes, self.weight)
        ctx.mwcsp_const = mwcsp_const
        return ctx

    def _pc_make_solution(self, ctx, edges, nodes, pcstp_obj, gap, runtime) -> 'PrizeCollectingSolution':
        """Build a PrizeCollectingSolution from a back-mapped PCSTP solution."""
        total_prize = sum(ctx.node_prizes.get(v, 0) for v in nodes)
        edge_cost = sum(
            ctx._orig_graph.get_edge_data(u, v).get(self.weight, 1) for (u, v) in edges
        )
        return PrizeCollectingSolution(
            gap=gap, runtime=runtime, objective=pcstp_obj,
            selected_edges=edges, original_selected_edges=edges,
            selected_nodes=nodes, penalties={}, total_prize=total_prize,
            edge_cost=edge_cost, was_preprocessed=False,
        )

    def _pc_finalize(self, ctx, edges, nodes, pcstp_obj, gap, runtime) -> 'PrizeCollectingSolution':
        """Compare with the best trivial (single-vertex/empty) solution, then build.

        The SAP can represent neither the empty tree nor a single non-proper /
        Steiner vertex, so the optimum might be one of those — compare explicitly.
        """
        from .pc_transform import best_trivial_pcstp
        triv_nodes, triv_obj = best_trivial_pcstp(ctx.node_prizes)
        if triv_obj < pcstp_obj - 1e-9:
            edges, nodes, pcstp_obj = [], triv_nodes, triv_obj
        return self._pc_make_solution(ctx, edges, nodes, pcstp_obj, gap, runtime)

    def _pc_exact_solution(self, time_limit, log_file, solver, threads=None) -> 'PrizeCollectingSolution':
        """Exact solve via the SAP transformation + dual-ascent-accelerated ILP."""
        import time as _time, math as _math
        from .pc_transform import map_sap_solution_to_pcstp
        from .dual_ascent import dual_ascent as _run_da, reduced_cost_fixing, steiner_cuts
        from .mathematical_model import solve_sap_highs, solve_sap_gurobi

        ctx = self._build_pc_transform()
        view = DirectedSteinerProblem(ctx.sap_graph, ctx.root, ctx.terminals, weight=ctx.weight)

        t0 = _time.time()
        da = _run_da(view, ctx.weight)
        fixing = da_primal = da_cuts = da_ub = None
        if da.feasible and not _math.isinf(da.lower_bound) and not _math.isinf(da.upper_bound):
            if abs(da.upper_bound - da.lower_bound) <= 1e-6 * max(1.0, abs(da.upper_bound)):
                # Proven optimal by dual ascent — skip the ILP entirely.
                edges, nodes, pcstp_obj = map_sap_solution_to_pcstp(ctx, da.primal_edges)
                return self._pc_finalize(ctx, edges, nodes, pcstp_obj, 0.0, _time.time() - t0)
            fixing = reduced_cost_fixing(view, da)
            da_primal = da.primal_edges
            da_cuts = steiner_cuts(da, view.roots)
            da_ub = da.upper_bound

        solve = solve_sap_gurobi if solver == "gurobi" else solve_sap_highs
        gap, _rt, _sap_obj, sap_arcs = solve(
            view, time_limit=time_limit, logfile=log_file,
            fixing=fixing, da_cuts=da_cuts, da_ub=da_ub, primal=da_primal,
            threads=threads,
        )

        edges, nodes, pcstp_obj = map_sap_solution_to_pcstp(ctx, sap_arcs)
        return self._pc_finalize(ctx, edges, nodes, pcstp_obj, gap, _time.time() - t0)

    def _pc_heuristic_solution(self) -> 'PrizeCollectingSolution':
        """Heuristic-only mode: the dual-ascent SAP primal, no ILP, valid gap.

        ``gap == 0.0`` certifies the primal is provably optimal; a positive gap
        bounds how far it could be from the optimum.
        """
        import time as _time, math as _math
        from .pc_transform import map_sap_solution_to_pcstp, best_trivial_pcstp
        from .dual_ascent import dual_ascent as _run_da

        ctx = self._build_pc_transform()
        view = DirectedSteinerProblem(ctx.sap_graph, ctx.root, ctx.terminals, weight=ctx.weight)
        t0 = _time.time()
        da = _run_da(view, ctx.weight)
        if not da.feasible or _math.isinf(da.upper_bound):
            raise RuntimeError(
                "dual-ascent heuristic found no feasible solution for the transformed SAP."
            )
        edges, nodes, pcstp_obj = map_sap_solution_to_pcstp(ctx, da.primal_edges)
        triv_nodes, triv_obj = best_trivial_pcstp(ctx.node_prizes)
        if triv_obj < pcstp_obj - 1e-9:
            edges, nodes, pcstp_obj = [], triv_nodes, triv_obj

        if _math.isinf(da.lower_bound):
            gap = _math.inf
        else:
            lb = da.lower_bound - ctx.offset
            gap = max(0.0, (pcstp_obj - lb) / max(1.0, abs(pcstp_obj)))
        return self._pc_make_solution(ctx, edges, nodes, pcstp_obj, gap, _time.time() - t0)

    def get_solution(self, time_limit: float = 300, log_file: str = "",
                     solver: str = "highs", pc_transform: bool = None,
                     exact: bool = True, threads: int = None) -> 'PrizeCollectingSolution':
        """Solve the prize-collecting problem.

        By default uses the penalty/Big-M flow ILP (HiGHS only). Two opt-in fast
        paths route through the classic PCSTP/MWCSP -> SAP transformation and the
        existing dual-ascent + directed-cut machinery:

        * ``exact=False`` — heuristic-only mode: return the dual-ascent SAP primal
          with no ILP; the ``Solution.gap`` is a *valid* optimality gap
          (``0.0`` certifies provable optimality).
        * ``pc_transform=True`` — exact solve via the SAP transformation with
          dual-ascent lower bound, reduced-cost fixing, cut-seeding, warm-start
          and proven-optimal early-exit.

        Both require the instance to be a classic forgo-prize PCSTP (or an MWCSP);
        otherwise a clear ``NotImplementedError`` is raised. ``solver`` selects
        ``"highs"`` (default) or ``"gurobi"`` for the transform path's ILP.
        """
        solver = solver.lower()
        use_transform = self.pc_transform if pc_transform is None else pc_transform

        if not exact:
            if not self._pc_eligible():
                raise NotImplementedError(
                    "heuristic mode (exact=False) supports classic forgo-prize "
                    "PCSTP and MWCSP only; not penalty_budget / multi-group / "
                    "penalty_cost != 0 variants."
                )
            return self._pc_heuristic_solution()

        if use_transform:
            if not self._pc_eligible():
                raise NotImplementedError(
                    "pc_transform=True supports classic forgo-prize PCSTP and "
                    "MWCSP only; not penalty_budget / multi-group / penalty_cost "
                    "!= 0 variants."
                )
            return self._pc_exact_solution(time_limit, log_file, solver, threads=threads)

        # Default: the existing penalty/Big-M flow ILP (HiGHS only), unchanged.
        model, x, y1, y2, z, f, node_vars, penalty_vars = build_prize_collecting_model(
            self, time_limit=time_limit, logfile=log_file, threads=threads
        )
        
        gap, runtime, objective, selected_edges, selected_nodes, penalties = run_prize_collecting_model(
            model, self, x, node_vars, penalty_vars
        )

        # Map solution back to original graph if preprocessing was used
        if self.preprocess:
            original_selected_edges = map_solution_to_original(selected_edges, self.reduction_tracker, self.graph)
        else:
            original_selected_edges = selected_edges

        edge_cost = sum(self.graph.edges[e][self.weight] for e in selected_edges)

        solution = PrizeCollectingSolution(
            gap=gap,
            runtime=runtime,
            objective=objective,
            selected_edges=selected_edges,
            original_selected_edges=original_selected_edges,
            selected_nodes=selected_nodes,
            penalties=penalties,
            total_prize=sum(self.node_prizes.get(node, 0) for node in selected_nodes),
            edge_cost=edge_cost,
            was_preprocessed=self.preprocess
        )

        return solution
    
class PrizeCollectingSolution(Solution):
    def __init__(self, selected_nodes: List[str] = None, penalties: Dict = None, 
                 total_prize: float = 0, edge_cost: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.selected_nodes = selected_nodes or []
        self.penalties = penalties or {}
        self.total_prize = total_prize
        self.edge_cost = edge_cost
        
    @property
    def net_value(self):
        """Total prize collected minus edge costs and penalties."""
        return self.total_prize - self.edge_cost - sum(self.penalties.values())
    
    def __repr__(self):
        return f"PrizeCollectingSolution(objective={self.objective:.2f}, prizes={self.total_prize:.2f}, nodes={len(self.selected_nodes)}, edges={len(self.edges)})"


# ---------------------------------------------------------------------------
# Node-Weighted Steiner Tree (NWST) and Maximum-Weight Connected Subgraph (MWCS)
# ---------------------------------------------------------------------------

class NodeWeightedSteinerProblem(BaseSteinerProblem):
    """
    Node-Weighted Steiner Tree Problem (NWST).

    Converts the node-weighted graph to an edge-weighted graph by splitting each
    node v into v_in and v_out connected by an edge of cost = node_weights[v].
    Then solves the standard Steiner Tree problem on the transformed graph.

    Terminal node costs are always incurred (they are part of every solution) and
    are added as a constant to the reported objective value.
    """

    def __init__(self, graph: nx.Graph, terminal_groups: List[List], node_weights: Dict,
                 weight: str = "weight", **kwargs):
        """
        :param graph: networkx undirected graph.
        :param terminal_groups: nested list of terminals.
        :param node_weights: dict mapping node -> node cost.
        :param weight: edge attribute name for weights.

        Note: graph preprocessing is not supported for node-weighted problems because
        the node-splitting transformation produces a directed graph internally.
        """
        self.node_weights = node_weights
        self.original_terminal_groups_nw = terminal_groups

        transformed_graph, new_terminal_groups, self._nw_node_map = node_split_graph(
            graph, terminal_groups, node_weights, weight=weight
        )

        # The transformed graph is a DiGraph; preprocessing is not applicable
        super().__init__(transformed_graph, new_terminal_groups, weight=weight,
                         preprocess=False, **kwargs)

    def get_solution(self, time_limit: float = 300, log_file: str = "", solver: str = "highs", threads: int = None) -> 'NodeWeightedSolution':
        """Solve and map solution back to the original node-weighted graph."""
        from .mathematical_model import build_model, run_model, build_model_gurobi, run_model_gurobi

        solver = solver.lower()
        if solver not in ("highs", "gurobi"):
            raise ValueError(f"Unknown solver '{solver}'. Choose 'highs' or 'gurobi'.")

        if solver == "gurobi":
            model, x, y1, y2, z = build_model_gurobi(self, time_limit=time_limit, logfile=log_file, threads=threads)
            gap, runtime, objective, _ = run_model_gurobi(model, self, x, y2, z)
        else:
            model, x, y1, y2, z = build_model(self, time_limit=time_limit, logfile=log_file, threads=threads)
            gap, runtime, objective, _ = run_model(model, self, x, y2, z)

        # Use arc (y1) variables for the actual directed tree structure instead of edge
        # (x) variables, to avoid degenerate zero-weight cross-edges being included.
        # Also exclude arcs pointing INTO root nodes (roots have no parents in an arborescence).
        root_nodes_split = set(self.roots)
        if solver == "gurobi":
            used_arcs = [
                (u, v) for (u, v) in self.arcs
                if y1[(u, v)].X > 0.5 and v not in root_nodes_split
            ]
        else:
            used_arcs = [
                (u, v) for (u, v) in self.arcs
                if model.variableValue(y1[(u, v)]) > 0.5 and v not in root_nodes_split
            ]

        # Terminal node costs are always incurred (not captured in the solver objective
        # because the path ends at t_in without traversing t_in -> t_out for non-root terminals)
        non_root_terminals = [
            t
            for group in self.original_terminal_groups_nw
            for t in group[1:]  # exclude root (first element) — its cost is already counted
        ]
        terminal_cost = sum(self.node_weights.get(t, 0) for t in non_root_terminals)
        adjusted_objective = objective + terminal_cost

        original_selected_edges = self._map_split_edges_to_original(used_arcs)
        selected_nodes = self._extract_original_nodes(used_arcs)

        return NodeWeightedSolution(
            gap=gap,
            runtime=runtime,
            objective=adjusted_objective,
            selected_edges=original_selected_edges,
            original_selected_edges=original_selected_edges,
            selected_nodes=selected_nodes,
        )

    def _is_node_cost_edge(self, u, v) -> bool:
        """Return True if the split-graph edge (u, v) represents a node cost."""
        orig_u = self._nw_node_map.get(u)
        orig_v = self._nw_node_map.get(v)
        return orig_u is not None and orig_u == orig_v

    def _map_split_edges_to_original(self, split_edges: List[Tuple]) -> List[Tuple]:
        """Map edges from the split graph back to edges in the original graph."""
        seen = set()
        original_edges = []
        for u, v in split_edges:
            if self._is_node_cost_edge(u, v):
                continue
            orig_u = self._nw_node_map.get(u)
            orig_v = self._nw_node_map.get(v)
            if orig_u is not None and orig_v is not None and orig_u != orig_v:
                key = tuple(sorted([str(orig_u), str(orig_v)]))
                if key not in seen:
                    seen.add(key)
                    original_edges.append((orig_u, orig_v))
        return original_edges

    def _extract_original_nodes(self, split_edges: List[Tuple]) -> List:
        """Extract original nodes from node-cost edges in the split graph, plus terminals."""
        nodes: set = set()
        for u, v in split_edges:
            if self._is_node_cost_edge(u, v):
                nodes.add(self._nw_node_map[u])
        # Terminal nodes are always part of the solution
        for group in self.original_terminal_groups_nw:
            for t in group:
                nodes.add(t)
        return sorted(nodes, key=lambda n: str(n))


class NodeWeightedSolution(Solution):
    """Solution for the Node-Weighted Steiner Tree Problem."""

    def __init__(self, selected_nodes: Optional[List] = None, **kwargs):
        super().__init__(**kwargs)
        self.selected_nodes = selected_nodes or []

    def __repr__(self):
        return (f"NodeWeightedSolution(objective={self.objective:.2f}, "
                f"nodes={len(self.selected_nodes)}, edges={len(self.edges)})")


class MaxWeightConnectedSubgraph(PrizeCollectingProblem):
    """
    Maximum-Weight Connected Subgraph (MWCS).

    Finds a connected subgraph maximising the sum of node weights.  Nodes with
    positive weights are modelled as optional terminals with prizes; nodes with
    negative weights are treated as Steiner points that are included only when
    they are necessary connectors.  A user-supplied (or automatically chosen)
    root node anchors the solution.
    """

    def __init__(self, graph: nx.Graph, node_weights: Dict, root=None,
                 weight: str = "weight", **kwargs):
        """
        :param graph: networkx undirected graph.
        :param node_weights: dict mapping node -> weight (positive or negative).
        :param root: optional root node; defaults to the highest-weight node.
        :param weight: edge attribute name for edge weights.
        """
        if root is None:
            root = max(graph.nodes(), key=lambda v: node_weights.get(v, 0))
        self.mwcs_root = root
        # Kept for the (opt-in) exact MWCSP -> PCSTP -> SAP transform path.
        self._mwcs_node_weights = dict(node_weights)

        # Nodes with positive weight become optional terminals with prizes.
        # Negative-weight nodes are left as Steiner points (no prize).
        positive_nodes = [v for v in graph.nodes() if node_weights.get(v, 0) > 0]
        all_terminals = [root] + [v for v in positive_nodes if v != root]
        terminal_groups = [all_terminals]

        node_prizes = {v: max(0.0, node_weights.get(v, 0.0)) for v in graph.nodes()}

        # Use a penalty equal to the maximum positive prize so that connecting
        # a terminal is always preferred over paying the penalty.
        max_prize = max(node_prizes.values()) if node_prizes else 1.0
        penalty_cost = max_prize + 1.0

        super().__init__(
            graph=graph,
            terminal_groups=terminal_groups,
            node_prizes=node_prizes,
            penalty_cost=penalty_cost,
            weight=weight,
            **kwargs,
        )

    # --- Opt-in transform path: MWCSP -> classic PCSTP -> SAP (exact mapping) ---

    def _pc_eligible(self) -> bool:
        """MWCSP maps exactly to a classic PCSTP, so the penalty_cost check that
        applies to native PCSTP does not apply here. Only the structural modifiers
        (budget / degree) would break the single-rooted SAP."""
        if getattr(self, 'budget', None) is not None:
            return False
        if getattr(self, 'max_degree', None) is not None:
            return False
        return True

    def _classic_pcstp(self):
        """MWCSP -> classic PCSTP (report Sec. 2.2)."""
        from .pc_transform import transform_mwcsp_to_pcstp
        pc_graph, prizes, mwcsp_const = transform_mwcsp_to_pcstp(
            self.original_graph, self._mwcs_node_weights, self.weight
        )
        return pc_graph, prizes, mwcsp_const

    def _pc_make_solution(self, ctx, edges, nodes, pcstp_obj, gap, runtime) -> 'PrizeCollectingSolution':
        """Report the MWCSP weight (sum of original node weights over the chosen
        connected subgraph) rather than the PCSTP objective."""
        mwcs_weight = ctx.mwcsp_const - pcstp_obj
        nw = self._mwcs_node_weights
        total_prize = sum(max(0.0, nw.get(v, 0.0)) for v in nodes)
        return PrizeCollectingSolution(
            gap=gap, runtime=runtime, objective=mwcs_weight,
            selected_edges=edges, original_selected_edges=edges,
            selected_nodes=nodes, penalties={}, total_prize=total_prize,
            edge_cost=0.0, was_preprocessed=False,
        )


# ---------------------------------------------------------------------------
# Degree-Constrained Steiner Tree Problem
# ---------------------------------------------------------------------------

class DegreeConstrainedSteinerProblem(SteinerProblem):
    """
    Degree-Constrained Steiner Tree Problem.

    .. deprecated:: 0.2.0
        Pass ``max_degree`` directly to :class:`SteinerProblem` (or any other
        problem class) instead::

            SteinerProblem(graph, terminal_groups, max_degree=2)

        This class is kept for backward compatibility and will be removed in a
        future version.
    """

    def __init__(self, graph: nx.Graph, terminal_groups: List[List], max_degree: int, **kwargs):
        """
        :param graph: networkx graph.
        :param terminal_groups: nested list of terminals.
        :param max_degree: maximum allowed degree for any node in the solution.
        """
        warnings.warn(
            "DegreeConstrainedSteinerProblem is deprecated. "
            "Pass max_degree directly to SteinerProblem (or any other problem class) instead: "
            "SteinerProblem(graph, terminal_groups, max_degree=max_degree).",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs['max_degree'] = max_degree
        super().__init__(graph, terminal_groups, **kwargs)


# ---------------------------------------------------------------------------
# Budget-Constrained Steiner Tree Problem
# ---------------------------------------------------------------------------

class BudgetConstrainedSteinerProblem(SteinerProblem):
    """
    Budget-Constrained Steiner Tree Problem.

    .. deprecated:: 0.2.0
        Pass ``budget`` directly to :class:`SteinerProblem` (or any other
        problem class) instead::

            SteinerProblem(graph, terminal_groups, budget=10.0)

        This class is kept for backward compatibility and will be removed in a
        future version.
    """

    def __init__(self, graph: nx.Graph, terminal_groups: List[List], budget: float, **kwargs):
        """
        :param graph: networkx graph.
        :param terminal_groups: nested list of terminals.
        :param budget: maximum total edge cost allowed.
        """
        warnings.warn(
            "BudgetConstrainedSteinerProblem is deprecated. "
            "Pass budget directly to SteinerProblem (or any other problem class) instead: "
            "SteinerProblem(graph, terminal_groups, budget=budget).",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs['budget'] = budget
        super().__init__(graph, terminal_groups, **kwargs)


class BudgetSolution(Solution):
    """Solution for the Budget-Constrained Steiner Tree Problem."""

    def __init__(self, connected_terminals: int = 0, total_terminals: int = 0,
                 penalties: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.connected_terminals = connected_terminals
        self.total_terminals = total_terminals
        self.penalties = penalties or {}

    def __repr__(self):
        return (f"BudgetSolution(connected={self.connected_terminals}/{self.total_terminals}, "
                f"edges={len(self.edges)})")


# ---------------------------------------------------------------------------
# Directed Steiner Tree Problem (Steiner Arborescence)
# ---------------------------------------------------------------------------

class DirectedSteinerProblem(BaseSteinerProblem):
    """
    Directed Steiner Tree Problem (Steiner Arborescence Problem).

    The graph is a directed graph (nx.DiGraph).  A designated root node must
    have a directed path to every terminal node.  Edges only allow flow in the
    direction they are defined; reverse arcs are not added.
    """

    def __init__(self, graph: nx.DiGraph, root, terminals: List, weight: str = "weight", **kwargs):
        """
        :param graph: networkx DiGraph.
        :param root: root node — the source of the arborescence.
        :param terminals: list of terminal nodes that must be reachable from root.
        :param weight: edge attribute for edge weights.
        """
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("DirectedSteinerProblem requires a directed graph (nx.DiGraph).")

        # Build a single terminal group with root as the first element
        terminal_group = [root] + [t for t in terminals if t != root]

        # Preprocessing is not supported for directed graphs
        kwargs['preprocess'] = False

        super().__init__(graph, [terminal_group], weight=weight, **kwargs)