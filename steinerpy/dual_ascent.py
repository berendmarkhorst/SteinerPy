"""Wong-style dual ascent + reduced-cost variable fixing for SteinerPy.

This module provides an optional accelerator for the directed-cut ("DO-D")
formulation solved in :mod:`steinerpy.mathematical_model`.  It implements

* Wong (1984) dual ascent for the Steiner arborescence directed-cut dual,
  yielding a valid **lower bound** and **reduced costs**;
* a saturation-graph **primal heuristic** yielding a feasible **upper bound**;
* **reduced-cost variable fixing** (Leitner et al. 2018): arcs/edges that
  cannot appear in any optimal solution are fixed to 0 before the ILP solve.

It covers single-group Steiner trees, multi-group Steiner **forests**
(via a sequential shared edge-budget that keeps the bound valid), and
**directed** Steiner arborescences (native Wong setting, no bidirection).

All functions are pure and operate on the *reduced* graph (``self.graph``) so
that reduced-cost fixing aligns column-for-column with the ILP variables.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple, Hashable, Optional
import math

import networkx as nx

EPS = 1e-9
Arc = Tuple[Hashable, Hashable]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class GroupAscent:
    """Dual-ascent outcome for a single rooted group."""
    root: Hashable
    terminals: List[Hashable]
    reduced_costs: Dict[Arc, float]
    lower_bound: float
    # Steiner cuts (node sets W with root not in W, a terminal in W) discovered
    # during the ascent; reused to warm-start the ILP cut loop. See
    # :func:`steiner_cuts`.
    cuts: List[FrozenSet] = field(default_factory=list)


@dataclass
class DualAscentResult:
    """Bundle returned by :func:`dual_ascent`."""
    lower_bound: float
    upper_bound: float
    primal_edges: List[Tuple]
    feasible: bool
    is_directed: bool
    arcs: List[Arc]
    edges: List[Tuple]
    groups: List[GroupAscent] = field(default_factory=list)
    residual: Optional[Dict[frozenset, float]] = None  # forest only


@dataclass
class FixingResult:
    """Variables to fix to 0 before the ILP solve."""
    fix_x_edges: Set[Tuple] = field(default_factory=set)
    fix_y1_arcs: Set[Arc] = field(default_factory=set)
    fix_y2: Dict[int, Set[Arc]] = field(default_factory=dict)

    def total(self) -> int:
        return len(self.fix_x_edges) + len(self.fix_y1_arcs) + sum(len(s) for s in self.fix_y2.values())


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _edge_cost(graph, u, v, weight: str) -> float:
    data = graph.get_edge_data(u, v)
    if data is None:
        data = graph.get_edge_data(v, u)
    if data is None:
        return 0.0
    return float(data.get(weight, 1))


def _node_universe(arcs, terminals, root) -> Set:
    nodes = set()
    for (u, v) in arcs:
        nodes.add(u)
        nodes.add(v)
    nodes.update(terminals)
    nodes.add(root)
    return nodes


def _arc_costs(graph, arcs, weight: str) -> Dict[Arc, float]:
    return {a: _edge_cost(graph, a[0], a[1], weight) for a in arcs}


# ---------------------------------------------------------------------------
# Core dual ascent (single root)
# ---------------------------------------------------------------------------

def _bfs(source, adj: Dict[Hashable, Set]) -> Set:
    """Reachable set from *source* over adjacency *adj* (including *source*)."""
    seen = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _ascent(arcs: List[Arc], root, terminals, costs: Dict[Arc, float],
            nodes: Set) -> GroupAscent:
    """Wong dual ascent on a fixed arc set with the given initial arc *costs*.

    Returns a :class:`GroupAscent` whose ``lower_bound`` is a valid lower bound
    on the min-cost arborescence rooted at ``root`` spanning ``terminals``.
    ``lower_bound == inf`` signals an unreachable terminal (disconnected).

    This is Wong (1984); the implementation keeps the saturation graph
    incrementally (saturated-arc adjacency in both directions, updated as arcs
    saturate) and uses hand-rolled BFS instead of rebuilding a ``networkx``
    graph and recomputing ``ancestors``/``descendants`` every iteration — the
    efficient-implementation idea from Duin / Pajor et al. (Ljubic 2020, sec.
    6).  It is exactly equivalent to the naive version (same lower bound,
    reduced costs, and Steiner cuts); only the per-iteration cost changes.
    """
    c = dict(costs)
    demand = set(terminals) - {root}
    lb = 0.0
    cuts: List[FrozenSet] = []
    seen: Set[FrozenSet] = set()

    # Arcs indexed by head (for delta_in), plus saturated-arc adjacency in both
    # directions, maintained incrementally as arcs saturate.
    in_arcs: Dict[Hashable, List[Arc]] = {}
    sat_succ: Dict[Hashable, Set] = {}
    sat_pred: Dict[Hashable, Set] = {}
    for a in arcs:
        in_arcs.setdefault(a[1], []).append(a)
        if c[a] <= EPS:
            sat_succ.setdefault(a[0], set()).add(a[1])
            sat_pred.setdefault(a[1], set()).add(a[0])

    while True:
        # Nodes reachable from root through saturated arcs (root included).
        reach = _bfs(root, sat_succ)
        unconnected = demand - reach
        if not unconnected:
            return GroupAscent(root, list(terminals), c, lb, cuts)

        # Wong's Steiner-cut dual: W = the set of nodes that can reach the
        # unconnected terminal t through saturated arcs (its ancestor-closure).
        # Since t is unconnected, root cannot reach t, so root not in W -> W is a
        # valid Steiner cut.  No saturated arc enters W (else its tail would also
        # reach t and be in W), so every in-cut arc has c~ > 0 and the dual
        # increment is strictly positive, guaranteeing termination.
        t = min(unconnected, key=lambda n: str(n))
        W = _bfs(t, sat_pred)

        fw = frozenset(W)
        if fw not in seen:
            seen.add(fw)
            cuts.append(fw)

        delta_in = [a for v in W for a in in_arcs.get(v, ()) if a[0] not in W]
        if not delta_in:
            # Terminal unreachable from root -> infeasible/disconnected.
            return GroupAscent(root, list(terminals), c, math.inf, cuts)

        delta = min(c[a] for a in delta_in)
        lb += delta
        for a in delta_in:
            c[a] -= delta
            if c[a] <= EPS:
                sat_succ.setdefault(a[0], set()).add(a[1])
                sat_pred.setdefault(a[1], set()).add(a[0])


def dual_ascent_arborescence(graph, arcs, root, terminals, weight="weight") -> GroupAscent:
    """Public single-root entry point (tree / directed)."""
    nodes = _node_universe(arcs, terminals, root)
    return _ascent(list(arcs), root, terminals, _arc_costs(graph, arcs, weight), nodes)


# Cap on how many candidate roots the multi-root ascent tries per group.
MAX_ROOTS = 8


def _candidate_roots(terminals, root, cap: int = MAX_ROOTS) -> List:
    """Candidate roots for multi-root ascent: the model root first (so the bound
    is never worse than the single-root one), then other terminals, capped."""
    others = sorted((t for t in terminals if t != root), key=lambda n: str(n))
    return [root] + others[: max(0, cap - 1)]


def _multi_root_group(graph, arcs: List[Arc], terminals, costs: Dict[Arc, float],
                      candidate_roots: List, weight, is_directed: bool):
    """Multi-root / multi-start dual ascent for one group on initial *costs*.

    Runs :func:`_ascent` from each candidate root and, for each, builds the
    saturation-biased primal heuristic.  Returns ``(ga_best, primal, ub)`` where:

    * ``ga_best`` is the :class:`GroupAscent` with the largest **finite** lower
      bound (every ascent is a valid dual, so the max is a valid lower bound).
      Its ``(lower_bound, reduced_costs, root)`` triple is mutually consistent —
      the caller uses it for reduced-cost fixing and cut seeding.
    * ``primal`` / ``ub`` are the **cheapest** feasible primal found across all
      roots (any feasible primal is a valid upper bound, so the minimum is too).
      A tighter upper bound yields more early-exits and stronger fixing — this is
      the main win, because the lower bound is typically already very tight.
    """
    ga_best: Optional[GroupAscent] = None
    best_primal, best_ub = None, math.inf
    for r in candidate_roots:
        nodes = _node_universe(arcs, terminals, r)
        ga = _ascent(list(arcs), r, terminals, costs, nodes)
        if ga_best is None \
                or (math.isinf(ga_best.lower_bound) and not math.isinf(ga.lower_bound)) \
                or (not math.isinf(ga.lower_bound) and ga.lower_bound > ga_best.lower_bound):
            ga_best = ga
        sat = {a for a in arcs if ga.reduced_costs[a] <= EPS}
        pe, ok = _primal_for_group(graph, arcs, r, terminals, sat, weight, is_directed)
        if ok:
            c = _edges_cost(graph, pe, weight)
            if c < best_ub:
                best_ub, best_primal = c, pe
        # Adaptive stop: once the best LB and best UB coincide the group is
        # proven optimal, and no remaining root can change either bound (its LB
        # is <= the running max, its UB >= the running min).  Skipping them is
        # therefore output-identical to running every root — it just avoids the
        # wasted work on the instances that early-exit without an ILP.
        if ga_best is not None and not math.isinf(ga_best.lower_bound) \
                and best_ub - ga_best.lower_bound <= 1e-9 * max(1.0, abs(best_ub)):
            break
    return ga_best, best_primal, best_ub


# ---------------------------------------------------------------------------
# Primal heuristic (upper bound)
# ---------------------------------------------------------------------------

def _primal_for_group(graph, arcs, root, terminals, sat: Set[Arc], weight,
                      is_directed: bool) -> Tuple[List[Tuple], bool]:
    """Shortest-path heuristic biased toward saturation arcs, with leaf pruning.

    Returns (selected_edges_or_arcs, feasible).
    """
    nodes = _node_universe(arcs, terminals, root)
    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    for a in arcs:
        w = 0.0 if a in sat else _edge_cost(graph, a[0], a[1], weight)
        H.add_edge(a[0], a[1], w=w)

    try:
        _, paths = nx.single_source_dijkstra(H, root, weight="w")
    except nx.NodeNotFound:
        return [], False

    term_set = set(terminals)
    sel_arcs: Set[Arc] = set()
    for t in terminals:
        if t == root:
            continue
        path = paths.get(t)
        if path is None:
            return [], False  # terminal unreachable
        for i in range(len(path) - 1):
            sel_arcs.add((path[i], path[i + 1]))

    # Prune non-terminal, non-root leaves (out-degree 0) to a fixpoint.
    tree = nx.DiGraph()
    tree.add_node(root)
    tree.add_edges_from(sel_arcs)
    changed = True
    while changed:
        changed = False
        for v in list(tree.nodes()):
            if v == root or v in term_set:
                continue
            if tree.out_degree(v) == 0:
                tree.remove_node(v)
                changed = True

    final_arcs = list(tree.edges())
    if is_directed:
        return final_arcs, True
    seen, edges = set(), []
    for (u, v) in final_arcs:
        key = frozenset((u, v))
        if key not in seen:
            seen.add(key)
            edges.append((u, v))
    return edges, True


def _edges_cost(graph, edges, weight) -> float:
    return sum(_edge_cost(graph, u, v, weight) for (u, v) in edges)


# ---------------------------------------------------------------------------
# Forest: sequential shared edge-budget dual ascent
# ---------------------------------------------------------------------------

def _dual_ascent_forest(steiner_problem, arcs, edges, weight) -> DualAscentResult:
    graph = steiner_problem.graph
    groups = steiner_problem.terminal_groups
    roots = steiner_problem.roots

    # Shared residual per undirected edge; both arc directions draw from it.
    residual: Dict[frozenset, float] = {
        frozenset(a): _edge_cost(graph, a[0], a[1], weight) for a in arcs
    }

    order = sorted(range(len(groups)), key=lambda k: len(groups[k]))
    group_results: Dict[int, GroupAscent] = {}
    group_primals: Dict[int, Optional[List[Tuple]]] = {}
    lb = 0.0
    for k in order:
        init_costs = {a: residual[frozenset(a)] for a in arcs}
        # Multi-root per group on the current shared residual; any root yields a
        # valid per-group dual, and the residual depletion below uses the chosen
        # ascent's own reduced costs, so Sum(loads) <= c_e still holds.  The
        # cheapest per-root primal feeds the union upper bound.
        cand = _candidate_roots(groups[k], roots[k])
        ga, primal_k, _ub_k = _multi_root_group(graph, arcs, groups[k], init_costs, cand, weight, False)
        group_results[k] = ga
        group_primals[k] = primal_k
        if math.isinf(ga.lower_bound):
            lb = math.inf
        elif not math.isinf(lb):
            lb += ga.lower_bound
        # Deplete the shared budget: consumed = max over the two directions,
        # so the new residual is the min of the two arc reduced costs.
        for key in residual:
            vals = tuple(key)
            u, v = vals[0], vals[-1]
            ruv = ga.reduced_costs.get((u, v), residual[key])
            rvu = ga.reduced_costs.get((v, u), residual[key])
            residual[key] = min(ruv, rvu)

    # Primal: union of per-group heuristics (shared edges counted once).
    union: Dict[frozenset, Tuple] = {}
    feasible = True
    for k in range(len(groups)):
        ge = group_primals[k]
        if ge is None:
            feasible = False
            continue
        for (u, v) in ge:
            union.setdefault(frozenset((u, v)), (u, v))
    primal_edges = list(union.values())
    ub = _edges_cost(graph, primal_edges, weight) if feasible else math.inf

    return DualAscentResult(
        lower_bound=lb, upper_bound=ub, primal_edges=primal_edges,
        feasible=feasible and not math.isinf(lb), is_directed=False,
        arcs=list(arcs), edges=list(edges),
        groups=[group_results[k] for k in range(len(groups))],
        residual=residual,
    )


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def dual_ascent(steiner_problem, weight: Optional[str] = None) -> DualAscentResult:
    """Run dual ascent + primal heuristic on ``steiner_problem.graph``."""
    weight = weight or steiner_problem.weight
    graph = steiner_problem.graph
    arcs = list(steiner_problem.arcs)
    edges = list(steiner_problem.edges)
    groups = steiner_problem.terminal_groups
    roots = steiner_problem.roots
    is_directed = isinstance(graph, nx.DiGraph)

    if is_directed or len(groups) == 1:
        root = roots[0]
        terminals = groups[0]
        costs = _arc_costs(graph, arcs, weight)
        # Directed arborescences have a fixed root; undirected trees may try
        # several roots and keep the tightest lower bound and primal upper bound.
        cand = [root] if is_directed else _candidate_roots(terminals, root)
        ga, primal, ub = _multi_root_group(graph, arcs, terminals, costs, cand, weight, is_directed)
        feasible = primal is not None and not math.isinf(ga.lower_bound) and not math.isinf(ub)
        if not feasible:
            primal, ub = [], math.inf
        return DualAscentResult(
            lower_bound=ga.lower_bound, upper_bound=ub, primal_edges=primal,
            feasible=feasible, is_directed=is_directed,
            arcs=arcs, edges=edges, groups=[ga], residual=None,
        )

    return _dual_ascent_forest(steiner_problem, arcs, edges, weight)


# ---------------------------------------------------------------------------
# Reduced-cost variable fixing
# ---------------------------------------------------------------------------

def _dist_from_root(arcs, costs, root, nodes) -> Dict:
    Gr = nx.DiGraph()
    Gr.add_nodes_from(nodes)
    for a in arcs:
        Gr.add_edge(a[0], a[1], w=max(0.0, costs[a]))
    return nx.single_source_dijkstra_path_length(Gr, root, weight="w")


def _dist_to_terminals(arcs, costs, terminals, nodes) -> Dict:
    """min_t d(j, t) for every node j, via a super-source on the reversed graph."""
    Grev = nx.DiGraph()
    Grev.add_nodes_from(nodes)
    for a in arcs:
        Grev.add_edge(a[1], a[0], w=max(0.0, costs[a]))
    src = object()  # unique sentinel super-source
    Grev.add_node(src)
    for t in terminals:
        Grev.add_edge(src, t, w=0.0)
    dist = nx.single_source_dijkstra_path_length(Grev, src, weight="w")
    dist.pop(src, None)
    return dist


def _fixable_arcs_single(arcs, costs, root, terminals, lb, ub) -> Set[Arc]:
    nodes = _node_universe(arcs, terminals, root)
    d_from = _dist_from_root(arcs, costs, root, nodes)
    d_to = _dist_to_terminals(arcs, costs, terminals, nodes)
    fix = set()
    for a in arcs:
        i, j = a
        incur = d_from.get(i, math.inf) + costs[a] + d_to.get(j, math.inf)
        if lb + incur > ub + EPS:
            fix.add(a)
    return fix


def _fixable_edges_forest(graph, arcs, edges, groups, roots, residual, lb, ub) -> Set[Tuple]:
    # Residual reduced cost on both directions of each edge.
    rc = {a: residual[frozenset(a)] for a in arcs}
    all_terms = [t for g in groups for t in g]
    nodes = _node_universe(arcs, all_terms, roots[0])
    d_from = {k: _dist_from_root(arcs, rc, roots[k], nodes) for k in range(len(groups))}
    d_to = {k: _dist_to_terminals(arcs, rc, groups[k], nodes) for k in range(len(groups))}

    fix = set()
    for e in edges:
        u, v = e
        re = residual[frozenset(e)]
        best = math.inf
        for k in range(len(groups)):
            for (i, j) in ((u, v), (v, u)):
                val = d_from[k].get(i, math.inf) + re + d_to[k].get(j, math.inf)
                if val < best:
                    best = val
        if lb + best > ub + EPS:
            fix.add(e)
    return fix


def reduced_cost_fixing(steiner_problem, da: DualAscentResult) -> FixingResult:
    """Compute variables fixable to 0 from a (finite, feasible) dual-ascent result."""
    result = FixingResult()
    if not da.feasible or math.isinf(da.lower_bound) or math.isinf(da.upper_bound):
        return result

    lb, ub = da.lower_bound, da.upper_bound
    arcs, edges = da.arcs, da.edges
    arc_set = set(arcs)

    if da.residual is None:
        # Single group / directed: tight arc-level fixing with arc reduced costs.
        ga = da.groups[0]
        fix_arcs = _fixable_arcs_single(arcs, ga.reduced_costs, ga.root, ga.terminals, lb, ub)
        result.fix_y1_arcs = set(fix_arcs)
        result.fix_y2 = {0: set(fix_arcs)}
        if da.is_directed:
            result.fix_x_edges = {e for e in edges if e in fix_arcs}
        else:
            result.fix_x_edges = {
                e for e in edges
                if (e[0], e[1]) in fix_arcs and (e[1], e[0]) in fix_arcs
            }
    else:
        # Forest: edge-level fixing with the (symmetric) final residual.
        fix_edges = _fixable_edges_forest(
            steiner_problem.graph, arcs, edges,
            steiner_problem.terminal_groups, steiner_problem.roots,
            da.residual, lb, ub,
        )
        result.fix_x_edges = set(fix_edges)
        fix_arcs = set()
        for e in fix_edges:
            for a in ((e[0], e[1]), (e[1], e[0])):
                if a in arc_set:
                    fix_arcs.add(a)
        result.fix_y1_arcs = fix_arcs
        result.fix_y2 = {k: set(fix_arcs) for k in range(len(steiner_problem.terminal_groups))}
    return result


# ---------------------------------------------------------------------------
# Applying fixes / warm start to the solver models
# ---------------------------------------------------------------------------

def _x_key(x, e):
    if e in x:
        return e
    rev = (e[1], e[0])
    return rev if rev in x else None


def apply_fixes_highs(model, x, y1, y2, fixing: FixingResult) -> int:
    """Fix variables to 0 in a HiGHS model via column-bound changes."""
    n = 0
    for a in fixing.fix_y1_arcs:
        if a in y1:
            model.changeColBounds(y1[a].index, 0, 0)
            n += 1
    for e in fixing.fix_x_edges:
        key = _x_key(x, e)
        if key is not None:
            model.changeColBounds(x[key].index, 0, 0)
            n += 1
    for k, group_arcs in fixing.fix_y2.items():
        for a in group_arcs:
            if (k, a) in y2:
                model.changeColBounds(y2[(k, a)].index, 0, 0)
                n += 1
    return n


def apply_fixes_gurobi(model, x, y1, y2, fixing: FixingResult) -> int:
    n = 0
    for a in fixing.fix_y1_arcs:
        if a in y1:
            y1[a].UB = 0
            n += 1
    for e in fixing.fix_x_edges:
        key = _x_key(x, e)
        if key is not None:
            x[key].UB = 0
            n += 1
    for k, group_arcs in fixing.fix_y2.items():
        for a in group_arcs:
            if (k, a) in y2:
                y2[(k, a)].UB = 0
                n += 1
    model.update()
    return n


def set_highs_warm_start(model, x, primal_edges) -> bool:
    """Best-effort MIP start over the x (edge) columns. Returns True on success."""
    try:
        import numpy as np
        selected = set()
        for (u, v) in primal_edges:
            key = _x_key(x, (u, v))
            if key is not None:
                selected.add(key)
        idx = [var.index for var in x.values()]
        val = [1.0 if e in selected else 0.0 for e in x.keys()]
        model.setSolution(len(idx),
                          np.array(idx, dtype=np.int32),
                          np.array(val, dtype=np.float64))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Cut initialization (warm-start the ILP cut loop with dual-ascent cuts)
# ---------------------------------------------------------------------------

def steiner_cuts(result: DualAscentResult, roots=None) -> List[Tuple[int, int, List[Arc]]]:
    """Convert the Steiner cuts found during dual ascent into model-ready triples.

    Each cut is a node set ``W`` (with the group root outside and a terminal
    inside). The corresponding directed-cut inequality is

        sum(y2[(k, a)] for a in delta_in(W)) >= z[(k, k)]

    where ``delta_in(W)`` are the arcs entering ``W``.  Triples are returned in
    the ``(group_id_k, group_id_l, cut_arcs)`` shape used by the ILP cut loop
    (see :func:`steinerpy.mathematical_model.run_model`), with ``l = k``; the
    group index ``k`` matches ``result.groups`` order (group 0..K-1).  Cuts with
    no entering arc are dropped and duplicates (by arc set) are removed per group.

    ``roots`` is the list of **model** roots (``steiner_problem.roots``). A cut
    ``W`` is a valid model inequality only when the model's group root is outside
    ``W``; with multi-root ascent the cut may have been discovered from a
    different root, so cuts containing the model root are filtered out here (a
    pure safety filter — dropping cuts is always sound).  When ``roots`` is
    ``None`` the ascent root is used, which the ascent guarantees is outside
    every ``W`` (backwards-compatible no-op filter).
    """
    triples: List[Tuple[int, int, List[Arc]]] = []
    arcs = result.arcs
    for k, ga in enumerate(result.groups):
        model_root = ga.root if roots is None else roots[k]
        seen: Set[FrozenSet] = set()
        for W in ga.cuts:
            if model_root in W:
                continue  # not a valid Steiner cut for the model's root
            din = [a for a in arcs if a[1] in W and a[0] not in W]
            if not din:
                continue
            key = frozenset(din)
            if key in seen:
                continue
            seen.add(key)
            triples.append((k, k, din))
    return triples


def seed_cuts_highs(model, y2, z, cuts: List[Tuple[int, int, List[Arc]]]) -> int:
    """Add dual-ascent Steiner cuts to a HiGHS model as initial constraints.

    Mirrors the cut added inside :func:`steinerpy.mathematical_model.run_model`.
    Returns the number of constraints added.
    """
    n = 0
    for (k, l, cut_arcs) in cuts:
        terms = [y2[(k, a)] for a in cut_arcs if (k, a) in y2]
        if not terms or (k, l) not in z:
            continue
        model.addConstr(sum(terms) >= z[(k, l)])
        n += 1
    return n


def seed_cuts_gurobi(model, y2, z, cuts: List[Tuple[int, int, List[Arc]]]) -> int:
    """Add dual-ascent Steiner cuts to a Gurobi model as plain (non-lazy)
    constraints before ``optimize()``, strengthening the root LP.

    Returns the number of constraints added.
    """
    import gurobipy as gp
    n = 0
    for (k, l, cut_arcs) in cuts:
        terms = [y2[(k, a)] for a in cut_arcs if (k, a) in y2]
        if not terms or (k, l) not in z:
            continue
        model.addConstr(gp.quicksum(terms) >= z[(k, l)])
        n += 1
    if n:
        model.update()
    return n


def set_highs_cutoff(model, ub: float) -> bool:
    """Best-effort objective cutoff for HiGHS. Returns True on success."""
    try:
        model.setOptionValue("objective_bound", float(ub))
        return True
    except Exception:
        return False


def set_gurobi_cutoff(model, ub: float) -> bool:
    """Best-effort objective cutoff for Gurobi. Returns True on success."""
    try:
        model.Params.Cutoff = float(ub)
        model.update()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Bound-based graph reduction (dual-ascent reduction test)
# ---------------------------------------------------------------------------

class _ProblemView:
    """Lightweight duck-typed view of a Steiner problem for :func:`dual_ascent`
    (avoids constructing a full ``SteinerProblem`` / import cycle)."""

    def __init__(self, graph, terminal_groups, weight):
        self.graph = graph
        self.weight = weight
        self.terminal_groups = terminal_groups
        self.edges = list(graph.edges())
        if isinstance(graph, nx.DiGraph):
            self.arcs = self.edges
        else:
            self.arcs = self.edges + [(v, u) for (u, v) in self.edges]
        self.roots = [g[0] for g in terminal_groups]


def _terminals_connected(graph, terminal_groups) -> bool:
    """True iff every group's terminals share one connected component (a safety
    net — reduced-cost fixing never removes an edge needed for feasibility)."""
    if graph.is_directed():
        return True
    comp: Dict = {}
    for i, cc in enumerate(nx.connected_components(graph)):
        for n in cc:
            comp[n] = i
    for grp in terminal_groups:
        ids = {comp.get(t) for t in grp}
        if None in ids or len(ids) > 1:
            return False
    return True


def reduce_graph_with_dual_ascent(graph, terminal_groups, weight, tracker,
                                  max_passes: int = 3):
    """Bound-based reduction test: delete edges that reduced-cost fixing proves
    are in **no** optimal solution, then cascade the degree-1/degree-2 reductions,
    iterating to a fixpoint.

    Only undirected graphs are reduced (directed problems skip preprocessing).
    The degree reductions are recorded in ``tracker`` so the existing solution
    back-mapping continues to work; deleted edges never appear in any solution,
    so they need no tracking.  Every removed edge is provably non-optimal, so the
    optimum is preserved; a connectivity check guards against numerical edge
    cases by aborting the offending pass.

    Returns the reduced graph (a copy; the input is never mutated).
    """
    from .graph_reducer import degree_one_reduction, degree_two_reduction

    if isinstance(graph, nx.DiGraph):
        return graph
    all_terms = {t for g in terminal_groups for t in g}
    G = graph.copy()

    for _ in range(max_passes):
        view = _ProblemView(G, terminal_groups, weight)
        da = dual_ascent(view, weight)
        if not da.feasible or math.isinf(da.lower_bound) or math.isinf(da.upper_bound):
            break
        fixing = reduced_cost_fixing(view, da)
        if not fixing.fix_x_edges:
            break

        snapshot = G.copy()
        before = (G.number_of_nodes(), G.number_of_edges())
        for e in list(G.edges()):
            if e in fixing.fix_x_edges or (e[1], e[0]) in fixing.fix_x_edges:
                G.remove_edge(*e)
        if not _terminals_connected(G, terminal_groups):
            return snapshot  # defensive: should never happen for sound fixing

        # Cascade the structural reductions enabled by the removals.
        G = degree_one_reduction(G, all_terms, tracker)
        G = degree_two_reduction(G, all_terms, weight, tracker)
        if (G.number_of_nodes(), G.number_of_edges()) == before:
            break

    return G
