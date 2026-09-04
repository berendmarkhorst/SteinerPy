"""Prize-safe edge-deletion reductions for the prize-collecting Steiner tree.

This module implements the *prize-constrained distance* (PCD) edge-deletion test
of Rehfeldt & Koch, "On the exact solution of prize-collecting Steiner tree
problems" (ZIB 20-11, 2020), Theorem 6 / Corollary 7 / Algorithm 1.

For an edge ``e = {v, w}`` with cost ``c(e)``, if there is a *prize-constrained
walk* between ``v`` and ``w`` whose prize-constrained length is below ``c(e)``,
then ``e`` is contained in *no* optimal solution (it can be re-routed along that
strictly-better detour), so it can be deleted.  The test is computed by a
modified Dijkstra (report Algorithm 1): a path's running distance is discounted
by the prize ``p(u)`` of each intermediate vertex ``u`` (floored at 0), while the
*undiscounted* cost is kept strictly below ``c(e)`` and each potential terminal
may be used at most once. The strict form is required when independently found
deletions are applied as a batch: Corollary 7's equality case only guarantees
an alternative optimum for one edge at a time.

The module also provides the opt-in terminal-regions lower bound of the same
paper (Definition (18), Proposition 12). Prize-bearing deletion candidates are
reported but deliberately retained until the legacy penalty model has an
objective-offset channel; this phase physically removes only certified
zero-prize nodes and certified edges.
"""

from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Dict, Optional, Set, Tuple

import networkx as nx

# Below this many edges the parallel PCD test isn't worth the process-pool /
# adjacency-pickling overhead; run it serially instead.
_PCD_PARALLEL_MIN_EDGES = 1500


@dataclass
class TerminalRegions:
    """A PCSTP terminal-regions decomposition (Rehfeldt & Koch 2020).

    ``regions[t]`` is the connected region containing exactly one proper
    potential terminal ``t``. ``unassigned`` is ``H0``; the construction keeps
    all non-proper potential terminals there. ``radii`` stores equation (18).
    """

    regions: Dict
    unassigned: Set
    radii: Dict
    proper_terminals: Set
    nonproper_terminals: Set


def _proper_potential_terminals(graph, node_prizes, weight):
    from .pc_transform import is_proper_potential_terminal

    potential = {v for v in graph if node_prizes.get(v, 0) > 0}
    proper = {
        v
        for v in potential
        if is_proper_potential_terminal(graph, v, node_prizes, weight)
    }
    return potential, proper, potential - proper


def _terminal_avoiding_distances(graph, source, potential, weight):
    """Shortest distances whose internal vertices avoid ``potential``."""
    distances = {source: 0.0}
    counter = itertools.count()
    queue = [(0.0, next(counter), source)]
    while queue:
        dist_v, _, v = heapq.heappop(queue)
        if dist_v > distances.get(v, math.inf):
            continue
        # A potential terminal may be an endpoint, but never an intermediate.
        if v in potential and v != source:
            continue
        for w, data in graph[v].items():
            candidate = dist_v + float(data.get(weight, 1))
            if candidate < distances.get(w, math.inf) - 1e-12:
                distances[w] = candidate
                heapq.heappush(queue, (candidate, next(counter), w))
    return distances


def terminal_regions_decomposition(
    graph: nx.Graph, node_prizes: Dict, weight: str = "weight"
) -> TerminalRegions:
    """Build a deterministic Voronoi terminal-regions decomposition.

    This is the valid baseline construction described immediately after
    Proposition 15 of Rehfeldt & Koch (2020): non-proper potential terminals
    form part of ``H0`` and every zero-prize vertex reachable without crossing
    another potential terminal is assigned to its nearest proper terminal.
    The more expensive region-improvement local search from that paper is left
    for a later experimental stage.
    """
    if isinstance(graph, nx.DiGraph):
        raise ValueError("terminal regions currently require an undirected graph")
    potential, proper, nonproper = _proper_potential_terminals(
        graph, node_prizes, weight
    )
    owner = {}
    distance = {}
    counter = itertools.count()
    queue = []
    for terminal in sorted(proper, key=lambda v: str(v)):
        owner[terminal] = terminal
        distance[terminal] = 0.0
        heapq.heappush(queue, (0.0, str(terminal), next(counter), terminal))

    while queue:
        dist_v, owner_key, _, v = heapq.heappop(queue)
        if dist_v > distance.get(v, math.inf) + 1e-12:
            continue
        if owner_key != str(owner.get(v)):
            continue
        root = owner[v]
        for w, data in graph[v].items():
            if w in potential and w != root:
                continue
            candidate = dist_v + float(data.get(weight, 1))
            old = distance.get(w, math.inf)
            old_owner = owner.get(w)
            better_tie = abs(candidate - old) <= 1e-12 and (
                old_owner is None or str(root) < str(old_owner)
            )
            if candidate < old - 1e-12 or better_tie:
                distance[w] = candidate
                owner[w] = root
                heapq.heappush(queue, (candidate, str(root), next(counter), w))

    regions = {terminal: set() for terminal in proper}
    for v, terminal in owner.items():
        regions[terminal].add(v)
    unassigned = set(graph) - set(owner)

    radii = {}
    for terminal in proper:
        distances = _terminal_avoiding_distances(graph, terminal, potential, weight)
        outside = set(graph) - regions[terminal]
        boundary_distance = min(
            (distances.get(v, math.inf) for v in outside), default=math.inf
        )
        radii[terminal] = min(float(node_prizes.get(terminal, 0)), boundary_distance)
    return TerminalRegions(
        regions=regions,
        unassigned=unassigned,
        radii=radii,
        proper_terminals=proper,
        nonproper_terminals=nonproper,
    )


def terminal_region_node_lower_bound(
    graph: nx.Graph,
    node_prizes: Dict,
    node,
    weight: str = "weight",
    decomposition: Optional[TerminalRegions] = None,
) -> float:
    """Proposition 12 lower bound for solutions required to contain ``node``.

    The implementation uses the Voronoi decomposition returned by
    :func:`terminal_regions_decomposition`, for which ``H^p`` is the set of all
    prize-bearing vertices. Paths therefore have no prize-bearing internal
    vertex, exactly matching the paper's distance ``d_{H^p}``.
    """
    decomposition = decomposition or terminal_regions_decomposition(
        graph, node_prizes, weight
    )
    if node in decomposition.proper_terminals:
        return -math.inf
    potential = decomposition.proper_terminals | decomposition.nonproper_terminals
    distances = _terminal_avoiding_distances(graph, node, potential, weight)
    nearest = sorted(distances[t] for t in potential if t != node and t in distances)
    if len(nearest) < 2:
        return -math.inf
    radii = sorted(decomposition.radii.values())
    n_radii = max(0, len(decomposition.proper_terminals) - 2)
    if len(radii) < n_radii:
        return -math.inf
    nonproper_prizes = sum(
        float(node_prizes.get(t, 0))
        for t in decomposition.nonproper_terminals
        if t != node
    )
    return nearest[0] + nearest[1] + sum(radii[:n_radii]) + nonproper_prizes


def pcstp_primal_upper_bound(
    graph: nx.Graph, node_prizes: Dict, weight: str = "weight"
) -> float:
    """Return a deterministic feasible PCSTP upper bound."""
    from .pc_transform import (
        best_trivial_pcstp,
        pcstp_steiner_candidate,
        refine_pcstp_tree,
    )

    _, upper_bound = best_trivial_pcstp(node_prizes)
    candidate = pcstp_steiner_candidate(graph, node_prizes, weight)
    if candidate is not None:
        _, _, objective = refine_pcstp_tree(
            graph, candidate[0], candidate[1], node_prizes, weight
        )
        upper_bound = min(upper_bound, objective)
    return float(upper_bound)


def terminal_region_bound_deletions(
    graph: nx.Graph,
    node_prizes: Dict,
    weight: str = "weight",
    upper_bound: Optional[float] = None,
    delete_edges: bool = True,
    delete_nodes: bool = False,
    protected_nodes: Optional[Set] = None,
    eps: float = 1e-9,
):
    """Return bound-certified edge/node deletion candidates.

    Node deletion is Proposition 12. Edge deletion uses the same proposition on
    an equivalent graph in which the edge is subdivided by a zero-prize node:
    every tree contains the original edge iff its subdivided image contains the
    new node. Only strict ``LB > UB`` candidates are returned.

    The returned tuple is ``(edges, zero_prize_nodes, protected_prize_nodes,
    upper_bound)``. Prize-bearing nodes can be certified by Proposition 12, but
    are deliberately reported in ``protected_prize_nodes`` rather than removed:
    SteinerPy's legacy penalty model has no constant-offset channel for their
    forgone prizes. This phase therefore preserves every prize explicitly.
    """
    if upper_bound is None:
        upper_bound = pcstp_primal_upper_bound(graph, node_prizes, weight)
    decomposition = terminal_regions_decomposition(graph, node_prizes, weight)
    edge_deletions = set()
    node_deletions = set()
    protected_prize_nodes = set()
    protected_nodes = set() if protected_nodes is None else set(protected_nodes)

    if delete_nodes:
        for v in graph:
            if v in decomposition.proper_terminals or v in protected_nodes:
                continue
            lower_bound = terminal_region_node_lower_bound(
                graph, node_prizes, v, weight, decomposition
            )
            if lower_bound > upper_bound + eps:
                if node_prizes.get(v, 0) > 0:
                    protected_prize_nodes.add(v)
                else:
                    node_deletions.add(v)

    if delete_edges:
        for edge_index, (u, v, data) in enumerate(graph.edges(data=True)):
            subdivided = graph.copy()
            subdivided.remove_edge(u, v)
            dummy = ("__pc_trd_subdivision__", edge_index)
            while dummy in subdivided:
                dummy = dummy + (edge_index,)
            half_cost = float(data.get(weight, 1)) / 2.0
            subdivided.add_edge(u, dummy, **{weight: half_cost})
            subdivided.add_edge(dummy, v, **{weight: half_cost})
            prizes = dict(node_prizes)
            prizes[dummy] = 0.0
            lower_bound = terminal_region_node_lower_bound(
                subdivided, prizes, dummy, weight
            )
            if lower_bound > upper_bound + eps:
                edge_deletions.add((u, v))
    return (
        edge_deletions,
        node_deletions,
        protected_prize_nodes,
        float(upper_bound),
    )


def _pcd_edge_deletable_adj(adj, vstart, vend, node_prizes, eps, max_settle) -> bool:
    """Report Algorithm 1 on a dict-of-dict adjacency ``adj[v][w] = cost``.

    Runs the modified, prize-discounted Dijkstra from ``vstart`` over
    ``E \\ {vstart, vend}`` and returns ``True`` as soon as ``vend`` is reached by
    a prize-constrained walk of undiscounted cost ``<= c({vstart, vend})``.
    """
    c0 = adj[vstart][vend]
    dist: Dict = {vstart: 0.0}
    forbidden: Dict = {vstart: True}  # endpoints / consumed potential terminals
    counter = itertools.count()  # tiebreaker for non-comparable node labels
    pq = [(0.0, next(counter), vstart)]
    settled = 0

    while pq and settled < max_settle:
        d, _, v = heapq.heappop(pq)
        if d > dist.get(v, float("inf")):
            continue
        settled += 1
        # A potential terminal may appear at most once in a prize-constrained walk.
        if node_prizes.get(v, 0) > 0:
            forbidden[v] = True

        for w, c_vw in adj[v].items():
            # Exclude the edge under test (walk lives in E \ {e}).
            if (v == vstart and w == vend) or (v == vend and w == vstart):
                continue
            if forbidden.get(w, False):
                continue
            # The implementation batches independent edge tests. Keep every
            # subwalk strictly below c(e), so the witness satisfies Theorem 6
            # (no optimum contains e), rather than Corollary 7 (there merely
            # exists an optimum avoiding one equality edge).
            if dist[v] + c_vw >= c0 - eps:
                continue
            # Equation (8) excludes both walk endpoints from the prize sum.
            # In particular, never discount the destination's prize: doing so
            # can falsely delete an optimal edge between prize-bearing nodes.
            prize = 0 if w == vend else node_prizes.get(w, 0)
            cand = dist[v] + c_vw - prize
            if cand < dist.get(w, float("inf")) - eps:
                if w == vend:
                    # Use Theorem 6's strict test. Corollary 7 permits deleting
                    # one equality edge because some optimum avoids it, but a
                    # collect-then-apply batch can delete several mutually
                    # substitutable equality edges and lose every optimum.
                    return True
                dist[w] = max(0.0, cand)  # floor the running distance at 0
                heapq.heappush(pq, (dist[w], next(counter), w))

    return False


def _pcd_for_edge(edge):
    """Worker: is ``edge`` deletable?  Reads shared ``(adj, prizes, eps, ms)``."""
    from ._parallel import get_shared

    adj, node_prizes, eps, max_settle = get_shared()
    u, v = edge
    if _pcd_edge_deletable_adj(
        adj, u, v, node_prizes, eps, max_settle
    ) or _pcd_edge_deletable_adj(adj, v, u, node_prizes, eps, max_settle):
        return edge
    return None


def prize_constrained_distance_deletions(
    graph: nx.Graph,
    node_prizes: Dict,
    weight: str = "weight",
    eps: float = 1e-9,
    max_settle: int = 2000,
    jobs: int = None,
) -> Set[Tuple]:
    """Edges deletable by the prize-constrained distance (PCD) test.

    Runs the restricted Algorithm 1 from *both* endpoints of each edge and marks
    the edge deletable if either direction finds a qualifying detour.  Edge-only
    and prize-safe.  The per-edge tests are independent, so on large graphs they
    run across worker processes (collect-then-apply); small graphs stay serial.

    :returns: a set of (u, v) edges that are in no optimal PCSTP solution.
    """
    from ._parallel import reduce_jobs, pmap

    adj = {
        v: {w: float(a.get(weight, 1)) for w, a in graph[v].items()}
        for v in graph.nodes()
    }
    edges = list(graph.edges())
    njobs = reduce_jobs() if jobs is None else jobs
    results = pmap(
        _pcd_for_edge,
        edges,
        njobs,
        (adj, node_prizes, eps, max_settle),
        min_items=_PCD_PARALLEL_MIN_EDGES,
    )
    return {e for e in results if e is not None}


def reduce_pcstp_graph(
    graph: nx.Graph,
    node_prizes: Dict,
    weight: str = "weight",
    max_passes: int = 3,
    bound_edges: bool = False,
    bound_nodes: bool = False,
    upper_bound: Optional[float] = None,
    protected_nodes: Optional[Set] = None,
    return_stats: bool = False,
):
    """Apply the selected prize-safe reductions to a bounded fixpoint.

    PCD is always applied, preserving the historical behavior. ``bound_edges``
    adds terminal-region edge deletion; ``bound_nodes`` also removes
    zero-prize vertices certified by Proposition 12. Both stronger options are
    experimental and default off. The input is never mutated.

    A PCD-deleted edge has a strictly better prize-constrained detour. Bound
    deletions use strict lower-bound comparisons, so the optimum is preserved.

    With ``return_stats=True``, return ``(graph, stats)``. The default remains
    the historical graph-only return value.
    """
    started = time.perf_counter()
    G = graph.copy()
    protected_prize_nodes = set()
    bound_edge_count = 0
    bound_node_count = 0
    pcd_edge_count = 0
    passes = 0
    active_upper_bound = upper_bound
    for _ in range(max_passes):
        passes += 1
        before_edges = G.number_of_edges()
        before_nodes = G.number_of_nodes()
        dels = prize_constrained_distance_deletions(G, node_prizes, weight)
        for u, v in dels:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                pcd_edge_count += 1
        if bound_edges or bound_nodes:
            edge_dels, node_dels, protected, active_upper_bound = (
                terminal_region_bound_deletions(
                    G,
                    node_prizes,
                    weight,
                    upper_bound=active_upper_bound,
                    delete_edges=bound_edges,
                    delete_nodes=bound_nodes,
                    protected_nodes=protected_nodes,
                )
            )
            protected_prize_nodes.update(protected)
            for u, v in edge_dels:
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    bound_edge_count += 1
            for v in node_dels:
                if v in G:
                    G.remove_node(v)
                    bound_node_count += 1
        if G.number_of_edges() == before_edges and G.number_of_nodes() == before_nodes:
            break

    stats = {
        "preprocessing_time": time.perf_counter() - started,
        "passes": passes,
        "nodes_removed": graph.number_of_nodes() - G.number_of_nodes(),
        "edges_removed": graph.number_of_edges() - G.number_of_edges(),
        "pcd_edges_removed": pcd_edge_count,
        "bound_edges_removed": bound_edge_count,
        "bound_nodes_removed": bound_node_count,
        "protected_prize_nodes": len(protected_prize_nodes),
        "upper_bound": active_upper_bound,
        # These reductions prove deletability but do not themselves certify a
        # complete optimum; keep the benchmark field explicit rather than
        # inferring it from an empty residual graph.
        "solved_in_preprocessing": False,
    }
    return (G, stats) if return_stats else G
