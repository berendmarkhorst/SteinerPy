"""Prize-safe edge-deletion reductions for the prize-collecting Steiner tree.

This module implements the *prize-constrained distance* (PCD) edge-deletion test
of Rehfeldt & Koch, "On the exact solution of prize-collecting Steiner tree
problems" (ZIB 20-11, 2020), Theorem 6 / Corollary 7 / Algorithm 1.

For an edge ``e = {v, w}`` with cost ``c(e)``, if there is a *prize-constrained
walk* between ``v`` and ``w`` whose prize-constrained length is at most ``c(e)``,
then ``e`` is contained in *no* optimal solution (it can be re-routed along that
strictly-no-worse detour), so it can be deleted.  The test is computed by a
modified Dijkstra (report Algorithm 1): a path's running distance is discounted
by the prize ``p(u)`` of each intermediate vertex ``u`` (floored at 0), while the
*undiscounted* cost is kept ``<= c(e)`` and each potential terminal may be used
at most once.

Unlike the degree-1/degree-2 and special-distance reductions in
:mod:`steinerpy.graph_reducer`, PCD only ever **deletes edges** — it never
removes a vertex — so it is **prize-safe**: every node prize is preserved and the
solution back-mapping is the identity on nodes.  This is exactly why the classic
reductions are unsound for prize-collecting problems while PCD is sound.
"""

import heapq
import itertools
from typing import Dict, Set, Tuple

import networkx as nx

# Below this many edges the parallel PCD test isn't worth the process-pool /
# adjacency-pickling overhead; run it serially instead.
_PCD_PARALLEL_MIN_EDGES = 1500


def _pcd_edge_deletable_adj(adj, vstart, vend, node_prizes, eps, max_settle) -> bool:
    """Report Algorithm 1 on a dict-of-dict adjacency ``adj[v][w] = cost``.

    Runs the modified, prize-discounted Dijkstra from ``vstart`` over
    ``E \\ {vstart, vend}`` and returns ``True`` as soon as ``vend`` is reached by
    a prize-constrained walk of undiscounted cost ``<= c({vstart, vend})``.
    """
    c0 = adj[vstart][vend]
    dist: Dict = {vstart: 0.0}
    forbidden: Dict = {vstart: True}     # endpoints / consumed potential terminals
    counter = itertools.count()          # tiebreaker for non-comparable node labels
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
            if dist[v] + c_vw > c0 + eps:        # undiscounted cost must stay <= c0
                continue
            cand = dist[v] + c_vw - node_prizes.get(w, 0)
            if cand < dist.get(w, float("inf")) - eps:
                if w == vend:
                    return True
                dist[w] = max(0.0, cand)         # floor the running distance at 0
                heapq.heappush(pq, (dist[w], next(counter), w))

    return False


def _pcd_for_edge(edge):
    """Worker: is ``edge`` deletable?  Reads shared ``(adj, prizes, eps, ms)``."""
    from ._parallel import get_shared
    adj, node_prizes, eps, max_settle = get_shared()
    u, v = edge
    if _pcd_edge_deletable_adj(adj, u, v, node_prizes, eps, max_settle) or \
       _pcd_edge_deletable_adj(adj, v, u, node_prizes, eps, max_settle):
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

    adj = {v: {w: float(a.get(weight, 1)) for w, a in graph[v].items()}
           for v in graph.nodes()}
    edges = list(graph.edges())
    njobs = reduce_jobs() if jobs is None else jobs
    results = pmap(_pcd_for_edge, edges, njobs, (adj, node_prizes, eps, max_settle),
                   min_items=_PCD_PARALLEL_MIN_EDGES)
    return {e for e in results if e is not None}


def reduce_pcstp_graph(
    graph: nx.Graph,
    node_prizes: Dict,
    weight: str = "weight",
    max_passes: int = 3,
) -> nx.Graph:
    """Apply the PCD edge-deletion test to a fixpoint.

    Returns a reduced **copy** (the input is never mutated).  Because PCD removes
    no vertices, every node and prize is preserved and a PCSTP solution on the
    reduced graph is valid on the original unchanged.  Deletions can lower a
    vertex's incident-edge cost and so enable further deletions, hence the
    fixpoint loop.

    A deleted edge always has a strictly-no-worse prize-constrained detour, so it
    can never be a bridge — connectivity of any optimal tree is preserved.
    """
    G = graph.copy()
    for _ in range(max_passes):
        dels = prize_constrained_distance_deletions(G, node_prizes, weight)
        if not dels:
            break
        for u, v in dels:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
    return G
