"""Exact Steiner tree dynamic program for few terminals.

Implements the Dreyfus–Wagner (1971) dynamic program in the
Erickson–Monma–Veinott (1987) formulation: labels ``l(S, v)`` — the cost of a
minimum Steiner tree spanning terminal subset ``S`` plus vertex ``v`` — are
computed by *merging* two complementary sub-labels at ``v`` and *growing*
merged labels along shortest paths.  Complexity ``O(3^k · n + 2^k · (m + n log
n))`` for ``k`` terminals, which for small ``k`` (roughly ``k <= 10``) solves
the instance outright, far faster than any ILP.  This DP + reductions is the
approach behind the winning solvers of the PACE 2018 Steiner tree challenge.

The implementation is vectorised: the merge step is elementwise ``numpy`` over
all vertices, and the grow step is one ``scipy`` Dijkstra per subset from a
*virtual source* connected to every vertex with its merged label as arc cost —
a standard trick to run Dijkstra with initial potentials.  scipy is required
(callers gate on :data:`steinerpy._fastgraph.HAS_SCIPY`).
"""

from __future__ import annotations

import os
from typing import Hashable, List, Tuple

from ._fastgraph import HAS_SCIPY

if HAS_SCIPY:  # pragma: no branch
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as _sp_dijkstra


def dw_max_terminals() -> int:
    """Terminal-count ceiling for auto-selecting the DP over the ILP.

    The DP is exponential in the terminal count (``3^k`` merge work and ``2^k``
    Dijkstra runs), so it is only auto-selected for ``k`` up to this bound.
    Configure with ``STEINERPY_DW_MAX_TERMINALS`` (``0`` disables the DP).
    """
    env = os.environ.get("STEINERPY_DW_MAX_TERMINALS")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return 10


def dreyfus_wagner(graph, terminals: List[Hashable], weight: str = "weight"
                   ) -> Tuple[float, List[Tuple]]:
    """Exact minimum Steiner tree via the Dreyfus–Wagner/EMV dynamic program.

    :param graph: undirected ``networkx`` graph with non-negative edge weights.
    :param terminals: terminals to connect (duplicates are ignored).
    :param weight: edge-weight attribute (missing weights default to 1).
    :returns: ``(cost, edges)`` of a minimum Steiner tree, with ``cost``
        recomputed from the returned edge set.  ``(inf, [])`` when the
        terminals are not connected in ``graph``.
    :raises RuntimeError: when scipy is not available.
    """
    if not HAS_SCIPY:
        raise RuntimeError("dreyfus_wagner requires scipy")

    terminals = list(dict.fromkeys(terminals))
    if len(terminals) <= 1:
        return 0.0, []

    nodes = list(graph.nodes())
    ni = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    virtual = n  # extra source node used by the grow step

    tails: List[int] = []
    heads: List[int] = []
    costs: List[float] = []
    for u, v, data in graph.edges(data=True):
        c = float(data.get(weight, 1))
        ui, vi = ni[u], ni[v]
        tails += (ui, vi)
        heads += (vi, ui)
        costs += (c, c)
    base_tails = np.asarray(tails, dtype=np.int32)
    base_heads = np.asarray(heads, dtype=np.int32)
    base_costs = np.asarray(costs, dtype=np.float64)

    t_idx = [ni[t] for t in terminals]
    root = t_idx[-1]
    base = t_idx[:-1]  # terminals carried in the subset masks
    kb = len(base)
    full = (1 << kb) - 1

    # Per subset S (index = bitmask over `base`):
    #   labels[S][v]  = l(S, v), length n+1 (the virtual entry is unused);
    #   grow_pred[S]  = predecessor array of the grow Dijkstra — `virtual`
    #                   marks the base vertex where the merged subtree sits
    #                   (for singletons: predecessors of the plain Dijkstra);
    #   split[S][v]   = the canonical submask chosen by the merge at v.
    labels = [None] * (full + 1)
    grow_pred = [None] * (full + 1)
    split = [None] * (full + 1)

    graph_csr = csr_matrix((base_costs, (base_tails, base_heads)),
                           shape=(n + 1, n + 1))

    # Base case: l({t}, v) = d(t, v) — one plain Dijkstra per base terminal.
    for i, ti in enumerate(base):
        dist, pred = _sp_dijkstra(graph_csr, directed=True, indices=ti,
                                  return_predecessors=True)
        labels[1 << i] = dist
        grow_pred[1 << i] = pred

    for mask in range(1, full + 1):
        if mask & (mask - 1) == 0:
            continue  # singleton, already done
        low = mask & -mask

        # Merge: m(S, v) = min over canonical splits S' (containing the lowest
        # set bit, so each unordered pair is tried once) of l(S') + l(S \ S').
        best = np.full(n + 1, np.inf)
        choice = np.zeros(n + 1, dtype=np.int64)
        sub = (mask - 1) & mask
        while sub:
            if sub & low:
                cand = labels[sub] + labels[mask ^ sub]
                upd = cand < best
                best[upd] = cand[upd]
                choice[upd] = sub
            sub = (sub - 1) & mask

        # Grow: l(S, v) = min_u m(S, u) + d(u, v) — a Dijkstra from a virtual
        # source whose arc to u costs m(S, u).
        finite = np.isfinite(best[:n])
        v_heads = np.nonzero(finite)[0].astype(np.int32)
        v_tails = np.full(v_heads.shape, virtual, dtype=np.int32)
        grow_csr = csr_matrix(
            (np.concatenate([base_costs, best[:n][finite]]),
             (np.concatenate([base_tails, v_tails]),
              np.concatenate([base_heads, v_heads]))),
            shape=(n + 1, n + 1),
        )
        dist, pred = _sp_dijkstra(grow_csr, directed=True, indices=virtual,
                                  return_predecessors=True)
        labels[mask] = dist
        grow_pred[mask] = pred
        split[mask] = choice

    total = labels[full][root]
    if not np.isfinite(total):
        return float("inf"), []

    # Reconstruction: walk grow predecessors back to the subtree base, then
    # expand the merge recorded there into its two sub-subsets.
    edge_keys = set()
    stack = [(full, root)]
    while stack:
        mask, v = stack.pop()
        if mask & (mask - 1) == 0:
            source = base[mask.bit_length() - 1]
            w = v
            while w != source:
                u = int(grow_pred[mask][w])
                edge_keys.add((u, w) if u < w else (w, u))
                w = u
            continue
        w = v
        while True:
            u = int(grow_pred[mask][w])
            if u == virtual:
                sub = int(split[mask][w])
                stack.append((sub, w))
                stack.append((mask ^ sub, w))
                break
            edge_keys.add((u, w) if u < w else (w, u))
            w = u

    # The union of the reconstructed subtrees attains the DP optimum, so the
    # recomputed cost equals `total` (up to float noise); return the recomputed
    # value so the objective always matches the returned edge set.
    edges: List[Tuple] = []
    cost = 0.0
    for a, b in edge_keys:
        u, v = nodes[a], nodes[b]
        cost += float(graph[u][v].get(weight, 1))
        edges.append((u, v))
    return cost, edges
