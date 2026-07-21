"""Array/CSR-backed graph primitives (scipy-accelerated, networkx fallback).

This module centralises the C-backed hot-path routines used across SteinerPy and
implements the data-structure / separation ideas from Rehfeldt's PhD thesis
(2021, TU Berlin), Ch. 6:

* an **arc-index / CSR substrate** (:class:`ArcCSR`) built once per (reduced)
  graph and reused for every inner-loop computation, so the hot loops avoid the
  per-call ``networkx`` dict-of-dict construction and name -> object hashing;
* **minimum-cut separation** via ``scipy.sparse.csgraph.maximum_flow`` +
  ``breadth_first_order`` (Ch. 6.2.4), replacing the pure-Python
  ``networkx.preflow_push``;
* **single-source shortest paths** via ``scipy.sparse.csgraph.dijkstra``,
  replacing ``networkx.single_source_dijkstra`` in the reductions / dual ascent.

The scipy routines release the GIL, which is what makes thread-based intra-solve
parallelism (separation, reductions) actually scale.  When scipy is unavailable
every routine degrades to a networkx / pure-Python fallback so the package still
imports and runs (just slower).
"""

from __future__ import annotations

import os
from typing import Dict, Hashable, List, Optional, Sequence, Set, Tuple

try:  # pragma: no cover - exercised via both branches in CI
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import (
        maximum_flow as _sp_maximum_flow,
        dijkstra as _sp_dijkstra,
        breadth_first_order as _sp_bfo,
    )
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False

Arc = Tuple[Hashable, Hashable]

# scipy's maximum_flow needs *integer* capacities. We scale the float capacities
# (y2 value + creep eps) by this factor and round.  1e6 keeps 6 significant
# digits, which is exact at integer MIP nodes (y2 in {0,1}, creep eps=1e-6 -> 1)
# and ample for fractional separation.  int64 avoids overflow.
FLOW_SCALE = 10 ** 6


def cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:  # pragma: no cover
        return 1


class ArcCSR:
    """Cached index/array substrate over a fixed node + arc set.

    Built once per (reduced) graph and reused for every max-flow / Dijkstra call.
    Holds the node<->index maps, the per-arc tail/head index arrays, and grouped
    out-arc indices for output-sensitive cut extraction.
    """

    __slots__ = (
        "nodes", "node_index", "n", "arcs", "arc_index",
        "tails", "heads", "out_by_tail",
    )

    def __init__(self, nodes: Sequence, arcs: Sequence[Arc]):
        ni: Dict[Hashable, int] = {}
        node_list: List = []
        for v in nodes:
            if v not in ni:
                ni[v] = len(node_list)
                node_list.append(v)
        arc_list = list(arcs)
        # Be defensive: arcs may reference a node not in `nodes`.
        for (u, v) in arc_list:
            if u not in ni:
                ni[u] = len(node_list); node_list.append(u)
            if v not in ni:
                ni[v] = len(node_list); node_list.append(v)

        self.nodes = node_list
        self.node_index = ni
        self.n = len(node_list)
        self.arcs = arc_list
        self.arc_index = {a: i for i, a in enumerate(arc_list)}

        tails = [ni[a[0]] for a in arc_list]
        heads = [ni[a[1]] for a in arc_list]
        # Out-arc indices grouped by tail node index (output-sensitive cuts).
        out_by_tail: List[List[int]] = [[] for _ in range(self.n)]
        for i, t in enumerate(tails):
            out_by_tail[t].append(i)
        self.out_by_tail = out_by_tail

        if HAS_SCIPY:
            self.tails = np.asarray(tails, dtype=np.int32)
            self.heads = np.asarray(heads, dtype=np.int32)
        else:  # pragma: no cover - fallback path
            self.tails = tails
            self.heads = heads

    # -- CSR capacity matrix -------------------------------------------------

    def build_int_csr(self, cap_float: Sequence[float]):
        """Build an integer-scaled ``csr_matrix`` of capacities aligned to arcs."""
        data = np.rint(np.asarray(cap_float, dtype=np.float64) * FLOW_SCALE).astype(np.int64)
        return csr_matrix((data, (self.tails, self.heads)), shape=(self.n, self.n))

    def build_csr(self, values: Sequence[float]):
        """Build a float ``csr_matrix`` aligned to arcs (used for Dijkstra)."""
        data = np.asarray(values, dtype=np.float64)
        return csr_matrix((data, (self.tails, self.heads)), shape=(self.n, self.n))

    # -- cut extraction ------------------------------------------------------

    def cut_arcs(self, source_side: Set[int]) -> List[Arc]:
        """Arcs leaving ``source_side`` (delta+(S)); output-sensitive."""
        heads = self.heads
        arcs = self.arcs
        out = self.out_by_tail
        res: List[Arc] = []
        for u in source_side:
            for ai in out[u]:
                if int(heads[ai]) not in source_side:
                    res.append(arcs[ai])
        return res


def get_arc_csr(problem) -> ArcCSR:
    """Return a cached :class:`ArcCSR` for a problem/view, building it on demand.

    Cached on the object as ``_arc_csr`` keyed by the identity of ``arcs`` so a
    later graph change (which rebuilds ``arcs``) transparently invalidates it.
    """
    arcs = problem.arcs
    cached = getattr(problem, "_arc_csr", None)
    if cached is not None and getattr(problem, "_arc_csr_key", None) == id(arcs):
        return cached
    csr = ArcCSR(problem.nodes, arcs)
    try:
        problem._arc_csr = csr
        problem._arc_csr_key = id(arcs)
    except Exception:  # pragma: no cover - object forbids attribute set
        pass
    return csr


# ---------------------------------------------------------------------------
# Minimum cut (scipy max-flow + residual reachability)
# ---------------------------------------------------------------------------

def min_cut_scipy(int_csr, source_idx: int, sink_idx: int
                  ) -> Tuple[float, Set[int], Set[int]]:
    """Minimum (source, sink) cut on an integer-capacity CSR matrix.

    :returns: ``(cut_value, source_side, back_side)`` where ``source_side`` (S)
        is the set of node indices reachable from ``source`` in the residual
        graph (the root-side min cut) and ``back_side`` is ``V \\ R`` with R the
        set of nodes that can reach ``sink`` (the terminal-side / back cut).
        ``cut_value`` is the float min-cut value (de-scaled).
    """
    res = _sp_maximum_flow(int_csr, source_idx, sink_idx)
    flow_value = res.flow_value / FLOW_SCALE

    # Residual capacities (cap - flow) over the union sparsity pattern; this
    # naturally includes reverse residual arcs (flow cancellation).
    residual = (int_csr - res.flow).tocsr()
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()

    order, _ = _sp_bfo(residual, source_idx, directed=True, return_predecessors=True)
    source_side: Set[int] = set(int(i) for i in order)

    # Back cut: nodes that can reach the sink = forward-reachable from sink in
    # the transposed residual graph.
    residual_t = residual.T.tocsr()
    order_b, _ = _sp_bfo(residual_t, sink_idx, directed=True, return_predecessors=True)
    reach_sink: Set[int] = set(int(i) for i in order_b)
    back_side: Set[int] = set(range(int_csr.shape[0])) - reach_sink

    return flow_value, source_side, back_side


# ---------------------------------------------------------------------------
# Shortest paths (scipy dijkstra)
# ---------------------------------------------------------------------------

def dijkstra_from(csr, source_indices, return_predecessors: bool = False):
    """``scipy`` single/multi-source Dijkstra wrapper (non-negative weights)."""
    return _sp_dijkstra(csr, directed=True, indices=source_indices,
                        return_predecessors=return_predecessors)
