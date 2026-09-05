import networkx as nx
from collections.abc import Mapping
from collections import deque
from collections.abc import MutableSet
from itertools import count
from typing import Set, Union, List, Tuple, Dict
from ._fastgraph import HAS_SCIPY, ArcCSR, dijkstra_from


class OrderedSet(MutableSet):
    """A set that iterates in insertion order.

    Several reduction passes below build worklists / deletion sets that feed
    straight back into a worklist, or into a solved MILP's own constraint
    order (see steinerpy.mathematical_model's Constraint 6/7/8) -- a plain
    `set`'s iteration order is randomized per-process by PYTHONHASHSEED, so
    using one there would make *which* of several simultaneously-valid
    reductions/contractions fires first (and hence the reduced graph's own
    shape, or which of several tied-cost solutions a MILP solver returns)
    silently depend on the process's hash seed rather than on the problem
    itself -- non-reproducible across runs of the very same input and
    `--seed`. Backed by a plain dict (whose insertion order *is* guaranteed
    by the language, regardless of hash randomization) purely for its O(1)
    membership/insertion/removal -- the dict's values are never read.

    Comparable to a plain `set`/`frozenset` via the inherited `Set.__eq__`
    (both are registered `collections.abc.Set` subclasses, so equality is
    content-based, not type-based) -- existing `x == {...}`-style assertions
    keep working unchanged even where this class replaces a real `set`.
    """

    __slots__ = ("_d",)

    def __init__(self, iterable=()):
        self._d = dict.fromkeys(iterable)

    def __contains__(self, x):
        return x in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def add(self, x):
        self._d[x] = None

    def discard(self, x):
        self._d.pop(x, None)

    def update(self, iterable):
        for x in iterable:
            self._d[x] = None

    def __repr__(self):
        return f"{type(self).__name__}({list(self._d)!r})"


class ReductionTracker:
    """Track graph reductions to enable solution mapping back to original graph."""

    def __init__(self):
        # Store (removed_node, u, w, weight_uv, weight_vw, edge_id)
        # edge_id helps distinguish between original and contracted edges
        self.degree_two_contractions = []
        self.degree_one_removals = []      # List of removed nodes (just for statistics)
        self._edge_counter = 0  # Unique ID for contracted edges
        # Fixed-edge channel (terminal contractions): reduced-graph edges proven
        # to lie in at least one optimal solution.  Their cost leaves the reduced
        # problem's objective (``fixed_cost`` must be added back), and their
        # expansions are appended to every mapped solution.
        self.fixed_edges = []              # (u, w, edge_id_or_None) at fix time
        self.fixed_cost = 0.0
        # Terminals merged away by a contraction: old terminal -> surviving node.
        self.terminal_merges: Dict = {}
        # Nodes promoted to terminals by the SL contraction (must be appended
        # to the single terminal group by the caller).
        self.added_terminals: List = []

    def add_degree_two_contraction(self, node: str, u: str, w: str, weight_uv: float, weight_vw: float, edge_id: str):
        """Record a degree-2 node contraction that actually created/modified an edge."""
        self.degree_two_contractions.append((node, u, w, weight_uv, weight_vw, edge_id))

    def add_degree_one_removal(self, node: str):
        """Record a degree-1 node removal."""
        self.degree_one_removals.append(node)

    def add_fixed_edge(self, u, w, cost: float, edge_id):
        """Record an edge fixed into every solution (terminal contraction)."""
        self.fixed_edges.append((u, w, edge_id))
        self.fixed_cost += cost

    def add_terminal_merge(self, gone, keep):
        """Record that terminal ``gone`` was merged into node ``keep``."""
        self.terminal_merges[gone] = keep

    def add_new_terminal(self, node):
        """Record a node promoted to terminal by a contraction (SL test).

        The caller's terminal groups do not contain this node; it must be
        appended to the (single) group after preprocessing.
        """
        self.added_terminals.append(node)

    def resolve_terminal(self, t):
        """Follow the merge chain of ``t`` to its surviving representative."""
        seen = set()
        while t in self.terminal_merges and t not in seen:
            seen.add(t)
            t = self.terminal_merges[t]
        return t

    def get_next_edge_id(self) -> str:
        """Generate unique ID for contracted edges."""
        self._edge_counter += 1
        return f"contracted_edge_{self._edge_counter}"


def _contract_terminal_edge(G: nx.Graph, keep, gone, weight: str,
                            terminals: OrderedSet, tracker: ReductionTracker) -> OrderedSet:
    """Fix edge ``{keep, gone}`` into every solution and merge ``gone`` into ``keep``.

    The caller guarantees the edge lies in at least one optimal solution of the
    current (reduced) problem.  The edge cost moves to the tracker's fixed-cost
    channel; every other edge ``{gone, x}`` is re-homed to ``{keep, x}`` (kept
    only when cheaper than an existing parallel) and recorded as a degree-2
    style contraction through ``gone``, so the existing back-mapping expands
    ``{keep, x}`` into ``{keep, gone} + {gone, x}`` unchanged.  ``keep``
    becomes (or stays) a terminal; a merged-away terminal is recorded so the
    caller can remap its terminal groups.

    Returns the worklist seeds (nodes whose degree changed), in
    `G.neighbors(gone)`'s own deterministic order (see `OrderedSet`'s own
    docstring for why that matters): callers feed this straight into
    `_structural_fixpoint`'s worklist.
    """
    data = G[keep][gone]
    edge_id = data.get('edge_id') if data.get('edge_type') == 'contracted' else None
    if tracker:
        tracker.add_fixed_edge(keep, gone, data.get(weight, 1), edge_id)
    seeds = OrderedSet([keep])
    for x in list(G.neighbors(gone)):
        if x == keep or x == gone:
            continue
        c_gx = G[gone][x].get(weight, 1)
        existing = G[keep][x] if G.has_edge(keep, x) else None
        if existing is None or c_gx < existing.get(weight, float("inf")):
            new_id = tracker.get_next_edge_id() if tracker else None
            attrs = {weight: c_gx, 'edge_type': 'contracted', 'edge_id': new_id}
            if existing is None:
                G.add_edge(keep, x, **attrs)
            else:
                existing.update(attrs)
            if tracker:
                tracker.add_degree_two_contraction(gone, keep, x, 0.0, c_gx, new_id)
        seeds.add(x)
    G.remove_node(gone)
    if gone in terminals:
        terminals.discard(gone)
        if tracker:
            tracker.add_terminal_merge(gone, keep)
    elif keep not in terminals and tracker:
        # Neither endpoint was a terminal (SL contraction): the merged node
        # becomes a brand-new terminal the caller must add to its group.
        tracker.add_new_terminal(keep)
    terminals.add(keep)
    return seeds


def _adjacent_terminal_pass(G: nx.Graph, terminals: OrderedSet, weight: str,
                            tracker: ReductionTracker) -> OrderedSet:
    """Contract terminal-terminal edges that are cheapest at one endpoint.

    Classic MST-style inclusion test: let ``e = {t, t'}`` with both endpoints
    terminals and ``c(e) <= c(f)`` for every ``f`` in ``delta(t)``.  Any optimal
    tree ``S`` without ``e`` contains a ``t``-``t'`` path whose first edge ``f``
    lies in ``delta(t)`` and costs at least ``c(e)``; removing ``f`` splits
    ``S`` with ``t`` on one side and ``t'`` on the other, and adding ``e``
    reconnects it at no larger cost.  Hence ``e`` is in at least one optimal
    tree and can be contracted.  Single terminal group only.

    Returns worklist seeds for the structural fixpoint (dict-as-ordered-set;
    see _contract_terminal_edge's own docstring for why).
    """
    seeds = OrderedSet()
    progress = True
    while progress and len(terminals) >= 2:
        progress = False
        for t in list(terminals):
            if t not in G or len(terminals) < 2:
                continue
            nbrs = list(G[t].items())
            if not nbrs:
                continue
            cmin = min(attr.get(weight, 1) for _x, attr in nbrs)
            cand = None
            for x, attr in nbrs:
                if x in terminals and attr.get(weight, 1) <= cmin:
                    cand = x
                    break
            if cand is None:
                continue
            seeds |= _contract_terminal_edge(G, t, cand, weight, terminals, tracker)
            progress = True
    return seeds


def _nv_pass(G: nx.Graph, terminals: OrderedSet, weight: str,
             tracker: ReductionTracker, vor) -> OrderedSet:
    """Nearest-Vertex (NV) inclusion test (Polzin & Vahdati 1998, Obs. 3.2).

    For a terminal ``t`` with cheapest incident edge ``e' = {t, v'}`` and
    second-cheapest cost ``c(e'')``: ``e'`` is in at least one minimum Steiner
    tree if there is a terminal ``tj != t`` with ``c(e') + d(v', tj) <=
    c(e'')``.  Proof (swap): an optimal tree without ``e'`` has a ``t``-``tj``
    path whose first edge ``f`` is incident to ``t``, so ``c(f) >= c(e'')``;
    removing ``f`` leaves ``t`` and ``tj`` in different components, and
    ``e'`` plus a shortest ``v'``-``tj`` path reconnects them at cost at most
    ``c(e') + d(v', tj) <= c(f)``.

    ``d(v', tj)`` is certified by the Voronoi labels from ``vor``: ``d1(v')``
    when the (merge-resolved) base differs from ``t``, else ``d2(v')`` — both
    are concrete path lengths, hence valid upper bounds on the true distance.
    Earlier contractions in the same pass only *shrink* distances (the stale
    label's path image survives node merging), so stale labels stay sound;
    bases are resolved through the tracker's merge map so ``tj != t`` is
    checked against the *current* terminal identities.  Incident-edge costs
    are re-read from ``G`` at every step.

    Returns worklist seeds (dict-as-ordered-set; see
    _contract_terminal_edge's own docstring for why); loops to a fixpoint
    internally.
    """
    d1, b1, d2, b2 = vor
    seeds = OrderedSet()
    changed = True
    while changed and len(terminals) >= 2:
        changed = False
        for t in list(terminals):
            if t not in G or len(terminals) < 2:
                continue
            inc = list(G[t].items())
            if len(inc) < 2:
                continue  # degree-1 terminals are handled by the fixpoint
            inc.sort(key=lambda kv: kv[1].get(weight, 1))
            v1, a1 = inc[0]
            c1 = a1.get(weight, 1)
            c2 = inc[1][1].get(weight, 1)
            if v1 in terminals:
                dt = 0.0
            else:
                dt = None
                b = b1.get(v1)
                if b is not None and tracker.resolve_terminal(b) != t:
                    dt = d1.get(v1)
                else:
                    b = b2.get(v1)
                    if b is not None and tracker.resolve_terminal(b) != t:
                        dt = d2.get(v1)
                if dt is None:
                    continue
            if c1 + dt <= c2:
                seeds |= _contract_terminal_edge(G, t, v1, weight, terminals, tracker)
                changed = True
    return seeds


def _sl_pass(G: nx.Graph, terminals: OrderedSet, weight: str,
             tracker: ReductionTracker, vor) -> OrderedSet:
    """Short-Links (SL) inclusion test (Polzin & Vahdati 1998, Obs. 3.3).

    A *link* of terminal ``t`` is an edge ``{u, w}`` leaving its Voronoi
    region (``base(u) = t``, ``base(w) != t``); its *length* is ``l(e) =
    d(t, u) + c(e) + d(w, base(w))``.  If every other link of ``t`` has plain
    cost at least ``l(e1)``, then ``e1`` is in at least one minimum Steiner
    tree: an optimal tree without ``e1`` contains, on its path from ``t`` to
    any other terminal, a first link ``f`` with ``c(f) >= l(e1)``; removing
    ``f`` and inserting the chain ``t ~ u — w ~ base(w)`` (two shortest paths
    plus ``e1``, total cost ``l(e1)``) reconnects it at no larger cost.  A
    region with a *single* link contracts unconditionally (every solution
    must use it).

    The merged endpoint becomes a **new terminal** (recorded via
    ``tracker.add_new_terminal`` when neither endpoint was one).  Because the
    premise quantifies over the *current* Voronoi boundary structure, this
    pass requires a **fresh** ``vor`` and applies at most **one** contraction
    per invocation; the caller rebuilds the diagram before other passes.

    Returns worklist seeds (empty when nothing fired); see OrderedSet's own
    docstring for why not a plain `set`.
    """
    if len(terminals) < 2:
        return OrderedSet()
    d1, b1, _d2, _b2 = vor
    inf = float("inf")
    links: Dict = {}  # base -> list of (cost, length, u, w)
    for u, w, attr in G.edges(data=True):
        bu, bw = b1.get(u), b1.get(w)
        if bu is None or bw is None or bu == bw:
            continue
        c = attr.get(weight, 1)
        links.setdefault(bu, []).append((c, d1[u] + c + d1[w], u, w))
        links.setdefault(bw, []).append((c, d1[w] + c + d1[u], w, u))

    best = None  # (length, u, w)
    for t, ls in links.items():
        if t not in terminals:
            continue  # stale base (should not happen on a fresh diagram)
        if len(ls) == 1:
            _c, length, u, w = ls[0]
            cand = (length, u, w)
        else:
            # Sort by cost via indices (node labels of mixed types must never
            # be compared on ties).
            order = sorted(range(len(ls)), key=lambda i: ls[i][0])
            i_min = order[0]
            c_min1, c_min2 = ls[order[0]][0], ls[order[1]][0]
            cand = None
            for i, (c, length, u, w) in enumerate(ls):
                # threshold: min cost among the OTHER links of this region
                thr = c_min2 if i == i_min else c_min1
                if length <= thr and (cand is None or length < cand[0]):
                    cand = (length, u, w)
        if cand is not None and (best is None or cand[0] < best[0]):
            best = cand

    if best is None:
        return OrderedSet()
    _length, u, w = best
    if not G.has_edge(u, w):
        return OrderedSet()
    return _contract_terminal_edge(G, u, w, weight, terminals, tracker)


def _voronoi_radii(G: nx.Graph, d1: Dict, b1: Dict, weight: str) -> Dict:
    """Voronoi radius per terminal: cheapest way to leave its region.

    ``radius(t) = min over edges {u, w} with base(u) = t != base(w) of
    d(t, u) + c(u, w)`` (Polzin & Vahdati 1998; Rehfeldt master thesis §2.2.4).
    """
    radius: Dict = {}
    for u, w, attr in G.edges(data=True):
        bu, bw = b1.get(u), b1.get(w)
        if bu is None or bw is None or bu == bw:
            continue
        c = attr.get(weight, 1)
        if d1[u] + c < radius.get(bu, float("inf")):
            radius[bu] = d1[u] + c
        if d1[w] + c < radius.get(bw, float("inf")):
            radius[bw] = d1[w] + c
    return radius


def _sph_upper_bound(G: nx.Graph, terminals, weight: str) -> float:
    """Cost of a Mehlhorn shortest-path-heuristic tree (``inf`` on failure)."""
    try:
        from networkx.algorithms.approximation import steiner_tree
        tree = steiner_tree(G, list(terminals), weight=weight, method="mehlhorn")
        if not all(t in tree for t in terminals):
            return float("inf")
        return sum(d.get(weight, 1) for _u, _v, d in tree.edges(data=True))
    except Exception:
        return float("inf")


def bound_based_deletions(G: nx.Graph, terminals: OrderedSet, weight: str,
                          vor, tmst=None, eps: float = 1e-9
                          ) -> Tuple[OrderedSet, OrderedSet]:
    """Bound-based (BND) node and edge deletions.

    Implements Observations 3.5 and 3.6 of Polzin & Vahdati Daneshmand (1998;
    also Rehfeldt master thesis, Lemmata 10/11): with the terminals' Voronoi
    *radii* sorted ascending and ``R = sum of the smallest s-2 radii``,

    * any minimum Steiner tree through non-terminal ``v`` costs at least
      ``d1(v) + d2(v) + R``;
    * any minimum Steiner tree through edge ``e = {u, w}`` costs at least
      ``c(e) + d1(u) + d1(w) + R``.

    A node/edge whose bound strictly exceeds a *feasible* upper bound (a
    Mehlhorn SPH tree on the current graph) is in no needed optimal solution
    and is deleted (a vertex appearing only as a removable zero-cost leaf of
    some optimum is likewise safe to delete).  Distances here are lower
    bounds by construction, so the test requires the Voronoi data to be
    **fresh** for the current graph.  Single terminal group only.

    (The Lemma-14 terminal-MST strengthening of the master thesis is NOT
    implemented: its cost function on the terminal graph is not recoverable
    from our copy of the text, and the natural boundary-path-length reading is
    provably not a lower bound — a 3-terminal star with unit edges has
    optimum 3 but boundary-MST weight 4.)

    :returns: ``(nodes_to_delete, edges_to_delete)``, each in
        G.nodes()/G.edges()'s own deterministic order (see `OrderedSet`'s
        own docstring for why a plain `set` here would leak PYTHONHASHSEED
        into the worklist order preprocess_graph seeds from these
        deletions).
    """
    s = len(terminals)
    if s < 3:
        return OrderedSet(), OrderedSet()
    d1, b1, d2, _b2 = vor

    ub = _sph_upper_bound(G, terminals, weight)
    if not (ub < float("inf")):
        return OrderedSet(), OrderedSet()

    radius = _voronoi_radii(G, d1, b1, weight)
    radii = sorted(radius.get(t, float("inf")) for t in terminals)
    R = sum(radii[: s - 2])
    if not (R < float("inf")):
        return OrderedSet(), OrderedSet()

    nodes = OrderedSet()
    for v in G.nodes():
        if v in terminals:
            continue
        D1, D2 = d1.get(v), d2.get(v)
        if D1 is None or D2 is None:
            continue
        if D1 + D2 + R > ub + eps:
            nodes.add(v)

    edges = OrderedSet()
    for u, w, attr in G.edges(data=True):
        if u in nodes or w in nodes:
            continue  # already covered by the node deletion
        D1u, D1w = d1.get(u), d1.get(w)
        if D1u is None or D1w is None:
            continue
        if attr.get(weight, 1) + D1u + D1w + R > ub + eps:
            edges.add((u, w))
    return nodes, edges


def _structural_fixpoint(G: nx.Graph, terminals: OrderedSet, weight: str,
                         tracker: ReductionTracker = None, seeds=None,
                         do_deg1: bool = True, do_deg2: bool = True,
                         contract_terminals: bool = False,
                         enumeration_safe: bool = False, eps: float = 1e-9) -> bool:
    """In-place degree-1 / degree-2 fixpoint driven by a worklist.

    Instead of rescanning every node per pass, a queue holds the nodes whose
    degree may have changed; removing or contracting a node only re-enqueues
    its (former) neighbours, so a full fixpoint costs time proportional to the
    work actually performed.  ``seeds`` restricts the initial worklist (e.g. to
    the endpoints of freshly deleted edges); by default all nodes are seeded.

    With ``contract_terminals`` (single terminal group only), a degree-1
    *terminal* is also handled: its sole incident edge is in **every** feasible
    solution (the terminal must be connected and has no other edge), so the
    edge is fixed and the terminal merged into its neighbour — see
    :func:`_contract_terminal_edge`.

    With ``enumeration_safe``, a degree-2 contraction whose new path cost
    *ties* an existing parallel edge {u, w} (within ``eps``) is skipped
    instead of applied: contracting would silently discard one of two equally
    optimal ways to connect u and w through v, which is exactly the tied-
    optimum loss #43 is about. ``v`` is left in the graph unsimplified. A
    strictly-better or strictly-worse new path is unaffected (no tie, so no
    alternative is lost by keeping only the cheaper one).

    Returns True iff the graph was modified.
    """
    queue = deque(G.nodes() if seeds is None else seeds)
    queued = set(queue)
    changed = False

    def enqueue(n):
        if n in G and n not in queued:
            queue.append(n)
            queued.add(n)

    while queue:
        v = queue.popleft()
        queued.discard(v)
        if v not in G:
            continue
        if v in terminals:
            if (contract_terminals and len(terminals) >= 2
                    and G.degree(v) == 1):
                (n,) = G.neighbors(v)
                for s in _contract_terminal_edge(G, n, v, weight, terminals, tracker):
                    enqueue(s)
                changed = True
            continue
        deg = G.degree(v)
        if do_deg1 and deg == 1:
            (n,) = G.neighbors(v)
            if tracker:
                tracker.add_degree_one_removal(v)
            G.remove_node(v)
            changed = True
            enqueue(n)
        elif do_deg2 and deg == 2:
            nbrs = list(G.neighbors(v))
            if len(nbrs) != 2:  # self-loop safety (degree counts loops twice)
                continue
            u, w = nbrs
            weight_uv = G[u][v].get(weight, 1)
            weight_vw = G[v][w].get(weight, 1)
            new_weight = weight_uv + weight_vw
            if enumeration_safe and G.has_edge(u, w):
                existing_weight = G[u][w].get(weight, float('inf'))
                if abs(new_weight - existing_weight) <= eps:
                    continue  # tied parallel alternative: keep v, don't erase it
            G.remove_node(v)
            changed = True
            if G.has_edge(u, w):
                # Parallel path: keep the cheaper of the existing edge and the
                # contracted path; record the contraction only when it wins.
                existing_weight = G[u][w].get(weight, float('inf'))
                if new_weight < existing_weight:
                    edge_id = tracker.get_next_edge_id() if tracker else None
                    G[u][w][weight] = new_weight
                    G[u][w]['edge_type'] = 'contracted'
                    G[u][w]['edge_id'] = edge_id
                    if tracker:
                        tracker.add_degree_two_contraction(v, u, w, weight_uv, weight_vw, edge_id)
            else:
                edge_id = tracker.get_next_edge_id() if tracker else None
                G.add_edge(u, w, **{
                    weight: new_weight,
                    'edge_type': 'contracted',
                    'edge_id': edge_id,
                })
                if tracker:
                    tracker.add_degree_two_contraction(v, u, w, weight_uv, weight_vw, edge_id)
            enqueue(u)
            enqueue(w)
    return changed


def degree_one_reduction(G: nx.Graph, terminals: Union[Set, List],
                        tracker: ReductionTracker = None) -> nx.Graph:
    """
    Iteratively remove degree-1 nodes that are not terminals.
    """
    reduced_graph = G.copy()
    _structural_fixpoint(reduced_graph, OrderedSet(terminals), None, tracker, do_deg2=False)
    return reduced_graph


def degree_two_reduction(G: nx.Graph, terminals: Union[Set, List], weight: str = "weight",
                        tracker: ReductionTracker = None,
                        enumeration_safe: bool = False) -> nx.Graph:
    """
    Replace degree-2 non-terminal nodes with direct edges between their neighbors.
    Only record contractions that actually create or modify edges.

    With ``enumeration_safe``, a contraction that would tie an existing
    parallel edge is skipped instead of applied (see
    :func:`_structural_fixpoint`'s own docstring).
    """
    reduced_graph = G.copy()

    # Add metadata to track which edges are contracted vs original
    for u, v in reduced_graph.edges():
        if 'edge_type' not in reduced_graph[u][v]:
            reduced_graph[u][v]['edge_type'] = 'original'

    _structural_fixpoint(reduced_graph, OrderedSet(terminals), weight, tracker, do_deg1=False,
                         enumeration_safe=enumeration_safe)
    return reduced_graph


# Stop repeating the heavy (Voronoi/MST-backed) sub-passes once a round removes
# less than this fraction of the current edges, or after this many rounds; the
# structural worklist reductions always run to a true fixpoint (they are cheap).
_HEAVY_MAX_ROUNDS = 10
_HEAVY_MIN_PROGRESS = 0.01
# Conservative fallback: rebuild the Voronoi diagram / terminal MST before the
# node-replacement sub-pass instead of reusing the per-round build (see the
# staleness argument in preprocess_graph).
_REBUILD_PER_SUBPASS = False


def preprocess_graph(G: nx.Graph, terminal_groups: List[List[str]], weight: str = "weight",
                     special_distance: bool = False, long_edge: bool = False,
                     max_settle: int = 2000, replace_nodes: bool = False,
                     contract: bool = False, bound_based: bool = False,
                     enumeration_safe: bool = False
                     ) -> Tuple[nx.Graph, ReductionTracker]:
    """Apply the structural (and, optionally, heavy) reductions to a fixpoint.

    Always applies the degree-1 and degree-2 reductions (via an in-place
    worklist, so the structural fixpoint costs time proportional to the work
    performed).  When ``special_distance`` or ``long_edge`` is set, the
    corresponding sound edge-deletion test (see :func:`heavy_edge_deletions`)
    is interleaved with the degree reductions; ``replace_nodes`` additionally
    enables the degree-k node replacement (pseudo-elimination) test.  Each
    deletion/replacement can cascade into further structural simplifications
    and vice versa.

    :param special_distance: enable the Special Distance test (Steiner *tree*
        only; automatically skipped when more than one terminal group is given).
    :param long_edge: enable the long-edge / alternative-path test (valid for
        Steiner tree and forest).
    :param max_settle: work cap for the long-edge bounded Dijkstra.
    :param replace_nodes: enable degree-k pseudo-elimination of non-terminals
        (Rehfeldt & Koch 2023, Prop. 4; Steiner *tree* only, skipped for
        multiple terminal groups).  Forced off when ``enumeration_safe`` (see
        below).
    :param contract: enable the *terminal contraction* tests (Steiner **tree**
        only, skipped for multiple terminal groups): degree-1 terminals (the
        sole incident edge is in every solution), adjacent-terminal edges
        that are cheapest at one endpoint (see
        :func:`_adjacent_terminal_pass`), the Nearest-Vertex test
        (:func:`_nv_pass`) and the Short-Links test (:func:`_sl_pass`).
        Fixed edges move their cost to ``tracker.fixed_cost`` (the caller
        must add it back to the objective); merged-away terminals are
        recorded in ``tracker.terminal_merges`` (the caller must remap its
        terminal groups via ``tracker.resolve_terminal``) and SL-promoted new
        terminals in ``tracker.added_terminals`` (the caller must append
        them to its single group).  With ``enumeration_safe``, only the
        degree-1 case still fires (see below).
    :param bound_based: enable the BND node/edge deletions
        (:func:`bound_based_deletions`; Steiner **tree** only).
    :param enumeration_safe: restrict reduction to tests that preserve the
        *complete set* of optimal solutions, not merely the optimal value —
        for use ahead of solution enumeration (steinerpy#43), where a
        reduction that keeps only one of several tied-optimal alternatives
        would silently make some optima unreachable. ``special_distance``,
        ``long_edge`` and ``bound_based`` are already exact (strict
        inequalities: they only ever delete an edge/node proven to lie in
        *no* optimal solution) and are unaffected. Non-terminal degree-1
        removal is likewise unaffected (an optimal tree always excludes a
        removable degree-1 non-terminal). Three tests are *not*
        optimum-set-preserving because their certificate only proves *at
        least one* optimum survives, not every one, so this flag adjusts
        them:

        * degree-2 contraction skips (leaves the node in place) exactly when
          the contracted path *ties* an existing parallel edge, instead of
          discarding the tied alternative (see
          :func:`_structural_fixpoint`'s own docstring);
        * node replacement (``replace_nodes``) is disabled outright, since a
          removed node's neighbour-pair reconnection is not guaranteed to
          reproduce every tied optimum through it;
        * of the terminal-contraction tests, only the degree-1 case (forced —
          the terminal's sole edge has no alternative to lose) still fires;
          the adjacent-terminal, Nearest-Vertex and Short-Links tests
          (:func:`_adjacent_terminal_pass`, :func:`_nv_pass`,
          :func:`_sl_pass`) are skipped, since each picks one edge among
          possibly several that equally satisfy its inclusion test.
    :returns: ``(reduced_graph, tracker)``.  Deleted edges need no tracking;
        the degree-2 contractions *and* node replacements are recorded for
        solution back-mapping (a replacement edge ``{u, w}`` through eliminated
        node ``v`` is exactly a ``(v, u, w)`` contraction).

    Soundness of reusing the Voronoi diagram / terminal MST within one round:
    every interleaved operation either (a) deletes an edge that is provably in
    *no* optimal solution — so later tests may still argue in the pre-deletion
    graph, whose optima all survive — or (b) removes a node while preserving
    all pairwise distances among the surviving nodes (degree-1 removal,
    degree-2 contraction, pseudo-elimination: any simple path through the
    removed node uses exactly two incident edges, which the replacement edge
    reproduces at equal cost).  The shared data is rebuilt at the top of each
    round; set ``_REBUILD_PER_SUBPASS`` to force a rebuild per sub-pass.
    """
    # Flatten terminal groups to get all terminals. An OrderedSet (not a
    # plain `set`) deliberately: `terminals` is iterated directly by
    # _adjacent_terminal_pass/_nv_pass (which terminal is *tried first* among
    # several that equally satisfy a contraction test) and by _voronoi/
    # _voronoi2 (seed order breaks exact-distance Voronoi-boundary ties) --
    # see OrderedSet's own docstring for the fuller reasoning.
    all_terminals = OrderedSet()
    for group in terminal_groups:
        all_terminals.update(group)
    single_group = len(terminal_groups) == 1

    reduced_graph = G.copy()
    tracker = ReductionTracker()
    ct_on = contract and single_group

    # Structural fixpoint first; the heavy tests then run on the smaller graph.
    _structural_fixpoint(reduced_graph, all_terminals, weight, tracker,
                         contract_terminals=ct_on, enumeration_safe=enumeration_safe)

    sd_on = special_distance and single_group
    # Node replacement's certificate only preserves *a* optimum, not every
    # tied one -- unsound for enumeration, so enumeration_safe forces it off
    # regardless of the caller's request (see this function's own docstring).
    rn_on = replace_nodes and single_group and not enumeration_safe
    bnd_on = bound_based and single_group
    rounds = 0
    while sd_on or long_edge or rn_on or ct_on or bnd_on:
        rounds += 1
        n0 = reduced_graph.number_of_nodes()
        m0 = reduced_graph.number_of_edges()
        seeds = OrderedSet()

        # Terminal contractions first: they change the terminal set, so they
        # must run before this round's Voronoi diagram / terminal MST is built.
        # Skipped under enumeration_safe: this pass picks one cheapest-edge
        # witness among possibly several tied ones (see this function's own
        # docstring); only the forced degree-1 case (inside
        # _structural_fixpoint, contract_terminals=ct_on) still fires.
        if ct_on and not enumeration_safe:
            ct_seeds = _adjacent_terminal_pass(reduced_graph, all_terminals,
                                               weight, tracker)
            if ct_seeds:
                _structural_fixpoint(reduced_graph, all_terminals, weight,
                                     tracker, seeds=ct_seeds,
                                     contract_terminals=True)

        def _build_terminal_data():
            vor = _voronoi2(reduced_graph, all_terminals, weight)
            tmst = _terminal_mst(reduced_graph, all_terminals, vor[0], vor[1], weight)
            bott = _bottleneck_from_mst(tmst, all_terminals)
            mst_weights = sorted(
                (d.get("weight", 1) for _u, _v, d in tmst.edges(data=True)),
                reverse=True,
            )
            return vor, tmst, bott, mst_weights

        # Shared per-round data for the terminal-based tests.
        vor = tmst = bott = mst_weights = None
        if sd_on or rn_on or (ct_on and not enumeration_safe) or bnd_on:
            vor, tmst, bott, mst_weights = _build_terminal_data()

        # Voronoi-based contraction tests (SL needs a fresh diagram and fires
        # at most once per round; NV tolerates the staleness its own
        # contractions introduce — see the docstrings). They shrink distances
        # and can add/merge terminals, so the terminal data is rebuilt before
        # the deletion tests, whose bounds must be *lower* bounds.
        if ct_on and not enumeration_safe and len(all_terminals) >= 2:
            ct_seeds = _sl_pass(reduced_graph, all_terminals, weight, tracker, vor)
            ct_seeds |= _nv_pass(reduced_graph, all_terminals, weight, tracker, vor)
            if ct_seeds:
                _structural_fixpoint(reduced_graph, all_terminals, weight,
                                     tracker, seeds=ct_seeds,
                                     contract_terminals=True)
                if sd_on or rn_on or bnd_on:
                    vor, tmst, bott, mst_weights = _build_terminal_data()

        # Sound edge deletions (every deleted edge is provably in no optimal
        # solution). Applied with an undo log; a connectivity guard reverts the
        # batch — which a sound deletion can never trigger, so this only
        # protects against numerical edge cases.
        dels = OrderedSet()
        bnd_nodes = OrderedSet()
        if sd_on:
            dels |= special_distance_deletions(reduced_graph, all_terminals, weight,
                                               vor=vor, bott=bott)
        if long_edge:
            dels |= long_edge_deletions(reduced_graph, weight, max_settle=max_settle)
        if bnd_on and len(all_terminals) >= 3:
            bnd_nodes, bnd_edges = bound_based_deletions(
                reduced_graph, all_terminals, weight, vor, tmst)
            dels |= bnd_edges
        if dels:
            undo = []
            for du, dv in dels:
                if reduced_graph.has_edge(du, dv):
                    undo.append((du, dv, dict(reduced_graph[du][dv])))
                    reduced_graph.remove_edge(du, dv)
            if not _groups_connected(reduced_graph, terminal_groups):
                for du, dv, attrs in undo:  # defensive: should never trigger
                    reduced_graph.add_edge(du, dv, **attrs)
            else:
                for du, dv, _attrs in undo:
                    seeds.add(du)
                    seeds.add(dv)
        if bnd_nodes:
            undo_nodes = []
            for v in bnd_nodes:
                if v in reduced_graph and v not in all_terminals:
                    undo_nodes.append(
                        (v, [(x, dict(reduced_graph[v][x]))
                             for x in reduced_graph[v]]))
                    reduced_graph.remove_node(v)
            if not _groups_connected(reduced_graph, terminal_groups):
                for v, adj in undo_nodes:  # defensive: should never trigger
                    reduced_graph.add_node(v)
                    for x, attrs in adj:
                        reduced_graph.add_edge(v, x, **attrs)
            else:
                for v, adj in undo_nodes:
                    seeds.update(x for x, _a in adj)
                    seeds.discard(v)

        # Node replacement last: it adds edges, whose survivors face next
        # round's deletion tests (and are pre-filtered by the SD bound).
        if rn_on:
            if _REBUILD_PER_SUBPASS:
                vor = _voronoi2(reduced_graph, all_terminals, weight)
                tmst = _terminal_mst(reduced_graph, all_terminals, vor[0], vor[1], weight)
                bott = _bottleneck_from_mst(tmst, all_terminals)
                mst_weights = sorted(
                    (d.get("weight", 1) for _u, _v, d in tmst.edges(data=True)),
                    reverse=True,
                )
            seeds |= _pseudo_eliminate_pass(reduced_graph, all_terminals, weight,
                                            tracker, vor, bott, mst_weights)

        if seeds:
            _structural_fixpoint(reduced_graph, all_terminals, weight, tracker,
                                 seeds=seeds, contract_terminals=ct_on,
                                 enumeration_safe=enumeration_safe)

        if len(all_terminals) <= 1:
            break  # trivial problem: the optimum is the fixed edges alone

        removed = (n0 - reduced_graph.number_of_nodes()) + \
                  (m0 - reduced_graph.number_of_edges())
        if removed <= 0 or rounds >= _HEAVY_MAX_ROUNDS:
            break
        if removed < max(1, _HEAVY_MIN_PROGRESS * m0):
            break

    return reduced_graph, tracker


def map_solution_to_original(reduced_solution_edges: List[Tuple[str, str]],
                           tracker: ReductionTracker,
                           reduced_graph: nx.Graph) -> List[Tuple[str, str]]:
    """
    Map a solution from the reduced graph back to the original graph.

    Only edges that the reduced graph itself marks as ``'contracted'`` are
    expanded; genuine original edges (and the ``preprocess=False`` pass-through)
    are returned unchanged.

    Expansion is *recursive*.  Degree-2 reduction can collapse a chain in several
    steps, e.g. for ``0-1-2-3`` node 1 is contracted into edge ``(0,2)`` and then
    node 2 into edge ``(0,3)``.  Expanding ``(0,3)`` yields ``(0,2)`` and ``(2,3)``,
    but ``(0,2)`` is itself a previously-contracted edge that no longer exists in
    the reduced graph.  We therefore walk the full contraction chain using the
    tracker (not reduced-graph metadata) for those intermediate synthetic edges.

    The result is de-duplicated (order-preserving): two replacement edges of one
    pseudo-eliminated node can expand to a shared original edge (only at cost
    ties), and no caller wants the duplicate.

    Args:
        reduced_solution_edges: List of edges in the reduced graph solution
        tracker: ReductionTracker that recorded the reductions
        reduced_graph: The reduced graph (to check edge metadata)

    Returns:
        List of edges in the original graph that correspond to the solution
    """
    # edge_id -> (removed_node, cu, cw). Edge ids are globally unique.
    id_to_contraction = {
        cid: (node, cu, cw)
        for (node, cu, cw, _weight_uv, _weight_vw, cid) in tracker.degree_two_contractions
        if cid is not None
    }
    # frozenset({cu, cw}) -> edge_id. The last contraction recorded for a pair
    # wins, mirroring the edge_id the reduced graph stored on the surviving edge.
    pair_to_id = {}
    for (node, cu, cw, _weight_uv, _weight_vw, cid) in tracker.degree_two_contractions:
        if cid is not None:
            pair_to_id[frozenset((cu, cw))] = cid

    def expand(u, w, edge_id, visited):
        """Recursively expand the contracted edge (u, w) into original edges."""
        if edge_id is None or edge_id in visited or edge_id not in id_to_contraction:
            return [(u, w)]
        visited = visited | {edge_id}
        node, _cu, _cw = id_to_contraction[edge_id]
        return _expand_sub(u, node, visited) + _expand_sub(node, w, visited)

    def _expand_sub(a, b, visited):
        """Expand a sub-edge: recurse if it is itself a contracted edge, else keep."""
        edge_id = pair_to_id.get(frozenset((a, b)))
        if edge_id is None:
            return [(a, b)]  # original (atomic) edge
        return expand(a, b, edge_id, visited)

    original_edges = []

    for edge in reduced_solution_edges:
        u, v = edge

        # Normalize edge direction to match graph storage
        if reduced_graph.has_edge(u, v):
            edge_data = reduced_graph[u][v]
        elif reduced_graph.has_edge(v, u):
            edge_data = reduced_graph[v][u]
            u, v = v, u  # Swap to match graph
        else:
            # Edge not found in reduced graph (e.g. preprocess=False) - keep as is
            original_edges.append(edge)
            continue

        # Only edges the reduced graph marks as contracted are expanded; this
        # keeps genuine original edges from being routed through the tracker map.
        if edge_data.get('edge_type') == 'contracted':
            original_edges.extend(expand(u, v, edge_data.get('edge_id'), frozenset()))
        else:
            # Original edge - keep as is
            original_edges.append(edge)

    # Fixed edges (terminal contractions) are in every solution of the original
    # problem but absent from the reduced graph; append their expansions.
    for (fu, fw, fid) in getattr(tracker, 'fixed_edges', ()):
        original_edges.extend(expand(fu, fw, fid, frozenset()))

    # De-duplicate while preserving order.
    seen: Set = set()
    deduped = []
    for e in original_edges:
        key = frozenset(e)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    return deduped


# ---------------------------------------------------------------------------
# Heavy (bound/alternative-based) edge-deletion reductions
# ---------------------------------------------------------------------------
#
# These implement provably optimum-preserving reduction tests from the
# Steiner-tree literature, complementing the simple degree-1/degree-2
# structural reductions above:
#
#   * Special Distance (bottleneck Steiner distance) test
#       Rehfeldt & Koch, "Implications, conflicts, and reductions for Steiner
#       trees", Math. Programming B 197 (2023), Theorem 1; see also Duin (1993),
#       Polzin & Vahdati Daneshmand (2001), and the survey Ljubic (2021),
#       Section 4 ("alternative-based" reduction tests).
#
#   * Long-edge / alternative-path test
#       The "an edge with a cheaper detour is in no optimal solution" criterion,
#       the cheapest special case of the Special Distance test, and the only one
#       of the two that stays valid for the Steiner *forest* generalisation.
#
#   * Degree-k node replacement (pseudo-elimination)
#       Rehfeldt & Koch (2023), Proposition 4: a non-terminal that provably has
#       degree <= 2 in at least one minimum Steiner tree is deleted and each
#       neighbour pair bridged by the two-edge path cost (see
#       :func:`_pseudo_eliminate_pass`).
#
# The deletion tests only ever *delete* edges that are provably contained in no
# optimal solution, so — exactly like the dual-ascent reduced-cost reduction —
# they need no entry in the ReductionTracker: a deleted edge can never appear
# in a mapped solution.  The degree reductions and node replacements they
# cascade into are tracked.
#
# Terminal *contraction* (fixed-edge) tests are implemented via the
# fixed-edge channel on the ReductionTracker (fixed_cost / fixed_edges /
# terminal_merges / added_terminals): degree-1 terminal contraction (inside
# _structural_fixpoint), the adjacent-terminal cheapest-edge test
# (_adjacent_terminal_pass), the Nearest-Vertex test (_nv_pass) and the
# Short-Links test (_sl_pass), the latter two following Polzin & Vahdati
# Daneshmand (1998), Observations 3.2/3.3.  Bound-based node/edge deletion
# (bound_based_deletions) implements their Observations 3.5/3.6 via Voronoi
# radii + an SPH upper bound.  Still open: the NV extension and SE
# (short-edges) tests, the NTDk analogue of Observation 3.7, and the
# terminal-MST lower-bound strengthening (Lemma 14 of the Rehfeldt master
# thesis — its cost function is not recoverable from our copy of the text).


def _groups_connected(G: nx.Graph, terminal_groups: List[List]) -> bool:
    """True iff every terminal group is internally connected in ``G``.

    Used as a defensive guard: a sound deletion can never disconnect a group, so
    a failure here aborts the offending pass instead of corrupting the optimum.
    """
    if G.number_of_nodes() == 0:
        return all(len(g) <= 1 for g in terminal_groups)
    # Union-find over the current edges.
    parent = {n: n for n in G.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in G.edges():
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    for group in terminal_groups:
        present = [t for t in group if t in parent]
        if len(present) <= 1:
            continue
        r0 = find(present[0])
        if any(find(t) != r0 for t in present[1:]):
            return False
    return True


def _voronoi(G: nx.Graph, terminals, weight: str):
    """Single multi-source Dijkstra building the terminal Voronoi diagram.

    Returns ``(dist, base)`` where ``dist[v]`` is the distance from ``v`` to its
    nearest terminal and ``base[v]`` is that terminal.  One Dijkstra over the
    whole graph replaces the |T| single-source Dijkstras the naive special
    distance computation would need.
    """
    import heapq
    from itertools import count
    dist: Dict = {}
    base: Dict = {}
    # Sequence tiebreaker: node labels of mixed types (e.g. group-Steiner
    # super-terminal strings next to int nodes) are not mutually comparable.
    ctr = count()
    pq = []
    for t in terminals:
        dist[t] = 0.0
        base[t] = t
        heapq.heappush(pq, (0.0, next(ctr), t))
    while pq:
        d, _c, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, attr in G[u].items():
            nd = d + attr.get(weight, 1)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                base[v] = base[u]
                heapq.heappush(pq, (nd, next(ctr), v))
    return dist, base


def _voronoi2(G: nx.Graph, terminals, weight: str):
    """Two-label multi-source Dijkstra: nearest and second-nearest terminal.

    Returns ``(d1, b1, d2, b2)``: per node the distance/terminal of the nearest
    terminal and of the nearest terminal *distinct from* ``b1`` (missing from
    ``d2``/``b2`` when no second terminal is reachable).  ``(d1, b1)`` matches
    :func:`_voronoi`.  Each node finalises at most two labels with distinct
    bases — the standard k-nearest-distinct-source Dijkstra — so the cost is
    about twice a single multi-source Dijkstra.
    """
    import heapq
    labels: Dict = {}  # v -> list[(dist, base)], ascending, distinct bases, <= 2
    pq = []
    ctr = count()  # tie-break so heterogeneous node labels never get compared
    for t in terminals:
        heapq.heappush(pq, (0.0, next(ctr), t, t))
    while pq:
        d, _, v, b = heapq.heappop(pq)
        lab = labels.setdefault(v, [])
        if len(lab) >= 2 or any(bb == b for _bd, bb in lab):
            continue
        lab.append((d, b))
        for u, attr in G[v].items():
            lu = labels.get(u)
            if lu is not None and (len(lu) >= 2 or any(bb == b for _bd, bb in lu)):
                continue  # prune: u already holds this base or is saturated
            heapq.heappush(pq, (d + attr.get(weight, 1), next(ctr), u, b))
    d1: Dict = {}
    b1: Dict = {}
    d2: Dict = {}
    b2: Dict = {}
    for v, lab in labels.items():
        d1[v], b1[v] = lab[0]
        if len(lab) > 1:
            d2[v], b2[v] = lab[1]
    return d1, b1, d2, b2


def _terminal_mst(G: nx.Graph, terminals, dist: Dict, base: Dict,
                  weight: str) -> nx.Graph:
    """Mehlhorn's Voronoi-boundary MST of the terminal distance network.

    Mehlhorn (1988) showed that the minimum spanning tree of the auxiliary graph
    built from the *Voronoi-boundary* edges — for each graph edge ``{u, w}`` whose
    endpoints lie in different terminal regions, a candidate terminal edge
    ``{base(u), base(w)}`` of weight ``dist(u) + c(u, w) + dist(w)`` — is a
    minimum spanning tree of the complete terminal distance network, at a cost
    of ``O(m + n log n)`` instead of ``O(|T| · (m + n log n))``.
    """
    K = nx.Graph()
    for t in terminals:
        K.add_node(t)
    # Keyed by frozenset: terminal labels of mixed types (e.g. SL-promoted int
    # nodes next to string super-terminals) are not mutually orderable.
    best: Dict = {}  # frozenset({ta, tb}) -> cheapest boundary distance
    for u, w, attr in G.edges(data=True):
        bu, bw = base.get(u), base.get(w)
        if bu is None or bw is None or bu == bw:
            continue
        cand = dist[u] + attr.get(weight, 1) + dist[w]
        key = frozenset((bu, bw))
        if cand < best.get(key, float("inf")):
            best[key] = cand
    for key, wq in best.items():
        a, b = tuple(key)
        K.add_edge(a, b, weight=wq)

    return nx.minimum_spanning_tree(K, weight="weight")


class _BottleneckRow(Mapping):
    """Lazy dictionary-compatible view of one terminal's tree distances."""

    def __init__(self, index, source):
        self.index = index
        self.source = source

    def __getitem__(self, target):
        index = self.index
        dest = index.node_index[target]
        if index.component[self.source] != index.component[dest]:
            raise KeyError(target)
        return index.query(self.source, dest)

    def __iter__(self):
        index = self.index
        comp = index.component[self.source]
        return (v for v, i in index.node_index.items() if index.component[i] == comp)

    def __len__(self):
        return sum(1 for _ in self)


class _BottleneckIndex(dict):
    """Tree-path maxima with O(t log t) storage/build and O(log t) queries.

    Binary lifting stores ancestors and maximum edge weights for paths of
    length 1, 2, 4, ... . Rows are lazy, so asking many different terminal
    pairs never creates a quadratic distance cache. Supports a forest too.
    """

    def __init__(self, mst, terminals):
        super().__init__()
        self.node_index = {v: i for i, v in enumerate(mst)}
        n = len(mst)
        self.depth = [0] * n
        self.component = [-1] * n
        parent = list(range(n))
        maximum = [0.0] * n
        for root in mst:
            ri = self.node_index[root]
            if self.component[ri] != -1:
                continue
            self.component[ri] = ri
            stack = [root]
            while stack:
                u = stack.pop()
                ui = self.node_index[u]
                for v, attrs in mst[u].items():
                    vi = self.node_index[v]
                    if self.component[vi] != -1:
                        continue
                    self.component[vi] = ri
                    self.depth[vi] = self.depth[ui] + 1
                    parent[vi] = ui
                    maximum[vi] = max(0.0, attrs.get("weight", 1))
                    stack.append(v)
        self.up = [parent]
        self.maximum = [maximum]
        for _ in range(1, max(self.depth, default=0).bit_length()):
            prev, weights = self.up[-1], self.maximum[-1]
            self.up.append([prev[prev[i]] for i in range(n)])
            self.maximum.append([max(weights[i], weights[prev[i]]) for i in range(n)])
        for terminal in terminals:
            if terminal in self.node_index:
                self[terminal] = _BottleneckRow(self, self.node_index[terminal])

    def query(self, a, b):
        """Maximum weight on a path between two connected node indices."""
        answer = 0.0
        if self.depth[a] < self.depth[b]:
            a, b = b, a
        difference = self.depth[a] - self.depth[b]
        while difference:
            level = difference.bit_length() - 1
            answer = max(answer, self.maximum[level][a])
            a = self.up[level][a]
            difference -= 1 << level
        if a == b:
            return answer
        for level in range(len(self.up) - 1, -1, -1):
            ancestors = self.up[level]
            if ancestors[a] != ancestors[b]:
                weights = self.maximum[level]
                answer = max(answer, weights[a], weights[b])
                a, b = ancestors[a], ancestors[b]
        return max(answer, self.maximum[0][a], self.maximum[0][b])


def _bottleneck_from_mst(mst: nx.Graph, terminals) -> Dict:
    """Lazy ``bott[a][b]`` lookup of maximum edge weights on MST paths.

    Any minimum spanning tree preserves the original graph's minimax path
    distances. Ordinary (non-minimum) spanning trees need not preserve them.
    """
    return _BottleneckIndex(mst, terminals)


def _mehlhorn_terminal_mst(G: nx.Graph, terminals, dist: Dict, base: Dict,
                           weight: str) -> Dict:
    """Lazy terminal bottleneck queries via Mehlhorn's construction.

    Compatibility wrapper composing :func:`_terminal_mst` and
    :func:`_bottleneck_from_mst`; returns ``bott[a][b]`` for terminals ``a, b``.
    """
    return _bottleneck_from_mst(_terminal_mst(G, terminals, dist, base, weight),
                                terminals)


def _sd_bound(u, v, vor, bott) -> float:
    """Upper bound on the bottleneck Steiner distance ``s(u, v)``.

    Takes the best (smallest) bottleneck over the concrete distance-network
    paths ``u -> t_a -> ... -> t_b -> v`` for ``t_a`` among ``u``'s two nearest
    terminals and ``t_b`` among ``v``'s — each such path's bottleneck,
    ``max(d(u, t_a), b_T(t_a, t_b), d(t_b, v))``, upper-bounds ``s(u, v)``.
    """
    d1, b1, d2, b2 = vor
    inf = float("inf")
    best = inf
    for du, tu in ((d1.get(u), b1.get(u)), (d2.get(u), b2.get(u))):
        if tu is None:
            continue
        row = bott.get(tu, {})
        for dv, tv in ((d1.get(v), b1.get(v)), (d2.get(v), b2.get(v))):
            if tv is None:
                continue
            cand = max(du, row.get(tv, inf), dv)
            if cand < best:
                best = cand
    return best


def special_distance_deletions(G: nx.Graph, terminals: OrderedSet, weight: str = "weight",
                               eps: float = 1e-9, vor=None, bott=None) -> OrderedSet:
    """Edges deletable by the Special Distance (bottleneck Steiner distance) test.

    For an edge ``e = {v, w}`` let ``s(v, w)`` be the bottleneck distance between
    ``v`` and ``w`` in the distance network over ``T ∪ {v, w}`` (Rehfeldt & Koch
    2023, Sec. 2.1).  By Theorem 1, if ``s(v, w) < c(e)`` then **no** minimum
    Steiner tree contains ``e``.

    We use the sound upper bound computed by :func:`_sd_bound`: the smallest
    bottleneck over the distance-network paths through any combination of the
    **two** nearest terminals of each endpoint (the nearest-only combination
    reproduces the classic single-Voronoi bound; the second label helps when
    both endpoints share a far nearest terminal but have a good second one).
    Any value strictly below ``c(e)`` certifies deletion.

    All ingredients are obtained the *fast* way (Mehlhorn 1988): one two-label
    multi-source Dijkstra (the terminal Voronoi diagram) yields the two nearest
    terminals per node. The Voronoi-boundary MST is built in
    ``O(m + n log n)`` without a separate shortest-path search per terminal.
    Its bottleneck index takes ``O(|T| log |T|)`` space and construction time,
    with ``O(log |T|)`` per path query. ``vor`` / ``bott`` allow a caller
    that already built the data (see :func:`preprocess_graph`) to pass it in.

    The terminal-hopping bound is valid for a single terminal group (ordinary
    Steiner tree).  The caller is responsible for only invoking it in that case;
    for the Steiner forest use :func:`long_edge_deletions` instead.
    """
    # Caller must pass an OrderedSet, not a plain set/list: _voronoi2's seed
    # order breaks ties among nodes at an exact equal distance from two
    # terminals, so the given order must survive here rather than being
    # re-randomized -- see OrderedSet's own docstring for the full reasoning.
    if not terminals:
        return OrderedSet()

    if vor is None:
        vor = _voronoi2(G, terminals, weight)
    if bott is None:
        bott = _bottleneck_from_mst(
            _terminal_mst(G, terminals, vor[0], vor[1], weight), terminals)

    b1 = vor[1]
    to_delete = OrderedSet()
    for u, v, data in G.edges(data=True):
        c = data.get(weight, 1)
        if b1.get(u) is None or b1.get(v) is None:
            continue
        if _sd_bound(u, v, vor, bott) < c - eps:
            to_delete.add((u, v))
    return to_delete


# Below this many vertices the parallel long-edge test isn't worth the
# process-pool / graph-pickling overhead; run it serially instead.
_LONG_EDGE_PARALLEL_MIN_NODES = 1500


def _long_edge_for_vertex(v0):
    """Deletable incident edges of ``v0`` (worker; reads shared adjacency).

    Shared payload is ``(adj, max_settle, eps)`` where ``adj[v]`` is a tuple of
    ``(neighbour, cost)`` pairs.  Returns a list of ``(v0, w)`` edges.
    """
    import heapq
    from ._parallel import get_shared
    adj, max_settle, eps = get_shared()
    nbrs = adj.get(v0, ())
    if not nbrs:
        return ()
    cmax = max(c for _w, c in nbrs)
    dist = {v0: 0.0}
    # Heap entries carry a sequence tiebreaker: node labels of mixed types
    # (e.g. the group-Steiner super-terminal strings next to int nodes) are not
    # mutually comparable, and equal distances would otherwise compare them.
    seq = 0
    pq = [(0.0, seq, v0)]
    settled = 0
    while pq and settled < max_settle:
        d, _seq, x = heapq.heappop(pq)
        if d > dist.get(x, float("inf")):
            continue
        if d >= cmax:                       # no cheaper detour for any incident edge
            break
        settled += 1
        for y, c in adj.get(x, ()):
            nd = d + c
            if nd < dist.get(y, float("inf")):
                dist[y] = nd
                seq += 1
                heapq.heappush(pq, (nd, seq, y))
    out = []
    for w, c in nbrs:
        if dist.get(w, float("inf")) < c - eps:
            out.append((v0, w))
    return out


def _long_edge_deletions_scipy(adj, nodes, eps):
    """Cost-bounded native Dijkstra with one shared CSR graph per pass."""
    arcs = [(u, v) for u in nodes for v, _ in adj[u]]
    costs = [cost for u in nodes for _, cost in adj[u]]
    csr = ArcCSR(nodes, arcs)
    matrix = csr.build_csr(costs)
    found = OrderedSet()
    for u in nodes:
        neighbors = adj[u]
        if not neighbors:
            continue
        distances = dijkstra_from(
            matrix, csr.node_index[u], limit=max(cost for _, cost in neighbors),
        )
        for v, cost in neighbors:
            if distances[csr.node_index[v]] < cost - eps:
                found.add((u, v))
    return found


def long_edge_deletions(G: nx.Graph, weight: str = "weight",
                        max_settle: int = 2000, eps: float = 1e-9,
                        jobs: int = None) -> OrderedSet:
    """Edges deletable because a strictly cheaper detour exists in ``G \\ e``.

    If the shortest-path distance between the endpoints of ``e = {v, w}`` is below
    ``c(e)``, then any solution using ``e`` can re-route along that cheaper path
    without increasing its cost (and without affecting any other terminal group),
    so ``e`` is in no minimum solution.  With positive edge costs a path shorter
    than ``c(e)`` cannot itself contain ``e``, so the shortest-path distance in
    ``G`` already certifies an ``e``-free detour.

    This is computed the fast way recommended by Rehfeldt & Koch (2023, Sec. 2.3):
    one cost-bounded Dijkstra **per vertex** ``v0`` (not per edge) settles every
    neighbour, and any incident edge ``{v0, w}`` whose neighbour is reached more
    cheaply than ``c({v0, w})`` is deleted.  The search is bounded by ``v0``'s
    largest incident edge cost (no cheaper detour for any incident edge can lie
    beyond it) and by ``max_settle`` nodes.  This is ``O(n)`` bounded Dijkstras
    instead of ``O(m)``, and stays valid for the Steiner *forest*.

    The per-vertex tests are independent, so on large graphs they are run across
    worker processes (thesis Ch. 6.3.1 collect-then-apply) and the deletable
    edges unioned; small graphs stay serial.
    """
    from ._parallel import reduce_jobs, pmap

    # Lightweight, picklable adjacency with pre-extracted float costs.
    adj = {v: tuple((w, float(a.get(weight, 1))) for w, a in G[v].items())
           for v in G.nodes()}
    nodes = list(G.nodes())
    # Native calls amortize their setup on denser, medium-sized graphs. Keep
    # the cheaper Python search for small/sparse graphs. SciPy has no settled-
    # node limit, so use it only when max_settle already covers the whole graph;
    # otherwise retain the existing work cap and multiprocessing path.
    if (HAS_SCIPY and 128 <= len(nodes) <= max_settle
            and G.number_of_edges() >= 2 * len(nodes)):
        return _long_edge_deletions_scipy(adj, nodes, eps)
    njobs = reduce_jobs() if jobs is None else jobs
    results = pmap(_long_edge_for_vertex, nodes, njobs, (adj, max_settle, eps),
                   min_items=_LONG_EDGE_PARALLEL_MIN_NODES)

    to_delete = OrderedSet()
    for r in results:
        to_delete.update(r)
    return to_delete

def heavy_edge_deletions(G: nx.Graph, terminal_groups: List[List], weight: str = "weight",
                         special_distance: bool = True, long_edge: bool = True,
                         max_settle: int = 2000) -> OrderedSet:
    """Combined sound edge-deletion set for one reduction pass.

    Applies the Special Distance test (only when there is a single terminal
    group, i.e. an ordinary Steiner tree) and/or the long-edge test (valid for
    tree and forest), returning the union of edges that are provably in no
    optimal solution.
    """
    all_terms = OrderedSet(t for g in terminal_groups for t in g)
    dels = OrderedSet()
    if special_distance and len(terminal_groups) == 1:
        dels |= special_distance_deletions(G, all_terms, weight)
    if long_edge:
        dels |= long_edge_deletions(G, weight, max_settle=max_settle)
    return dels


# ---------------------------------------------------------------------------
# Degree-k node replacement (pseudo-elimination)
# ---------------------------------------------------------------------------

# Largest non-terminal degree considered for pseudo-elimination. Degree 3 is
# always net-shrinking (<= 3 replacement edges for 3 deleted ones); degree 4 is
# admitted only under the growth guard below.
_NTD_MAX_DEGREE = 4


def _pseudo_eliminate_pass(G: nx.Graph, terminals: OrderedSet, weight: str,
                           tracker: ReductionTracker, vor, bott,
                           mst_weights_desc: List[float],
                           eps: float = 1e-9) -> OrderedSet:
    """One pass of degree-k non-terminal replacement (Rehfeldt & Koch 2023, Prop. 4).

    Let ``Y`` be an MST of the terminal distance network (the Mehlhorn boundary
    MST is one; if its edge weights overestimate the exact distances the
    criterion only becomes more conservative), with weights sorted descending
    ``w_1 >= w_2 >= ...`` and prefix sums ``W(s)``.  A non-terminal ``v`` with
    degree ``d`` and ascending incident-cost prefix sums ``C(s)`` satisfies the
    replacement criterion iff

        for every s in {3, ..., min(d, |T|)}:   W(s-1) <= C(s) + eps.

    Checking one subset per size suffices: over all ``Δ ⊆ δ(v)`` with
    ``|Δ| = s`` the left side is constant and the right side is minimised by
    the ``s`` cheapest incident edges.  Sizes above ``|T|`` are vacuous: a
    minimum Steiner tree using ``v`` at degree ``k`` splits at ``v`` into ``k``
    terminal-containing components (all leaves are terminals), so ``k <= |T|``.
    If the criterion holds, at least one minimum Steiner tree uses ``v`` with
    degree <= 2 — removing ``v`` from a tree that used it at degree ``k``
    leaves ``k`` components reconnectable by ``k - 1`` ``Y``-edges (real
    terminal-terminal shortest paths) of total cost ``W(k-1) <= C(k) <= c(Δ)``.

    The test is **non-strict** (the literature form): it preserves *at least
    one* optimum, which is the soundness contract of this module.  Replacement
    then deletes ``v`` and bridges each neighbour pair ``{u, w}`` with an edge
    of weight ``c(v,u) + c(v,w)``, recorded exactly like a degree-2 contraction
    ``(v, u, w)`` so the existing back-mapping applies unchanged.  Candidate
    edges are pre-filtered by the SD bound (adding an edge and immediately
    SD-deleting it is sound, so it is never added), merged into cheaper
    existing parallels without recording, and — for degree 4 — the node is
    skipped entirely unless at most ``deg(v)`` genuinely new edges result.

    Single terminal group only (the reconnection argument fails across groups).
    Returns the touched neighbour nodes for worklist seeding (see
    OrderedSet's own docstring for why not a plain `set`).
    """
    if mst_weights_desc is None:
        return OrderedSet()
    # W[k] = sum of the k largest Y-MST weights.
    W = [0.0]
    for wq in mst_weights_desc:
        W.append(W[-1] + wq)
    n_terms = len(terminals)
    touched = OrderedSet()

    for v in list(G.nodes()):
        if v in terminals or v not in G:
            continue
        deg = G.degree(v)
        if deg < 3 or deg > _NTD_MAX_DEGREE:
            continue
        nbrs = list(G.neighbors(v))
        if len(nbrs) != deg:  # self-loop safety
            continue

        cost_to = {u: G[v][u].get(weight, 1) for u in nbrs}
        csum = [0.0]
        for c in sorted(cost_to.values()):
            csum.append(csum[-1] + c)

        ok = True
        for s in range(3, min(deg, n_terms) + 1):
            if s - 1 > len(mst_weights_desc):
                ok = False  # cannot certify (terminals not fully connected)
                break
            if W[s - 1] > csum[s] + eps:
                ok = False
                break
        if not ok:
            continue

        # Classify the candidate replacement edge of every neighbour pair.
        plans = []  # (kind, u, w, new_weight)
        n_add = 0
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                u, w = nbrs[i], nbrs[j]
                nw = cost_to[u] + cost_to[w]
                if _sd_bound(u, w, vor, bott) < nw - eps:
                    plans.append(("drop", u, w, nw))
                elif G.has_edge(u, w):
                    if nw < G[u][w].get(weight, float("inf")):
                        plans.append(("improve", u, w, nw))
                    else:
                        plans.append(("merge", u, w, nw))
                else:
                    plans.append(("add", u, w, nw))
                    n_add += 1
        if n_add > deg:  # growth guard (binds only for degree 4)
            continue

        G.remove_node(v)
        for kind, u, w, nw in plans:
            if kind in ("drop", "merge"):
                continue
            edge_id = tracker.get_next_edge_id() if tracker else None
            if kind == "improve":
                G[u][w][weight] = nw
                G[u][w]['edge_type'] = 'contracted'
                G[u][w]['edge_id'] = edge_id
            else:
                G.add_edge(u, w, **{weight: nw, 'edge_type': 'contracted',
                                    'edge_id': edge_id})
            if tracker:
                tracker.add_degree_two_contraction(v, u, w, cost_to[u], cost_to[w], edge_id)
        touched.update(nbrs)

    return touched


def reduction_stats(original: nx.Graph, reduced: nx.Graph) -> dict:
    """Calculate statistics about the graph reduction."""
    return {
        "original_nodes": original.number_of_nodes(),
        "original_edges": original.number_of_edges(),
        "reduced_nodes": reduced.number_of_nodes(),
        "reduced_edges": reduced.number_of_edges(),
        "nodes_removed": original.number_of_nodes() - reduced.number_of_nodes(),
        "edges_removed": original.number_of_edges() - reduced.number_of_edges(),
        "node_reduction_percent": (1 - reduced.number_of_nodes() / max(1, original.number_of_nodes())) * 100,
        "edge_reduction_percent": (1 - reduced.number_of_edges() / max(1, original.number_of_edges())) * 100
    }
