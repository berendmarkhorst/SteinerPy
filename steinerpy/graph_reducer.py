import networkx as nx
from typing import Set, Union, List, Tuple, Dict


class ReductionTracker:
    """Track graph reductions to enable solution mapping back to original graph."""
    
    def __init__(self):
        # Store (removed_node, u, w, weight_uv, weight_vw, edge_id)
        # edge_id helps distinguish between original and contracted edges
        self.degree_two_contractions = []  
        self.degree_one_removals = []      # List of removed nodes (just for statistics)
        self._edge_counter = 0  # Unique ID for contracted edges
    
    def add_degree_two_contraction(self, node: str, u: str, w: str, weight_uv: float, weight_vw: float, edge_id: str):
        """Record a degree-2 node contraction that actually created/modified an edge."""
        self.degree_two_contractions.append((node, u, w, weight_uv, weight_vw, edge_id))
    
    def add_degree_one_removal(self, node: str):
        """Record a degree-1 node removal."""
        self.degree_one_removals.append(node)
    
    def get_next_edge_id(self) -> str:
        """Generate unique ID for contracted edges."""
        self._edge_counter += 1
        return f"contracted_edge_{self._edge_counter}"


def degree_one_reduction(G: nx.Graph, terminals: Set[any], 
                        tracker: ReductionTracker = None) -> nx.Graph:
    """
    Iteratively remove degree-1 nodes that are not terminals.
    """
    reduced_graph = G.copy()
    
    changed = True
    while changed:
        changed = False
        nodes_to_remove = []
        
        for node in list(reduced_graph.nodes()):  # Convert to list to avoid iteration issues
            if reduced_graph.degree(node) == 1 and node not in terminals:
                nodes_to_remove.append(node)
        
        if nodes_to_remove:
            changed = True
            for node in nodes_to_remove:
                if tracker:
                    tracker.add_degree_one_removal(node)
                reduced_graph.remove_node(node)
    
    return reduced_graph


def degree_two_reduction(G: nx.Graph, terminals: Set[any], weight: str = "weight",
                        tracker: ReductionTracker = None) -> nx.Graph:
    """
    Replace degree-2 non-terminal nodes with direct edges between their neighbors.
    Only record contractions that actually create or modify edges.
    """
    reduced_graph = G.copy()
    
    # Add metadata to track which edges are contracted vs original
    for u, v in reduced_graph.edges():
        if 'edge_type' not in reduced_graph[u][v]:
            reduced_graph[u][v]['edge_type'] = 'original'
    
    changed = True
    while changed:
        changed = False
        nodes_to_contract = []
        
        # Find all degree-2 non-terminals in current graph state
        for node in list(reduced_graph.nodes()):  # Convert to list to avoid iteration issues
            if (reduced_graph.degree(node) == 2 and 
                node not in terminals and 
                reduced_graph.has_node(node)):  # Double-check node still exists
                
                neighbors = list(reduced_graph.neighbors(node))
                if len(neighbors) == 2:  # Safety check
                    nodes_to_contract.append((node, neighbors[0], neighbors[1]))
        
        # Contract all identified nodes
        for node, u, w in nodes_to_contract:
            # Check if node and neighbors still exist (might have been removed in previous contractions)
            if (reduced_graph.has_node(node) and 
                reduced_graph.has_node(u) and 
                reduced_graph.has_node(w) and
                reduced_graph.degree(node) == 2):  # Re-verify degree
                
                changed = True
                
                # Get weights of the two edges
                weight_uv = reduced_graph[u][node].get(weight, 1)
                weight_vw = reduced_graph[node][w].get(weight, 1)
                new_weight = weight_uv + weight_vw
                
                # Remove the degree-2 node
                reduced_graph.remove_node(node)
                
                # Check if edge (u,w) already exists
                if reduced_graph.has_edge(u, w):
                    # Edge already exists - only record contraction if we're improving the weight
                    existing_weight = reduced_graph[u][w].get(weight, float('inf'))
                    if new_weight < existing_weight:
                        # We're replacing the edge weight - record the contraction
                        edge_id = tracker.get_next_edge_id() if tracker else None
                        reduced_graph[u][w][weight] = new_weight
                        reduced_graph[u][w]['edge_type'] = 'contracted'
                        reduced_graph[u][w]['edge_id'] = edge_id
                        
                        if tracker:
                            tracker.add_degree_two_contraction(node, u, w, weight_uv, weight_vw, edge_id)
                    # If existing weight is better, don't record contraction since we're not using this path
                else:
                    # No existing edge - we're creating a new contracted edge
                    edge_id = tracker.get_next_edge_id() if tracker else None
                    reduced_graph.add_edge(u, w, **{
                        weight: new_weight,
                        'edge_type': 'contracted',
                        'edge_id': edge_id
                    })
                    
                    if tracker:
                        tracker.add_degree_two_contraction(node, u, w, weight_uv, weight_vw, edge_id)
    
    return reduced_graph


def preprocess_graph(G: nx.Graph, terminal_groups: List[List[str]], weight: str = "weight",
                     special_distance: bool = False, long_edge: bool = False,
                     max_settle: int = 2000) -> Tuple[nx.Graph, ReductionTracker]:
    """Apply the structural (and, optionally, heavy) reductions to a fixpoint.

    Always applies the degree-1 and degree-2 reductions.  When ``special_distance``
    or ``long_edge`` is set, the corresponding sound edge-deletion test (see
    :func:`heavy_edge_deletions`) is interleaved with the degree reductions, so
    each deletion can cascade into further structural simplifications and vice
    versa, until nothing changes.

    :param special_distance: enable the Special Distance test (Steiner *tree*
        only; automatically skipped when more than one terminal group is given).
    :param long_edge: enable the long-edge / alternative-path test (valid for
        Steiner tree and forest).
    :param max_settle: work cap for the long-edge bounded Dijkstra.
    :returns: ``(reduced_graph, tracker)``.  Deleted edges need no tracking; only
        the degree reductions are recorded for solution back-mapping.
    """
    # Flatten terminal groups to get all terminals
    all_terminals = set()
    for group in terminal_groups:
        all_terminals.update(group)
    
    reduced_graph = G.copy()
    tracker = ReductionTracker()
    
    # Keep applying reductions until no more changes occur
    max_iterations = len(G.nodes())  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        initial_nodes = reduced_graph.number_of_nodes()
        initial_edges = reduced_graph.number_of_edges()
        
        # Apply degree-1 reduction first
        reduced_graph = degree_one_reduction(reduced_graph, all_terminals, tracker)
        
        # Apply degree-2 reduction
        reduced_graph = degree_two_reduction(reduced_graph, all_terminals, weight, tracker)

        # Optionally apply the heavy edge-deletion tests (Special Distance /
        # long-edge). Every deleted edge is provably in no optimal solution, so
        # the optimum is preserved and the degree reductions above can cascade
        # further on the next iteration. A connectivity guard reverts a pass that
        # would (numerically) disconnect a terminal group — which a sound
        # deletion can never do, so this only protects against edge cases.
        if special_distance or long_edge:
            dels = heavy_edge_deletions(
                reduced_graph, terminal_groups, weight,
                special_distance=special_distance, long_edge=long_edge,
                max_settle=max_settle,
            )
            if dels:
                snapshot = reduced_graph.copy()
                for du, dv in dels:
                    if reduced_graph.has_edge(du, dv):
                        reduced_graph.remove_edge(du, dv)
                if not _groups_connected(reduced_graph, terminal_groups):
                    reduced_graph = snapshot  # defensive: should never trigger

        # Check if any changes occurred
        if (reduced_graph.number_of_nodes() == initial_nodes and 
            reduced_graph.number_of_edges() == initial_edges):
            break
            
        iteration += 1
    
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

    return original_edges


# ---------------------------------------------------------------------------
# Heavy (bound/alternative-based) edge-deletion reductions
# ---------------------------------------------------------------------------
#
# These implement two provably optimum-preserving edge-deletion tests from the
# Steiner-tree reduction literature, complementing the simple degree-1/degree-2
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
# Both tests only ever *delete* edges that are provably contained in no optimal
# solution, so — exactly like the dual-ascent reduced-cost reduction — they need
# no entry in the ReductionTracker: a deleted edge can never appear in a mapped
# solution.  Only the degree reductions they cascade into are tracked.


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
    dist: Dict = {}
    base: Dict = {}
    pq = []
    for t in terminals:
        dist[t] = 0.0
        base[t] = t
        heapq.heappush(pq, (0.0, t))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, attr in G[u].items():
            nd = d + attr.get(weight, 1)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                base[v] = base[u]
                heapq.heappush(pq, (nd, v))
    return dist, base


def _mehlhorn_terminal_mst(G: nx.Graph, terminals, dist: Dict, base: Dict,
                           weight: str) -> Dict:
    """All-pairs terminal *bottleneck* distance via Mehlhorn's construction.

    Mehlhorn (1988) showed that the minimum spanning tree of the auxiliary graph
    built from the *Voronoi-boundary* edges — for each graph edge ``{u, w}`` whose
    endpoints lie in different terminal regions, a candidate terminal edge
    ``{base(u), base(w)}`` of weight ``dist(u) + c(u, w) + dist(w)`` — is a
    minimum spanning tree of the complete terminal distance network.  Because the
    minimax (bottleneck) distance between two nodes is identical across *all*
    spanning trees of a graph, the bottleneck distances read off this tree equal
    those of the exact distance network — at a cost of ``O(m + n log n)`` instead
    of ``O(|T| · (m + n log n))``.

    Returns ``bott[a][b]`` for terminals ``a, b``.
    """
    K = nx.Graph()
    for t in terminals:
        K.add_node(t)
    best: Dict = {}  # (ta, tb) with ta <= tb -> cheapest boundary distance
    for u, w, attr in G.edges(data=True):
        bu, bw = base.get(u), base.get(w)
        if bu is None or bw is None or bu == bw:
            continue
        cand = dist[u] + attr.get(weight, 1) + dist[w]
        key = (bu, bw) if bu <= bw else (bw, bu)
        if cand < best.get(key, float("inf")):
            best[key] = cand
    for (a, b), wq in best.items():
        K.add_edge(a, b, weight=wq)

    mst = nx.minimum_spanning_tree(K, weight="weight")

    bott: Dict = {}
    for root in terminals:
        if not mst.has_node(root):
            continue
        bott[root] = {root: 0.0}
        seen = {root}
        stack = [(root, 0.0)]
        while stack:
            u, mx = stack.pop()
            for v in mst.neighbors(u):
                if v in seen:
                    continue
                seen.add(v)
                nm = max(mx, mst[u][v].get("weight", 1))
                bott[root][v] = nm
                stack.append((v, nm))
    return bott


def special_distance_deletions(G: nx.Graph, terminals: Set, weight: str = "weight",
                               eps: float = 1e-9) -> Set[Tuple]:
    """Edges deletable by the Special Distance (bottleneck Steiner distance) test.

    For an edge ``e = {v, w}`` let ``s(v, w)`` be the bottleneck distance between
    ``v`` and ``w`` in the distance network over ``T ∪ {v, w}`` (Rehfeldt & Koch
    2023, Sec. 2.1).  By Theorem 1, if ``s(v, w) < c(e)`` then **no** minimum
    Steiner tree contains ``e``.

    We use the sound upper bound

        s(v, w) ≤ max( d(v, t_v), b_T(t_v, t_w), d(t_w, w) )

    where ``t_v`` / ``t_w`` are the terminals nearest ``v`` / ``w``, ``d`` is the
    shortest-path distance, and ``b_T`` is the terminal-network bottleneck
    distance.  This is the bottleneck of one concrete path
    ``v → t_v → … → t_w → w`` in the distance network, hence an upper bound on
    ``s``; any value strictly below ``c(e)`` certifies deletion.

    Both ingredients are obtained the *fast* way (Mehlhorn 1988): a single
    multi-source Dijkstra (the terminal Voronoi diagram) yields every nearest
    terminal and its distance, and the Voronoi-boundary MST yields the terminal
    bottleneck distances — ``O(m + n log n)`` overall, versus ``O(|T|)`` separate
    shortest-path computations.

    The terminal-hopping bound is valid for a single terminal group (ordinary
    Steiner tree).  The caller is responsible for only invoking it in that case;
    for the Steiner forest use :func:`long_edge_deletions` instead.
    """
    terminals = set(terminals)
    if not terminals:
        return set()

    dist, base = _voronoi(G, terminals, weight)
    bott = _mehlhorn_terminal_mst(G, terminals, dist, base, weight)

    to_delete: Set[Tuple] = set()
    for u, v, data in G.edges(data=True):
        c = data.get(weight, 1)
        bu, bv = base.get(u), base.get(v)
        if bu is None or bv is None:
            continue
        du = dist.get(u, float("inf"))
        dv = dist.get(v, float("inf"))
        b = bott.get(bu, {}).get(bv, float("inf"))
        if max(du, b, dv) < c - eps:
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
    pq = [(0.0, v0)]
    settled = 0
    while pq and settled < max_settle:
        d, x = heapq.heappop(pq)
        if d > dist.get(x, float("inf")):
            continue
        if d >= cmax:                       # no cheaper detour for any incident edge
            break
        settled += 1
        for y, c in adj.get(x, ()):
            nd = d + c
            if nd < dist.get(y, float("inf")):
                dist[y] = nd
                heapq.heappush(pq, (nd, y))
    out = []
    for w, c in nbrs:
        if dist.get(w, float("inf")) < c - eps:
            out.append((v0, w))
    return out


def long_edge_deletions(G: nx.Graph, weight: str = "weight",
                        max_settle: int = 2000, eps: float = 1e-9,
                        jobs: int = None) -> Set[Tuple]:
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
    njobs = reduce_jobs() if jobs is None else jobs
    results = pmap(_long_edge_for_vertex, nodes, njobs, (adj, max_settle, eps),
                   min_items=_LONG_EDGE_PARALLEL_MIN_NODES)

    to_delete: Set[Tuple] = set()
    for r in results:
        to_delete.update(r)
    return to_delete

def heavy_edge_deletions(G: nx.Graph, terminal_groups: List[List], weight: str = "weight",
                         special_distance: bool = True, long_edge: bool = True,
                         max_settle: int = 2000) -> Set[Tuple]:
    """Combined sound edge-deletion set for one reduction pass.

    Applies the Special Distance test (only when there is a single terminal
    group, i.e. an ordinary Steiner tree) and/or the long-edge test (valid for
    tree and forest), returning the union of edges that are provably in no
    optimal solution.
    """
    all_terms = {t for g in terminal_groups for t in g}
    dels: Set[Tuple] = set()
    if special_distance and len(terminal_groups) == 1:
        dels |= special_distance_deletions(G, all_terms, weight)
    if long_edge:
        dels |= long_edge_deletions(G, weight, max_settle=max_settle)
    return dels


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