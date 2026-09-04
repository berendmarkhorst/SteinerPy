"""Opt-in primal heuristics and local improvement for undirected Steiner trees.

The local-search neighborhoods follow Uchoa & Werneck, *Fast Local Search for
Steiner Trees in Graphs* (JEA 17, 2012):

* vertex elimination removes one non-terminal from the current vertex set and
  recomputes the best spanning tree induced by the remaining vertices;
* key-path exchange removes a maximal path whose internal vertices are
  degree-two Steiner vertices and reconnects the two resulting components by a
  shorter path.

The implied-profit shortest-path heuristic implements Rehfeldt & Koch,
*Implications, Conflicts, and Reductions for Steiner Trees* (Mathematical
Programming 197, 2023), Section 5.1.1, equations (28)--(29).  As suggested in
the paper, the inexpensive minimum alternative incident-edge cost is used in
place of the full bottleneck value in equation (28).

All routines are deliberately isolated and opt-in.  They validate feasibility
and accept a local-search move only when its objective strictly decreases.
They do not provide a lower bound and therefore never change exactness or gap
semantics on their own.
"""

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx


Edge = Tuple[Hashable, Hashable]
EPS = 1e-9


@dataclass
class PrimalImprovementResult:
    """Result and audit counters for :func:`improve_steiner_tree`."""

    edges: List[Edge]
    objective_before: float
    objective_after: float
    vertex_eliminations: int = 0
    key_path_exchanges: int = 0
    rounds: int = 0


def edges_cost(graph: nx.Graph, edges: Iterable[Edge], weight: str) -> float:
    """Return the cost of undirected ``edges`` in ``graph``."""
    total = 0.0
    for u, v in edges:
        data = graph.get_edge_data(u, v)
        if data is None:
            data = graph.get_edge_data(v, u)
        if data is None:
            raise ValueError("primal edge {!r} is not in the graph".format((u, v)))
        total += float(data.get(weight, 1))
    return total


def connects_terminals(edges: Iterable[Edge], terminals: Sequence) -> bool:
    """Whether ``edges`` connect all distinct ``terminals``."""
    terms = list(dict.fromkeys(terminals))
    if len(terms) <= 1:
        return True
    tree = nx.Graph()
    tree.add_edges_from(edges)
    return all(t in tree for t in terms) and all(
        nx.has_path(tree, terms[0], t) for t in terms[1:]
    )


def _prune_nonterminal_leaves(tree: nx.Graph, terminals: Set) -> None:
    queue = [v for v in tree if v not in terminals and tree.degree(v) <= 1]
    while queue:
        v = queue.pop()
        if v not in tree or v in terminals or tree.degree(v) > 1:
            continue
        neighbors = list(tree.neighbors(v))
        tree.remove_node(v)
        queue.extend(neighbors)


def _mst_on_vertices(
    graph: nx.Graph, vertices: Set, terminals: Sequence, weight: str
) -> Optional[List[Edge]]:
    """Cheapest induced-vertex MST candidate, pruned to the terminal subtree."""
    terms = list(dict.fromkeys(terminals))
    if not terms:
        return []
    if any(t not in vertices or t not in graph for t in terms):
        return None
    induced = graph.subgraph(vertices)
    component = nx.node_connected_component(induced, terms[0])
    if any(t not in component for t in terms):
        return None
    tree = nx.minimum_spanning_tree(induced.subgraph(component), weight=weight)
    _prune_nonterminal_leaves(tree, set(terms))
    candidate = list(tree.edges())
    return candidate if connects_terminals(candidate, terms) else None


def _solution_vertices(edges: Iterable[Edge], terminals: Sequence) -> Set:
    vertices = set(terminals)
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)
    return vertices


def _vertex_elimination_move(
    graph: nx.Graph, edges: List[Edge], terminals: Sequence, weight: str
) -> Optional[List[Edge]]:
    """Return the best strictly improving single-vertex elimination, if any."""
    current_cost = edges_cost(graph, edges, weight)
    vertices = _solution_vertices(edges, terminals)
    terminal_set = set(terminals)
    best = None
    best_cost = current_cost
    for v in sorted(vertices - terminal_set, key=lambda node: str(node)):
        candidate = _mst_on_vertices(graph, vertices - {v}, terminals, weight)
        if candidate is None:
            continue
        cost = edges_cost(graph, candidate, weight)
        if cost < best_cost - EPS:
            best, best_cost = candidate, cost
    return best


def _key_paths(edges: List[Edge], terminals: Sequence) -> List[List]:
    """Enumerate maximal key paths of the current tree deterministically."""
    tree = nx.Graph()
    tree.add_edges_from(edges)
    terminal_set = set(terminals)
    key_vertices = terminal_set | {
        v for v in tree if v not in terminal_set and tree.degree(v) >= 3
    }
    seen = set()
    paths: List[List] = []
    for start in sorted(key_vertices, key=lambda node: str(node)):
        if start not in tree:
            continue
        for neighbor in sorted(tree.neighbors(start), key=lambda node: str(node)):
            first = frozenset((start, neighbor))
            if first in seen:
                continue
            path = [start, neighbor]
            seen.add(first)
            previous, current = start, neighbor
            while current not in key_vertices:
                nxts = [v for v in tree.neighbors(current) if v != previous]
                if len(nxts) != 1:
                    break
                nxt = nxts[0]
                seen.add(frozenset((current, nxt)))
                path.append(nxt)
                previous, current = current, nxt
            if path[-1] in key_vertices and path[-1] != start:
                paths.append(path)
    return paths


def _key_path_exchange_move(
    graph: nx.Graph, edges: List[Edge], terminals: Sequence, weight: str
) -> Optional[List[Edge]]:
    """Return the best strictly improving key-path exchange, if any."""
    current = nx.Graph()
    current.add_edges_from(edges)
    current_cost = edges_cost(graph, edges, weight)
    best = None
    best_cost = current_cost

    for key_path in _key_paths(edges, terminals):
        path_edges = list(zip(key_path, key_path[1:]))
        blocked = {frozenset(e) for e in path_edges}
        remainder = current.copy()
        remainder.remove_edges_from(path_edges)
        left = nx.node_connected_component(remainder, key_path[0])
        right = nx.node_connected_component(remainder, key_path[-1])
        if left & right:
            continue

        def edge_ok(u, v):
            return frozenset((u, v)) not in blocked

        search_graph = nx.subgraph_view(graph, filter_edge=edge_ok)
        try:
            distances, paths = nx.multi_source_dijkstra(
                search_graph, list(left), weight=weight
            )
        except (nx.NetworkXError, nx.NetworkXNoPath):
            continue
        targets = [v for v in right if v in distances]
        if not targets:
            continue
        target = min(targets, key=lambda v: (distances[v], str(v)))
        old_cost = edges_cost(graph, path_edges, weight)
        if distances[target] >= old_cost - EPS:
            continue

        replacement = list(zip(paths[target], paths[target][1:]))
        union_edges = [e for e in edges if frozenset(e) not in blocked]
        union_edges.extend(replacement)
        vertices = _solution_vertices(union_edges, terminals)
        candidate = _mst_on_vertices(graph, vertices, terminals, weight)
        if candidate is None:
            continue
        cost = edges_cost(graph, candidate, weight)
        if cost < best_cost - EPS:
            best, best_cost = candidate, cost
    return best


def improve_steiner_tree(
    graph: nx.Graph,
    edges: Iterable[Edge],
    terminals: Sequence,
    weight: str = "weight",
    vertex_elimination: bool = True,
    key_path_exchange: bool = True,
    max_rounds: int = 20,
) -> PrimalImprovementResult:
    """Apply cost-monotone local search to one feasible undirected tree.

    The returned candidate always connects ``terminals`` and never costs more
    than the input.  A failed or non-improving neighborhood leaves the incumbent
    unchanged.  Multi-group forests and directed arborescences intentionally
    stay outside this first experimental implementation.
    """
    if isinstance(graph, nx.DiGraph):
        raise ValueError("primal local search currently requires an undirected graph")
    terms = list(dict.fromkeys(terminals))
    current = list(edges)
    if not connects_terminals(current, terms):
        raise ValueError("primal local search requires a terminal-feasible tree")
    before = edges_cost(graph, current, weight)
    # Normalize the incumbent once; induced-vertex MST + pruning cannot increase
    # its cost because the incumbent itself is available in the induced graph.
    normalized = _mst_on_vertices(
        graph, _solution_vertices(current, terms), terms, weight
    )
    if normalized is not None and edges_cost(graph, normalized, weight) <= before + EPS:
        current = normalized

    n_vertex = 0
    n_key_path = 0
    rounds = 0
    for _ in range(max(0, max_rounds)):
        changed = False
        if vertex_elimination:
            candidate = _vertex_elimination_move(graph, current, terms, weight)
            if candidate is not None:
                current = candidate
                n_vertex += 1
                changed = True
        if key_path_exchange:
            candidate = _key_path_exchange_move(graph, current, terms, weight)
            if candidate is not None:
                current = candidate
                n_key_path += 1
                changed = True
        rounds += 1
        if not changed:
            break

    after = edges_cost(graph, current, weight)
    if after > before + EPS or not connects_terminals(current, terms):
        raise AssertionError("local search violated its cost/feasibility invariant")
    return PrimalImprovementResult(
        edges=current,
        objective_before=before,
        objective_after=after,
        vertex_eliminations=n_vertex,
        key_path_exchanges=n_key_path,
        rounds=rounds,
    )


def _implied_profits(
    graph: nx.Graph, terminals: Set, connected: Set, weight: str
) -> Dict:
    """Equation (28), using the paper's inexpensive alternative-edge bound."""
    profit: Dict = {}
    unconnected = terminals - connected
    for v in graph:
        if v in terminals:
            profit[v] = 0.0
            continue
        best = 0.0
        for w in graph.neighbors(v):
            if w not in unconnected:
                continue
            edge_cost = float(graph[v][w].get(weight, 1))
            alternatives = [
                float(data.get(weight, 1)) for x, data in graph[w].items() if x != v
            ]
            bottleneck = min(alternatives) if alternatives else math.inf
            best = max(best, bottleneck - edge_cost)
        profit[v] = max(0.0, best)
    return profit


def implied_profit_shortest_path(
    graph: nx.Graph,
    terminals: Sequence,
    weight: str = "weight",
    start=None,
) -> List[Edge]:
    """Build one implied-profit-biased shortest-path Steiner tree.

    This is the constructive rule of Rehfeldt & Koch (2023), Section 5.1.1.
    It is a primal heuristic, not an approximation certificate; callers should
    compare it with other feasible candidates and retain a separately proven
    lower bound for any reported ``Solution.gap``.
    """
    if isinstance(graph, nx.DiGraph):
        raise ValueError(
            "implied-profit heuristic currently requires an undirected graph"
        )
    terms = list(dict.fromkeys(terminals))
    if len(terms) <= 1:
        return []
    if any(t not in graph for t in terms):
        raise nx.NodeNotFound("all terminals must be present in the graph")
    if any(data.get(weight, 1) < 0 for _, _, data in graph.edges(data=True)):
        raise ValueError("implied-profit heuristic requires non-negative edge costs")
    if start is None:
        start = terms[0]
    if start not in graph:
        raise nx.NodeNotFound("start vertex is not in the graph")

    terminal_set = set(terms)
    tree_nodes = {start}
    tree_edges: Dict[frozenset, Edge] = {}
    connected = terminal_set & tree_nodes
    distance = {v: math.inf for v in graph}
    predecessor = {start: start}
    distance[start] = 0.0
    counter = itertools.count()
    queue = [(0.0, next(counter), start)]
    profit = _implied_profits(graph, terminal_set, connected, weight)

    while queue and connected != terminal_set:
        dist_v, _, v = heapq.heappop(queue)
        if dist_v > distance[v] + EPS:
            continue
        if v in terminal_set and v not in connected:
            path = [v]
            seen = {v}
            cursor = v
            while cursor not in tree_nodes:
                cursor = predecessor.get(cursor)
                if cursor is None or cursor in seen:
                    raise RuntimeError("failed to reconstruct implied-profit path")
                seen.add(cursor)
                path.append(cursor)
            for a, b in zip(path, path[1:]):
                tree_edges.setdefault(frozenset((a, b)), (a, b))
            tree_nodes.update(path)
            connected = terminal_set & tree_nodes
            for u in path:
                distance[u] = 0.0
                predecessor[u] = u
                heapq.heappush(queue, (0.0, next(counter), u))
            profit = _implied_profits(graph, terminal_set, connected, weight)
            continue

        for w, data in graph[v].items():
            cost = float(data.get(weight, 1))
            candidate = dist_v + cost - min(cost, profit.get(v, 0.0), dist_v)
            if candidate < distance[w] - EPS:
                distance[w] = candidate
                predecessor[w] = v
                heapq.heappush(queue, (candidate, next(counter), w))

    if connected != terminal_set:
        raise nx.NetworkXNoPath("the terminals are disconnected")
    candidate = _mst_on_vertices(graph, tree_nodes, terms, weight)
    if candidate is None or not connects_terminals(candidate, terms):
        raise RuntimeError("implied-profit heuristic returned an infeasible tree")
    return candidate


def implied_profit_candidates(
    graph: nx.Graph,
    terminals: Sequence,
    weight: str = "weight",
    max_starts: int = 8,
) -> List[List[Edge]]:
    """Return deterministic multi-start implied-profit candidates."""
    starts = sorted(dict.fromkeys(terminals), key=lambda node: str(node))[
        : max(1, max_starts)
    ]
    candidates = []
    for start in starts:
        try:
            candidates.append(
                implied_profit_shortest_path(graph, terminals, weight, start=start)
            )
        except (nx.NetworkXError, RuntimeError, ValueError):
            continue
    return candidates
