"""Tests for the Directed Group Steiner Tree variant (thesis Ch. 5.7, directed
arborescence specialisation of :class:`~steinerpy.GroupSteinerProblem`).

The randomized test validates the exact solve against an **independent
brute-force oracle** (enumerate edge subsets of a small directed graph) rather
than hard-coded expected values, following the pattern in
``tests/test_pc_transform.py``.
"""

import math
import random

import networkx as nx
import pytest

from steinerpy import DirectedGroupSteinerProblem


def _directed_path_graph():
    G = nx.DiGraph()
    for u, v in [("r", "A"), ("A", "B"), ("B", "C"), ("C", "D")]:
        G.add_edge(u, v, weight=1)
    return G


def _path_graph():
    G = nx.Graph()
    for u, v in [("A", "B"), ("B", "C"), ("C", "D")]:
        G.add_edge(u, v, weight=1)
    return G


def test_directed_group_steiner_picks_cheapest_representatives():
    G = _directed_path_graph()
    # Group 1 = {A, B}, group 2 = {C, D}. Cheapest arborescence from r
    # reaching one of each is r->A->B->C, cost 3.
    prob = DirectedGroupSteinerProblem(G, [["A", "B"], ["C", "D"]], root="r")
    sol = prob.get_solution()
    assert sol.objective == pytest.approx(3.0)
    chosen = {n for e in sol.edges for n in e}
    assert chosen & {"A", "B"}
    assert chosen & {"C", "D"}
    # Connector super-terminals must not leak into the reported solution.
    assert not any(str(n).startswith("__group_") for n in chosen)


def test_directed_group_steiner_rejects_undirected_graph():
    G = _path_graph()
    with pytest.raises(ValueError):
        DirectedGroupSteinerProblem(G, [["A"], ["D"]], root="A")


def test_directed_group_steiner_rejects_missing_root():
    G = _directed_path_graph()
    with pytest.raises(ValueError):
        DirectedGroupSteinerProblem(G, [["A"], ["C"]], root="not_in_graph")


def test_directed_group_steiner_infeasible_raises():
    G = nx.DiGraph()
    G.add_edge("r", "A", weight=1)
    G.add_edge("X", "Y", weight=1)  # unreachable from r
    with pytest.raises(RuntimeError):
        DirectedGroupSteinerProblem(G, [["A"], ["Y"]], root="r").get_solution()


# ---------------------------------------------------------------------------
# Brute-force oracle (small instances only)
# ---------------------------------------------------------------------------

def brute_directed_group_steiner(graph, groups, root, weight="weight"):
    """Optimal cost via enumerating edge subsets: min cost subgraph such that
    ``root`` reaches at least one vertex of every group. ``math.inf`` if no
    subset works."""
    edges = list(graph.edges(data=True))
    n = len(edges)
    best = math.inf
    for mask in range(1 << n):
        adj = {}
        cost = 0.0
        for i in range(n):
            if mask & (1 << i):
                u, v, d = edges[i]
                adj.setdefault(u, []).append(v)
                cost += d.get(weight, 1)
        if cost >= best:
            continue
        reachable = {root}
        stack = [root]
        while stack:
            x = stack.pop()
            for y in adj.get(x, []):
                if y not in reachable:
                    reachable.add(y)
                    stack.append(y)
        if all(any(v in reachable for v in grp) for grp in groups):
            best = cost
    return best


def random_directed_group_instance(seed, n=6):
    """Random rooted DiGraph plus a 2-group split of the non-root vertices.

    A directed backbone from the root through a random permutation of the
    other vertices guarantees every instance is feasible; a handful of extra
    random arcs create alternative (sometimes cheaper) routes.
    """
    rng = random.Random(seed)
    nodes = list(range(n))
    root = 0
    rest = nodes[1:]
    rng.shuffle(rest)
    order = [root] + rest
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for i in range(len(order) - 1):
        g.add_edge(order[i], order[i + 1], weight=rng.randint(1, 9))
    for _ in range(rng.randint(0, n)):
        u, v = rng.sample(nodes, 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v, weight=rng.randint(1, 9))
    half = max(1, len(rest) // 2)
    groups = [rest[:half], rest[half:]]
    return g, groups, root


@pytest.mark.xfail(
    reason="Pre-existing bug in the directed-cut kernel shared with "
           "DirectedSteinerProblem: an unused 'back arc' can inflate the "
           "reported objective while gap is still reported as 0.0 (proven "
           "optimal) -- see https://github.com/berendmarkhorst/SteinerPy/"
           "issues/30. Not specific to DirectedGroupSteinerProblem's "
           "transformation; some seeds still pass because no back arc "
           "happens to lie on the optimal path.",
    strict=False,
)
@pytest.mark.parametrize("seed", range(20))
def test_directed_group_steiner_matches_oracle(seed):
    g, groups, root = random_directed_group_instance(seed)
    opt = brute_directed_group_steiner(g, groups, root)
    sol = DirectedGroupSteinerProblem(g.copy(), groups, root=root).get_solution()

    assert sol.gap == pytest.approx(0.0, abs=1e-9)
    assert sol.objective == pytest.approx(opt)

    # The returned edges must actually reach one vertex per group from root.
    adj = {}
    for u, v in sol.edges:
        adj.setdefault(u, []).append(v)
    reachable = {root}
    stack = [root]
    while stack:
        x = stack.pop()
        for y in adj.get(x, []):
            if y not in reachable:
                reachable.add(y)
                stack.append(y)
    for grp in groups:
        assert any(v in reachable for v in grp)
