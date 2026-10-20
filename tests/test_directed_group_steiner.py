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

from steinerpy import DirectedGroupSteinerProblem, Problem


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


def brute_directed_group_steiner(graph, groups, root, weight="weight", hop_limit=None):
    """Optimal cost via enumerating edge subsets: min cost subgraph such that
    ``root`` reaches at least one vertex of every group. ``math.inf`` if no
    subset works."""
    edges = list(graph.edges(data=True))
    n = len(edges)
    best = math.inf
    for mask in range(1 << n):
        if hop_limit is not None and bin(mask).count("1") > hop_limit:
            continue
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


@pytest.fixture(scope="module", params=["highs", "gurobi"])
def solver(request):
    if request.param == "gurobi":
        gp = pytest.importorskip("gurobipy")
        try:
            with gp.Env(params={"OutputFlag": 0}) as env:
                with gp.Model(env=env):
                    pass
        except gp.GurobiError:
            pytest.skip("Gurobi license is not available")
    return request.param


@pytest.mark.parametrize("api", ["legacy", "inferred", "explicit"])
@pytest.mark.parametrize("limit, expected", [(1, 9), (2, 0)])
def test_group_hops_count_original_zero_cost_arcs(api, limit, expected, solver):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [("r", "__group_0__", 0), ("__group_0__", "t", 0), ("r", "t", 9)]
    )
    original = graph.copy()
    options = dict(groups=[["t"], ["r", "t"], ["t"]], root="r", hop_limit=limit)
    if api == "legacy":
        problem = DirectedGroupSteinerProblem(graph, **options)
    else:
        problem = Problem(
            graph, variant="directed_group" if api == "explicit" else None, **options
        )
    solution = problem.get_solution(solver=solver)
    assert solution.objective == expected and solution.gap == 0
    assert len(solution.edges) == limit
    assert all(graph.has_edge(*arc) for arc in solution.edges)
    assert nx.utils.graphs_equal(graph, original)


@pytest.mark.parametrize("covers_root", [True, False])
def test_zero_group_hops(covers_root, solver):
    graph = nx.DiGraph()
    graph.add_edge("r", "t", weight=0)
    groups = [["r", "t"], ["r"]] if covers_root else [["t"]]
    problem = Problem(graph, groups=groups, root="r", hop_limit=0)
    if covers_root:
        solution = problem.get_solution(solver=solver)
        assert solution.objective == 0 and solution.edges == []
    else:
        with pytest.raises(RuntimeError, match="infeasible"):
            problem.get_solution(solver=solver)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("limit", [0, 2, 4])
def test_group_hops_match_original_graph_oracle(seed, limit, solver):
    graph, groups, root = random_directed_group_instance(seed, n=5)
    groups += [[root, groups[0][0]], [groups[0][0], groups[1][0]]]
    expected = brute_directed_group_steiner(graph, groups, root, hop_limit=limit)
    problem = Problem(graph, groups=groups, root=root, hop_limit=limit)
    if math.isinf(expected):
        with pytest.raises(RuntimeError, match="infeasible"):
            problem.get_solution(solver=solver)
        return
    solution = problem.get_solution(solver=solver)
    assert solution.objective == pytest.approx(expected) and solution.gap == 0
    assert len(solution.edges) <= limit
    selected = nx.DiGraph()
    selected.add_node(root)
    selected.add_edges_from(solution.edges)
    reachable = {root} | nx.descendants(selected, root)
    assert all(reachable.intersection(group) for group in groups)
    assert all(graph.has_edge(*arc) for arc in solution.edges)
    assert sum(graph[u][v]["weight"] for u, v in solution.edges) == expected


def test_decoded_group_hop_limit_is_checked(monkeypatch):
    from steinerpy.objects import SteinerProblem, Solution

    graph = nx.path_graph(3, create_using=nx.DiGraph)
    problem = DirectedGroupSteinerProblem(graph, [[2]], root=0, hop_limit=1)
    monkeypatch.setattr(
        SteinerProblem,
        "get_solution",
        lambda *a, **k: Solution(
            objective=2,
            gap=0,
            runtime=0,
            selected_edges=[(0, 1), (1, 2)],
            original_selected_edges=[(0, 1), (1, 2)],
        ),
    )
    with pytest.raises(RuntimeError, match="original-arc hop limit"):
        problem.get_solution()


@pytest.mark.parametrize("limit", [-1, 1.5, True, "2", math.inf, math.nan])
def test_legacy_group_rejects_invalid_hop_limit(limit):
    with pytest.raises(ValueError, match="non-negative integer"):
        DirectedGroupSteinerProblem(
            nx.path_graph(2, create_using=nx.DiGraph), [[1]], root=0, hop_limit=limit
        )


def test_group_hops_with_no_original_arcs(solver):
    graph = nx.DiGraph()
    graph.add_node("r")
    solution = Problem(
        graph, groups=[["r"], ["r"]], root="r", hop_limit=0
    ).get_solution(solver=solver)
    assert solution.objective == 0 and solution.edges == []
