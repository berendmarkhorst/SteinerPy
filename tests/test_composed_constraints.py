"""Independent original-edge subset checks for split-graph constraints."""

import itertools
import math
import random

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy import _composition as composition
from .test_directed_group_steiner import solver


def candidates(graph):
    for node in graph:
        yield [], [node]
    for flags in itertools.product((False, True), repeat=graph.number_of_edges()):
        edges = [e for e, flag in zip(graph.edges, flags) if flag]
        if edges:
            selected = nx.Graph(edges)
            if nx.is_connected(selected):
                yield edges, list(selected)


def degree_optimum(graph, groups, costs, degree, hops, terminals=()):
    best = math.inf
    for edges, nodes in candidates(graph):
        if not set(terminals) <= set(nodes) or any(
            set(nodes).isdisjoint(g) for g in groups
        ):
            continue
        if hops is not None and len(edges) > hops:
            continue
        selected = nx.Graph()
        selected.add_nodes_from(nodes)
        selected.add_edges_from(edges)
        if any(d > degree for _, d in selected.degree()):
            continue
        best = min(
            best,
            sum(graph.edges[e].get("weight", 1) for e in edges)
            + sum(costs.get(n, 0) for n in nodes),
        )
    return best


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "degree,hops", [(0, None), (1, None), (2, None), (2, 2), (3, 3)]
)
def test_composed_degrees_match_original_subset_oracle(seed, degree, hops, solver):
    rng = random.Random(seed)
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2)
    nx.set_edge_attributes(graph, {e: rng.randint(0, 4) for e in graph.edges}, "weight")
    costs = {n: rng.randint(0, 4) for n in graph}
    groups = [[0, 1], [2, 3], [1, 3]]
    terminals = [0] if seed % 2 else []
    options = {"terminals": terminals} if terminals else {}
    problem = Problem(
        graph,
        groups=groups,
        node_costs=costs,
        max_degree=degree,
        hop_limit=hops,
        **options
    )
    expected = degree_optimum(graph, groups, costs, degree, hops, terminals)
    if math.isinf(expected):
        with pytest.raises(RuntimeError, match="infeasible"):
            problem.get_solution(solver=solver)
        return
    sol = problem.get_solution(solver=solver)
    assert sol.objective == expected and sol.gap == 0
    selected = nx.Graph()
    selected.add_nodes_from(sol.selected_nodes)
    selected.add_edges_from(sol.edges)
    assert nx.is_connected(selected)
    assert all(d <= degree for _, d in selected.degree())
    assert hops is None or len(sol.edges) <= hops
    assert set(terminals) <= set(selected)
    assert all(not set(selected).isdisjoint(g) for g in groups)
    assert sol.objective == sum(graph.edges[e]["weight"] for e in sol.edges) + sum(
        costs[n] for n in selected
    )


def test_split_ports_do_not_hide_original_branch_degree(solver):
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 0), (0, 2, 0), (0, 3, 0), (1, 2, 5), (2, 3, 5)]
    )
    sol = Problem(
        graph, groups=[[0], [1], [2], [3]], node_costs={0: 2}, max_degree=2
    ).get_solution(solver=solver)
    assert sol.objective == 7 and sol.gap == 0
    assert max(dict(nx.Graph(sol.edges).degree()).values()) == 2


def test_decoded_degree_limit_is_independently_checked():
    p = Problem(nx.path_graph(3), groups=[[0], [2]], node_costs={}, max_degree=1)
    spec = p.implementation.spec
    compiled = composition.compile_groups(spec)
    arcs = set()
    for terminal in compiled.terminals:
        path = nx.shortest_path(compiled.graph, compiled.root, terminal)
        arcs.update(zip(path, path[1:]))
    with pytest.raises(RuntimeError, match="original-node degree"):
        composition.decode_groups(spec, compiled, arcs, 2)
