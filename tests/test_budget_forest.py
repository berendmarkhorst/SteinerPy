"""Budgeted forest requirements can overlap and share purchased edges."""

import itertools
import random

import networkx as nx
import pytest

from steinerpy import Problem


def exhaustive_connected_count(graph, groups, budget, max_degree=None):
    best = 0
    for flags in itertools.product((False, True), repeat=graph.number_of_edges()):
        edges = [edge for edge, keep in zip(graph.edges, flags) if keep]
        if sum(graph.edges[edge]["cost"] for edge in edges) > budget:
            continue
        selected = type(graph)()
        selected.add_nodes_from(graph)
        selected.add_edges_from(edges)
        if max_degree is not None and any(
            degree > max_degree for _, degree in selected.degree()
        ):
            continue
        best = max(
            best,
            sum(
                nx.has_path(selected, group[0], node)
                for group in groups
                for node in group
            ),
        )
    return best


def check_solution(graph, groups, budget, max_degree=None):
    problem = Problem(
        graph,
        terminals=groups,
        budget=budget,
        max_degree=max_degree,
        weight="cost",
        preprocess=False,
        **({"root": groups[0][0]} if graph.is_directed() else {}),
    )
    solution = problem.get_solution()
    assert solution.gap == 0
    assert solution.objective == exhaustive_connected_count(
        graph, groups, budget, max_degree
    )
    selected = type(graph)()
    selected.add_nodes_from(graph)
    selected.add_edges_from(solution.edges)
    assert sum(graph.edges[edge]["cost"] for edge in solution.edges) <= budget
    if max_degree is not None:
        assert all(degree <= max_degree for _, degree in selected.degree())
    assert solution.objective == sum(
        nx.has_path(selected, group[0], node) for group in groups for node in group
    )


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "groups",
    [
        [[0, 1], [0, 2]],
        [[0, 2], [1, 2]],
        [[0, 1], [2, 3]],
        [[0, 3], [3, 0]],
        [[0], [0, 2]],
        [[0, 1], [2, 1]],
        [[0, 1], [0, 1]],
    ],
)
def test_budgeted_path_forest_matches_edge_subset_oracle(groups, budget):
    graph = nx.path_graph(4)
    nx.set_edge_attributes(graph, 1, "cost")
    check_solution(graph, groups, budget)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("max_degree", [None, 1, 2])
def test_budgeted_shared_forest_matches_weighted_oracle(seed, max_degree):
    rng = random.Random(seed)
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2)
    nx.set_edge_attributes(
        graph, {edge: rng.randint(0, 3) for edge in graph.edges}, "cost"
    )
    groups = [[0, 1, 2], [3, 2], [1, 3]]
    check_solution(graph, groups, budget=3, max_degree=max_degree)


def test_budgeted_forest_with_isolated_shared_root():
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    graph.add_edge(1, 2, cost=1)
    check_solution(graph, [[0, 1], [0, 2]], budget=1)


@pytest.mark.parametrize("budget", [0, 1, 4, 8])
def test_directed_budget_still_requires_forward_arcs(budget):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (1, 2, 3), (2, 0, 0), (3, 1, 0)], weight="cost"
    )
    check_solution(graph, [[0, 2, 3]], budget)
