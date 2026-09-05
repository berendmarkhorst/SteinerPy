"""Correctness tests for the opt-in Steiner primal-heuristic portfolio."""

import random

import networkx as nx
import pytest

from steinerpy import SteinerProblem
from steinerpy.primal_heuristics import (
    connects_terminals,
    edges_cost,
    implied_profit_candidates,
    implied_profit_shortest_path,
    improve_steiner_tree,
)


def _random_labeled_tree(order, rng):
    """Build a labelled tree without depending on NetworkX's generator API."""
    sequence = [rng.randrange(order) for _ in range(order - 2)]
    degree = [1] * order
    for node in sequence:
        degree[node] += 1

    graph = nx.Graph()
    graph.add_nodes_from(range(order))
    for node in sequence:
        leaf = next(index for index, value in enumerate(degree) if value == 1)
        graph.add_edge(leaf, node)
        degree[leaf] -= 1
        degree[node] -= 1

    remaining = [node for node, value in enumerate(degree) if value == 1]
    graph.add_edge(*remaining)
    return graph


def test_vertex_elimination_strictly_improves_and_stays_feasible():
    graph = nx.Graph()
    graph.add_edge("a", "x", weight=3)
    graph.add_edge("x", "c", weight=3)
    graph.add_edge("a", "b", weight=4)
    graph.add_edge("b", "c", weight=4)
    incumbent = [("a", "x"), ("x", "c"), ("a", "b")]

    result = improve_steiner_tree(
        graph,
        incumbent,
        ["a", "b", "c"],
        key_path_exchange=False,
    )

    assert result.vertex_eliminations == 1
    assert result.objective_before == 10
    assert result.objective_after == 8
    assert connects_terminals(result.edges, ["a", "b", "c"])


def test_key_path_exchange_strictly_improves_and_stays_feasible():
    graph = nx.Graph()
    graph.add_edge("a", "x", weight=5)
    graph.add_edge("x", "b", weight=5)
    graph.add_edge("a", "y", weight=2)
    graph.add_edge("y", "b", weight=2)

    result = improve_steiner_tree(
        graph,
        [("a", "x"), ("x", "b")],
        ["a", "b"],
        vertex_elimination=False,
    )

    assert result.key_path_exchanges == 1
    assert result.objective_before == 10
    assert result.objective_after == 4
    assert connects_terminals(result.edges, ["a", "b"])


def test_local_search_rejects_infeasible_input():
    graph = nx.path_graph(4)
    nx.set_edge_attributes(graph, 1, "weight")
    with pytest.raises(ValueError, match="terminal-feasible"):
        improve_steiner_tree(graph, [(0, 1)], [0, 3])


def test_implied_profit_known_instance_is_feasible():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            ("a", "v", 2),
            ("v", "b", 1),
            ("v", "c", 1),
            ("a", "b", 5),
            ("a", "c", 5),
        ]
    )
    edges = implied_profit_shortest_path(graph, ["a", "b", "c"])
    assert connects_terminals(edges, ["a", "b", "c"])
    assert edges_cost(graph, edges, "weight") == 4


def test_opt_in_portfolio_never_worsens_public_heuristic_solution(monkeypatch):
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "0")
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            ("a", "x", 5),
            ("x", "b", 5),
            ("a", "y", 2),
            ("y", "b", 2),
            ("x", "c", 2),
            ("y", "c", 2),
        ]
    )
    baseline = SteinerProblem(
        graph.copy(), [["a", "b", "c"]], preprocess=False
    ).get_solution(exact=False)
    problem = SteinerProblem(
        graph.copy(),
        [["a", "b", "c"]],
        preprocess=False,
        primal_local_search=True,
        implied_profit=True,
    )
    stronger = problem.get_solution(exact=False)

    assert stronger.objective <= baseline.objective + 1e-9
    assert connects_terminals(stronger.selected_edges, ["a", "b", "c"])
    stats = problem.heuristic_stats
    assert stats["objective_after"] <= stats["objective_before"]


@pytest.mark.parametrize("seed", range(30))
def test_random_candidates_are_feasible_and_local_search_is_monotone(seed):
    rng = random.Random(seed)
    graph = _random_labeled_tree(9, rng)
    for u, v in graph.edges():
        graph[u][v]["weight"] = rng.randint(1, 9)
    for _ in range(10):
        u, v = rng.sample(list(graph), 2)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v, weight=rng.randint(1, 9))
    terminals = rng.sample(list(graph), 4)

    from networkx.algorithms.approximation import steiner_tree

    initial = list(
        steiner_tree(graph, terminals, weight="weight", method="kou").edges()
    )
    result = improve_steiner_tree(graph, initial, terminals)
    assert connects_terminals(result.edges, terminals)
    assert result.objective_after <= result.objective_before + 1e-9

    candidates = implied_profit_candidates(graph, terminals)
    assert candidates
    for candidate in candidates:
        assert connects_terminals(candidate, terminals)
        assert edges_cost(graph, candidate, "weight") >= 0
