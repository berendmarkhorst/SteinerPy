"""Automatic formulation choice preserves eligibility and ignores irrelevant leaves."""

from copy import deepcopy

import networkx as nx
import pytest

from steinerpy import Problem, SteinerProblem
from steinerpy import _uniform_node as uniform


@pytest.fixture(autouse=True)
def automatic_selection(monkeypatch):
    monkeypatch.delenv("STEINERPY_UNIFORM_NODE_MODEL", raising=False)
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "0")
    monkeypatch.setenv("STEINERPY_DECOMPOSE", "0")


def _problem(graph, terminals):
    return SteinerProblem(graph, [terminals], preprocess=False)


def _snapshot(graph):
    return deepcopy(
        (
            graph.graph,
            dict(graph.nodes(data=True)),
            {frozenset(edge): data for *edge, data in graph.edges(data=True)},
        )
    )


@pytest.mark.parametrize("degree,expected", [(4, False), (5, True)])
def test_incidence_degree_boundary(degree, expected):
    graph = nx.Graph()
    for node in range(8):
        graph.add_edges_from((node, 8 + (node + step) % 8) for step in range(degree))
    assert uniform.prefer_node_model(_problem(graph, list(range(8)))) is expected


def test_boundary_uses_average_connector_degree():
    graph = nx.Graph()
    graph.add_edges_from((node, "a") for node in range(4))
    graph.add_edges_from((node, "b") for node in range(6))
    terminals = list(range(6))
    assert uniform.prefer_node_model(_problem(graph, terminals))
    graph.remove_edge(0, "b")
    assert nx.is_connected(graph)
    assert not uniform.prefer_node_model(_problem(graph, terminals))


@pytest.mark.parametrize("added_edge", [(0, 1), (5, 6)])
def test_dense_graph_with_internal_edges_retains_directed_model(added_edge):
    graph = nx.complete_bipartite_graph(5, 4)
    graph.add_edge(*added_edge)
    assert not uniform.prefer_node_model(_problem(graph, list(range(5))))


def test_connected_terminals_need_no_steiner_choices():
    graph = nx.cycle_graph(8)
    assert uniform.prefer_node_model(_problem(graph, [0, 1, 2]))


@pytest.mark.parametrize(
    "case,expected", [("grid", False), ("dense", True), ("cube", True)]
)
def test_original_benchmark_structures(case, expected):
    if case == "grid":
        graph = nx.grid_2d_graph(8, 8)
        terminals = [node for node in graph if all(value % 2 == 0 for value in node)]
    elif case == "dense":
        graph = nx.complete_bipartite_graph(16, 32)
        terminals = list(range(16))
    else:
        graph = nx.hypercube_graph(5)
        terminals = [node for node in graph if sum(node) % 2 == 0]
    assert uniform.prefer_node_model(_problem(graph, terminals)) is expected


def test_leaf_chains_and_disconnected_junk_do_not_change_selection_or_state():
    graph = nx.complete_bipartite_graph(5, 4)
    nx.add_path(graph, [0, "leaf-a", "leaf-b", "leaf-c"])
    nx.add_path(graph, [5, "leaf-d", "leaf-e"])
    graph.add_edges_from(nx.complete_graph([("junk", n) for n in range(12)]).edges)
    graph.add_node("isolated")
    graph.nodes[0]["metadata"] = {"values": [1, 2]}
    graph[0][5]["metadata"] = {"values": [3, 4]}
    problem = _problem(graph, list(range(5)))
    graph_before = _snapshot(graph)
    tracker_before = deepcopy(vars(problem.reduction_tracker))
    groups_before = deepcopy(problem.terminal_groups)

    assert uniform.prefer_node_model(problem)
    assert uniform.prefer_node_model(problem)

    assert problem.graph is graph
    assert _snapshot(graph) == graph_before
    assert vars(problem.reduction_tracker) == tracker_before
    assert problem.terminal_groups == groups_before


def test_terminal_in_another_component_cannot_be_ignored():
    graph = nx.complete_bipartite_graph(5, 4)
    graph.add_edge("other-terminal", "other-neighbor")
    assert not uniform.prefer_node_model(
        _problem(graph, list(range(5)) + ["other-terminal"])
    )


def test_relabeling_order_and_duplicate_terminals_do_not_change_selection():
    graph = nx.complete_bipartite_graph(5, 4)
    labels = {node: ("node", node) if node % 2 else str(node) for node in graph}
    relabeled = nx.Graph()
    relabeled.add_nodes_from(labels[node] for node in reversed(list(graph)))
    relabeled.add_edges_from(
        (labels[right], labels[left]) for left, right in reversed(list(graph.edges))
    )
    terminals = [labels[node] for node in reversed(range(5))] + [labels[0]]
    assert uniform.prefer_node_model(_problem(relabeled, terminals))


@pytest.mark.parametrize(
    "setting,expected", [(None, False), ("auto", False), ("1", True), ("0", False)]
)
def test_force_and_disable_override_only_the_performance_preference(
    monkeypatch, setting, expected
):
    if setting is not None:
        monkeypatch.setenv("STEINERPY_UNIFORM_NODE_MODEL", setting)
    problem = _problem(nx.cycle_graph(6), [0, 2, 4])
    solution = problem.get_solution(time_limit=20)
    assert solution.gap == 0 and solution.objective == 4
    assert (problem.cut_stats.get("formulation") == "uniform_node") is expected


@pytest.mark.parametrize("kind", ["unequal", "constrained", "forest"])
def test_force_does_not_bypass_mathematical_eligibility(monkeypatch, kind):
    def unexpected_node_model(*args, **kwargs):
        raise AssertionError("Forcing the node model bypassed mathematical eligibility")

    monkeypatch.setenv("STEINERPY_UNIFORM_NODE_MODEL", "1")
    monkeypatch.setattr(uniform, "solve_uniform_node", unexpected_node_model)
    graph = nx.cycle_graph(6)
    if kind == "unequal":
        graph[0][1]["weight"] = 2
        problem = _problem(graph, [0, 2, 4])
    elif kind == "constrained":
        problem = SteinerProblem(graph, [[0, 2, 4]], preprocess=False, max_degree=2)
    else:
        problem = SteinerProblem(graph, [[0, 1], [3, 4]], preprocess=False)

    solution = problem.get_solution(time_limit=20)
    assert solution.gap == 0
    assert problem.cut_stats.get("formulation") != "uniform_node"


@pytest.mark.parametrize(
    "case,expected", [("incidence", True), ("connected", True), ("cycle", False)]
)
def test_default_public_dispatch_selects_expected_formulation(case, expected):
    if case == "incidence":
        graph = nx.complete_bipartite_graph(5, 3)
        terminals, optimum = list(range(5)), 5
    else:
        graph = nx.cycle_graph(8)
        terminals, optimum = ([0, 1, 2], 2) if case == "connected" else ([0, 2, 4], 4)
    problem = Problem(graph, terminals=terminals, preprocess=False)

    solution = problem.get_solution(time_limit=20)

    assert solution.gap == 0 and solution.objective == optimum
    assert (
        problem.implementation.cut_stats.get("formulation") == "uniform_node"
    ) is expected
    tree = nx.Graph(solution.edges)
    assert nx.is_tree(tree) and set(terminals) <= set(tree)


def test_selection_uses_reduced_working_graph_and_preserves_backmapping():
    graph = nx.cycle_graph(6)
    terminals = [0, 2, 4]
    assert not uniform.prefer_node_model(_problem(graph, terminals))
    problem = SteinerProblem(graph, [terminals], heavy=False, contract_terminals=False)
    tracker_before = deepcopy(vars(problem.reduction_tracker))
    assert problem.graph.number_of_nodes() == 3
    assert uniform.uniform_cost(problem) == 2
    assert uniform.prefer_node_model(problem)

    solution = problem.get_solution(time_limit=20)

    assert solution.gap == 0 and solution.objective == 4
    assert problem.cut_stats["formulation"] == "uniform_node"
    assert len(solution.selected_edges) == 2 and len(solution.edges) == 4
    assert vars(problem.reduction_tracker) == tracker_before
    assert _snapshot(graph) == _snapshot(nx.cycle_graph(6))
