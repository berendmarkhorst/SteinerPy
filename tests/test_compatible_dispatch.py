"""Faster dispatch must preserve the original feature and result contracts."""

import networkx as nx
import pytest

from benchmarks.pace2018 import composition_problem
from steinerpy import Problem, SteinerProblem
from steinerpy._feature_composition import FeatureComposition
from steinerpy.objects import NodeWeightedSolution


@pytest.mark.parametrize("costs", [{}, None, {0: 0, 1: 0, 2: 0}])
@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("terminals", [[0, 2], [[0, 1], [2]]])
def test_inactive_node_costs_use_classic_and_preserve_composed_results(
    costs, preprocess, terminals
):
    graph = nx.cycle_graph(4)
    nx.set_edge_attributes(graph, 1, "cost")
    problem = Problem(
        graph,
        terminals=terminals,
        node_costs=costs,
        weight="cost",
        preprocess=preprocess,
    )
    expected_type = SteinerProblem if terminals == [0, 2] else FeatureComposition
    assert type(problem.implementation) is expected_type
    actual = problem.get_solution()
    expected = FeatureComposition(problem._spec).get_solution()
    assert type(actual) is type(expected) is NodeWeightedSolution
    assert actual.objective == expected.objective
    assert actual.gap == expected.gap == 0
    assert actual.total_cost == expected.total_cost
    assert actual.penalties == expected.penalties == {}
    assert actual.selected_edges == actual.original_selected_edges
    assert set(n for group in problem._spec.terminal_groups for n in group) <= set(
        actual.selected_nodes
    )


@pytest.mark.parametrize(
    "options",
    [
        {"node_budget": None},
        {"penalty_budget": None},
        {"penalty_cost": None},
        {"partial_terminals": []},
    ],
)
def test_other_inactive_options_use_classic(options):
    problem = Problem(nx.path_graph(3), terminals=[0, 2], **options)
    assert type(problem.implementation) is SteinerProblem
    assert not problem.implementation.preprocess  # Composition's default.
    assert problem.get_solution().objective == 2


@pytest.mark.parametrize(
    "options",
    [
        {"node_costs": {1: 2}},
        {"groups": [[1, 2]]},
        {"root": 0},
        {"node_prizes": {}},
        {"node_weights": {}},
        {"budget": 0},
        {"node_budget": 0},
        {"penalty_cost": 0},
        {"penalty_budget": 0},
        {"max_degree": 2},
        {"hop_limit": 2},
        {"partial_terminals": [2]},
    ],
)
def test_active_features_keep_composition(options):
    opts = dict(node_costs={})
    opts.update(options)
    problem = Problem(nx.path_graph(3), terminals=[0, 2], **opts)
    assert not isinstance(problem.implementation, SteinerProblem)
    # Existing exhaustive combination suites check these strategies' semantics.


def test_direction_and_explicit_variants_are_preserved():
    problem = Problem(
        nx.path_graph(3).to_directed(), terminals=[2], root=0, node_costs={}
    )
    assert not isinstance(problem.implementation, SteinerProblem)
    assert problem.get_solution().objective == 2
    plain = Problem(nx.path_graph(3), terminals=[0, 2], variant="steiner")
    assert type(plain.implementation) is SteinerProblem and plain.variant == "steiner"


@pytest.mark.parametrize("preprocess", [False, True])
def test_enumeration_keeps_ties_even_with_zero_edges(preprocess):
    graph = nx.cycle_graph(4)
    nx.set_edge_attributes(graph, 1, "weight")
    graph[0][1]["weight"] = graph[0][3]["weight"] = 0
    problem = Problem(
        graph,
        terminals=[0, 2],
        node_costs={},
        preprocess=preprocess,
        enumeration_safe=True,
    )
    pool = problem.get_optimal_solutions(limit=10)
    assert pool.exhausted and len(pool) == 2
    assert all(
        type(s) is NodeWeightedSolution and s.objective == 1 and s.gap == 0
        for s in pool
    )
    assert {frozenset(map(frozenset, s.edges)) for s in pool} == {
        frozenset(map(frozenset, [(0, 1), (1, 2)])),
        frozenset(map(frozenset, [(0, 3), (3, 2)])),
    }


def test_snapshot_and_isolated_terminals_are_preserved():
    graph = nx.empty_graph(3)
    costs = {0: 0}
    problem = Problem(graph, terminals=[[0], [2]], node_costs=costs)
    graph.clear()
    costs[0] = 99
    solution = problem.get_solution()
    assert solution.edges == [] and solution.objective == 0
    assert solution.selected_nodes == [0, 2]


@pytest.mark.parametrize(
    "options",
    [
        {"node_costs": {99: 0}},
        {"node_costs": {0: float("nan")}},
        {"node_costs": []},
        {"node_prizes": None},
        {"unknown": False},
    ],
)
def test_validation_precedes_inactive_feature_selection(options):
    opts = dict(node_costs={})
    opts.update(options)
    with pytest.raises((TypeError, ValueError)):
        Problem(nx.path_graph(3), terminals=[0, 2], **opts)


def test_selected_solver_and_dependency_errors_are_forwarded(monkeypatch):
    problem = Problem(nx.path_graph(3), terminals=[0, 2], node_costs={})
    calls = []

    def unavailable(**kwargs):
        calls.append(kwargs)
        raise ImportError("requested backend unavailable")

    monkeypatch.setattr(problem.implementation, "get_solution", unavailable)
    with pytest.raises(ImportError, match="requested backend"):
        problem.get_solution(solver="gurobi", threads=1, time_limit=2)
    assert calls == [dict(solver="gurobi", threads=1, time_limit=2)]


def test_comparison_benchmark_still_forces_generic_formulation():
    graph = nx.path_graph(3)
    problem = composition_problem(graph, [[0, 2]])
    assert type(problem) is FeatureComposition
    assert problem.get_solution().objective == 2


def test_infeasible_fast_path_cannot_return_an_empty_success():
    with pytest.raises(RuntimeError):
        Problem(nx.empty_graph(3), terminals=[0, 2], node_costs={}).get_solution()
