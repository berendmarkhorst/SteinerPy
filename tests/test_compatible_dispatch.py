"""Faster dispatch must preserve the original feature and result contracts."""

import itertools
import math
import random

import networkx as nx
import pytest

from benchmarks.pace2018 import composition_problem
from steinerpy import Problem, SteinerProblem
from steinerpy._feature_composition import FeatureComposition
from steinerpy.objects import NodeWeightedSolution, Solution


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
    assert calls == [
        dict(solver="gurobi", threads=1, time_limit=2, log_file="", exact=True)
    ]


def test_composed_positional_options_keep_their_meaning(monkeypatch):
    problem = Problem(nx.path_graph(3), terminals=[0, 2], node_costs={})
    solution = problem.implementation.get_solution()
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        return solution

    monkeypatch.setattr(problem.implementation, "get_solution", solve)
    assert problem.get_solution(4, "run.log", "highs", 2, True).objective == 2
    assert calls == [
        dict(time_limit=4, log_file="run.log", solver="highs", threads=2, exact=True)
    ]


def test_composed_heuristic_restriction_is_preserved():
    problem = Problem(nx.path_graph(3), terminals=[0, 2], node_costs={})
    with pytest.raises(NotImplementedError, match="heuristic-only"):
        problem.get_solution(exact=False)


def test_comparison_benchmark_still_forces_generic_formulation():
    graph = nx.path_graph(3)
    problem = composition_problem(graph, [[0, 2]])
    assert type(problem) is FeatureComposition
    assert problem.get_solution().objective == 2


def test_infeasible_fast_path_cannot_return_an_empty_success():
    with pytest.raises(RuntimeError, match="infeasible"):
        Problem(nx.empty_graph(3), terminals=[0, 2], node_costs={}).get_solution()


@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("dp_limit", [0, 10])
@pytest.mark.parametrize("seed", range(3))
def test_mandatory_node_costs_match_composition_and_edge_subset_oracle(
    monkeypatch, preprocess, dp_limit, seed
):
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", str(dp_limit))
    graph = nx.cycle_graph(5)
    graph.add_edges_from([(0, 2), (1, 4)])
    rng = random.Random(seed)
    nx.set_edge_attributes(graph, {e: rng.randint(1, 7) for e in graph.edges}, "cost")
    terminals, costs = [0, 2, 4], {0: 3, 2: 5, 1: 0}
    problem = Problem(
        graph,
        terminals=terminals,
        node_costs=costs,
        weight="cost",
        preprocess=preprocess,
    )
    assert type(problem.implementation) is SteinerProblem
    best = math.inf
    for count in range(len(graph.edges) + 1):
        for edges in itertools.combinations(graph.edges, count):
            selected = nx.Graph(edges)
            selected.add_nodes_from(terminals)
            if set(terminals) <= nx.node_connected_component(selected, 0):
                best = min(best, sum(graph.edges[e]["cost"] for e in edges) + 8)
    actual = problem.get_solution()
    expected = FeatureComposition(problem._spec).get_solution()
    assert type(actual) is type(expected) is NodeWeightedSolution
    assert actual.objective == actual.total_cost == expected.objective == best
    assert actual.gap == expected.gap == 0
    assert set(terminals) <= set(actual.selected_nodes)
    assert actual.penalties == {}


@pytest.mark.parametrize("edge_cost", [0, 0.25, 10])
@pytest.mark.parametrize("gap", [0, 0.2, math.inf])
def test_mandatory_cost_gap_preserves_bound_difference(monkeypatch, edge_cost, gap):
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=edge_cost)
    problem = Problem(graph, terminals=[0, 1], node_costs={0: 5})
    incumbent = Solution(gap, 0.1, edge_cost, [(0, 1)])
    monkeypatch.setattr(problem.implementation, "get_solution", lambda **kw: incumbent)
    actual = problem.get_solution()
    assert actual.objective == edge_cost + 5
    if math.isfinite(gap):
        assert actual.gap * max(1, actual.objective) == pytest.approx(
            gap * max(1, edge_cost)
        )
    else:
        assert math.isinf(actual.gap)


@pytest.mark.parametrize("preprocess", [False, True])
def test_mandatory_costs_preserve_snapshots_and_enumerated_ties(preprocess):
    graph = nx.cycle_graph(4)
    costs = {0: 3, 2: 5}
    problem = Problem(graph, terminals=[0, 2], node_costs=costs, preprocess=preprocess)
    graph.clear()
    costs[0] = 99
    assert problem.get_solution().objective == 10
    pool = problem.get_optimal_solutions(limit=10)
    assert pool.exhausted and len(pool) == 2
    assert all(s.objective == s.total_cost == 10 and s.gap == 0 for s in pool)
    assert {frozenset(map(frozenset, s.edges)) for s in pool} == {
        frozenset(map(frozenset, [(0, 1), (1, 2)])),
        frozenset(map(frozenset, [(0, 3), (3, 2)])),
    }


def test_mandatory_singleton_cost_is_charged_without_edges():
    problem = Problem(nx.empty_graph(3), terminals=[1], node_costs={1: 7})
    assert type(problem.implementation) is SteinerProblem
    result = problem.get_solution()
    assert result.edges == [] and result.selected_nodes == [1]
    assert result.objective == result.total_cost == 7 and result.gap == 0


@pytest.mark.parametrize(
    "options",
    [
        {"node_costs": {0: 3, 1: 1}},
        {"terminals": [[0], [2]]},
        {"groups": [[1, 2]]},
        {"root": 0},
        {"node_prizes": {}},
        {"node_weights": {}},
        {"budget": 10},
        {"node_budget": 3},
        {"penalty_cost": 0},
        {"penalty_budget": 0},
        {"max_degree": 2},
        {"hop_limit": 2},
        {"partial_terminals": [2]},
    ],
)
def test_mandatory_costs_do_not_bypass_other_active_features(options):
    opts = dict(terminals=[0, 2], node_costs={0: 3})
    opts.update(options)
    problem = Problem(nx.path_graph(3), **opts)
    assert not isinstance(problem.implementation, SteinerProblem)
