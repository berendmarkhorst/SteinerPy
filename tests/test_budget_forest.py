"""Budgeted forest requirements can overlap and share purchased edges."""

import itertools
import random

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy import mathematical_model as mm


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


def check_solution(graph, groups, budget, max_degree=None, *, solver="highs"):
    problem = Problem(
        graph,
        terminals=groups,
        budget=budget,
        max_degree=max_degree,
        weight="cost",
        preprocess=False,
        **({"root": groups[0][0]} if graph.is_directed() else {}),
    )
    solution = problem.get_solution(solver=solver)
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
def test_budgeted_path_forest_matches_edge_subset_oracle(groups, budget, solver):
    graph = nx.path_graph(4)
    nx.set_edge_attributes(graph, 1, "cost")
    check_solution(graph, groups, budget, solver=solver)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("max_degree", [None, 1, 2])
def test_budgeted_shared_forest_matches_weighted_oracle(seed, max_degree, solver):
    rng = random.Random(seed)
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2)
    nx.set_edge_attributes(
        graph, {edge: rng.randint(0, 3) for edge in graph.edges}, "cost"
    )
    groups = [[0, 1, 2], [3, 2], [1, 3]]
    check_solution(graph, groups, budget=3, max_degree=max_degree, solver=solver)


def test_budgeted_forest_with_isolated_shared_root(solver):
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    graph.add_edge(1, 2, cost=1)
    check_solution(graph, [[0, 1], [0, 2]], budget=1, solver=solver)


def test_original_unweighted_overlapping_group_example(solver):
    solution = Problem(
        nx.path_graph(3),
        terminals=[[0, 1], [0, 2]],
        budget=1,
        preprocess=False,
    ).get_solution(solver=solver)
    assert solution.gap == 0
    assert solution.objective == 3
    assert solution.edges == [(0, 1)]


@pytest.mark.parametrize("budget", [0, 1, 4, 8])
def test_directed_budget_still_requires_forward_arcs(budget, solver):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (1, 2, 3), (2, 0, 0), (3, 1, 0)], weight="cost"
    )
    check_solution(graph, [[0, 2, 3]], budget, solver=solver)


@pytest.mark.parametrize("groups", [[[0], [1]], [[0, 1], [0, 2]]])
def test_edgeless_budget_model(groups, solver):
    graph = nx.empty_graph(3)
    check_solution(graph, groups, budget=0, solver=solver)


def test_budget_respects_directed_hop_limit(solver):
    graph = nx.path_graph(3, create_using=nx.DiGraph)
    problem = Problem(graph, root=0, terminals=[1, 2], hop_limit=1, budget=2)
    solution = problem.get_solution(solver=solver)
    assert solution.objective == 2
    assert solution.gap == 0
    assert len(solution.edges) == 1


def test_budget_tiny_limit_is_status_safe(solver):
    problem = Problem(
        nx.complete_graph(14), terminals=[[0, 4], [4, 7]], budget=3, preprocess=False
    )
    try:
        solution = problem.get_solution(solver=solver, time_limit=1e-9)
    except RuntimeError as exc:
        assert "valid feasible incumbent" in str(exc)
    else:
        assert len(solution.edges) <= 3
        selected = nx.Graph(solution.edges)
        selected.add_nodes_from(problem.implementation.nodes)
        assert solution.objective <= sum(
            nx.has_path(selected, group[0], node)
            for group in [[0, 4], [4, 7]]
            for node in group
        )
        assert solution.gap in (0, float("inf"))
        if solution.gap == 0:
            assert solution.objective == 4


@pytest.mark.parametrize("outcome", ["unproven", "invalid", "missing"])
def test_budget_unproven_or_invalid_incumbent_is_not_certified(
    solver, monkeypatch, outcome
):
    problem = Problem(
        nx.path_graph(3), terminals=[0, 2], budget=2, preprocess=False
    ).implementation
    model, x, _, _, _, _, penalties = mm.build_budget_model(problem, solver=solver)
    try:
        if solver == "gurobi":
            monkeypatch.setattr(
                mm,
                "_gurobi_outcome",
                lambda *args: ("incomplete", outcome != "missing"),
            )
        else:
            monkeypatch.setattr(
                mm, "_highs_outcome", lambda *args: ("incomplete", outcome != "missing")
            )
        if outcome == "invalid":
            monkeypatch.setattr(mm, "_groups_reachable", lambda *args: False)
        result = mm.run_budget_model(
            model, problem, x, penalties, return_status=True, solver=solver
        )
        assert result[-1] == "incomplete"
        assert result[0] == float("inf")
        if outcome in {"invalid", "missing"}:
            assert result[2:5] == (0, [], {})
        else:
            assert result[2] == 2
            assert len(result[3]) == 2
    finally:
        if solver == "gurobi":
            model.dispose()
