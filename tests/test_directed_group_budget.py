"""Budgeted directed groups checked against original-arc subset enumeration."""

import itertools
import math

import networkx as nx
import pytest

from steinerpy import BudgetSolution, DirectedGroupSteinerProblem, Problem
from steinerpy import mathematical_model as mm
from steinerpy import objects
from .test_directed_group_steiner import random_directed_group_instance, solver


def exhaustive_groups(graph, groups, root, budget, degree=None, hops=None):
    best = 0
    for flags in itertools.product((False, True), repeat=graph.number_of_edges()):
        edges = [e for e, keep in zip(graph.edges, flags) if keep]
        if sum(graph.edges[e].get("cost", 1) for e in edges) > budget:
            continue
        if hops is not None and len(edges) > hops:
            continue
        selected = nx.DiGraph()
        selected.add_nodes_from(graph)
        selected.add_edges_from(edges)
        if degree is not None and any(d > degree for _, d in selected.degree()):
            continue
        reached = {root} | nx.descendants(selected, root)
        best = max(best, sum(not reached.isdisjoint(g) for g in groups))
    return best


def check_solution(graph, groups, root, budget, solution, degree=None, hops=None):
    assert isinstance(solution, BudgetSolution)
    assert all(graph.has_edge(*e) for e in solution.edges)
    assert sum(graph.edges[e].get("cost", 1) for e in solution.edges) <= budget + 1e-7
    selected = nx.DiGraph()
    selected.add_node(root)
    selected.add_edges_from(solution.edges)
    reached = {root} | nx.descendants(selected, root)
    assert set(selected) == reached
    if degree is not None:
        assert all(d <= degree for _, d in selected.degree())
    if hops is not None:
        assert len(solution.edges) <= hops
    missed = {
        "group_{}".format(i): 1 for i, g in enumerate(groups) if reached.isdisjoint(g)
    }
    assert solution.penalties == missed
    assert solution.total_terminals == len(groups)
    assert (
        solution.objective == solution.connected_terminals == len(groups) - len(missed)
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("budget", [0, 2, 5])
@pytest.mark.parametrize(
    "degree, hops", [(None, None), (1, None), (2, None), (2, 2), (None, 1), (0, 0)]
)
def test_budgeted_groups_match_original_graph_oracle(
    seed, budget, degree, hops, solver
):
    graph, groups, root = random_directed_group_instance(seed, n=5)
    for i, edge in enumerate(graph.edges):
        weight = graph.edges[edge].pop("weight")
        if i % 3 == 0:
            graph.edges[edge]["cost"] = 0
        elif i % 3 == 1:
            graph.edges[edge]["cost"] = weight
        # Remaining edges deliberately have no cost attribute (default one).
    groups += [[root, groups[0][0]], [groups[0][0], groups[1][0]], list(groups[0])]
    original = graph.copy()
    problem = Problem(
        graph,
        groups=groups,
        root=root,
        budget=budget,
        max_degree=degree,
        hop_limit=hops,
        weight="cost",
    )
    solution = problem.get_solution(solver=solver)
    check_solution(graph, groups, root, budget, solution, degree, hops)
    assert solution.gap == 0
    assert solution.objective == exhaustive_groups(
        graph, groups, root, budget, degree, hops
    )
    coefficients = problem._spec.objective(problem.variant)
    assert (
        coefficients.value(solution.edges, penalty_count=len(solution.penalties))
        == solution.objective
    )
    assert nx.utils.graphs_equal(graph, original)


@pytest.mark.parametrize("api", ["legacy", "inferred", "explicit"])
def test_zero_budget_shared_members_root_count_and_label_collisions(api, solver):
    graph = nx.DiGraph()
    graph.add_edge("r", "__group_0__", cost=0)
    graph.add_edge("__group_0__", "t", cost=3)
    groups = [["__group_0__"], ["__group_0__"], ["r"], ["t"]]
    options = dict(
        groups=groups, root="r", budget=0, max_degree=1, hop_limit=1, weight="cost"
    )
    if api == "legacy":
        problem = DirectedGroupSteinerProblem(graph, **options)
    else:
        problem = Problem(
            graph, variant="directed_group" if api == "explicit" else None, **options
        )
    solution = problem.get_solution(solver=solver)
    check_solution(graph, groups, "r", 0, solution, 1, 1)
    assert solution.objective == 3 and solution.gap == 0
    assert solution.penalties == {"group_3": 1}


@pytest.mark.parametrize("root_in_group", [False, True])
def test_budgeted_groups_with_no_original_arcs(root_in_group, solver):
    graph = nx.DiGraph()
    graph.add_nodes_from(["r", "unreachable"])
    groups = [["r"]] if root_in_group else [["unreachable"]]
    solution = Problem(
        graph, root="r", groups=groups, budget=0, max_degree=0, hop_limit=0
    ).get_solution(solver=solver)
    check_solution(graph, groups, "r", 0, solution, 0, 0)
    assert solution.objective == int(root_in_group) and solution.gap == 0
    assert solution.edges == []


@pytest.mark.parametrize("budget", [-1, True, "3", math.inf, math.nan])
def test_legacy_rejects_invalid_group_budget(budget):
    with pytest.raises(ValueError, match="budget.*finite and non-negative"):
        DirectedGroupSteinerProblem(
            nx.path_graph(2, create_using=nx.DiGraph), [[1]], root=0, budget=budget
        )


@pytest.mark.parametrize("outcome", ["unproven", "missing", "invalid"])
def test_budgeted_group_solver_status_is_preserved(solver, monkeypatch, outcome):
    graph = nx.path_graph(3, create_using=nx.DiGraph)
    problem = Problem(
        graph, root=0, groups=[[1], [2]], budget=2, max_degree=2, hop_limit=2
    )
    monkeypatch.setattr(
        mm,
        "_gurobi_outcome" if solver == "gurobi" else "_highs_outcome",
        lambda *a: ("incomplete", outcome != "missing"),
    )
    if outcome == "invalid":
        monkeypatch.setattr(mm, "_groups_reachable", lambda *a: False)
    if outcome in {"missing", "invalid"}:
        with pytest.raises(RuntimeError, match="valid feasible incumbent"):
            problem.get_solution(solver=solver)
    else:
        solution = problem.get_solution(solver=solver)
        assert solution.objective == 2 and math.isinf(solution.gap)


@pytest.mark.parametrize(
    "case", ["cost", "coverage", "unclaimed", "disconnected", "nonoriginal"]
)
def test_decoded_budget_groups_validate_original_space(monkeypatch, case):
    graph = nx.DiGraph()
    graph.add_edge(0, 1, cost=2)
    graph.add_edge(2, 3, cost=0)
    problem = DirectedGroupSteinerProblem(
        graph, [[1]], root=0, budget=1 if case == "cost" else 2, weight="cost"
    )
    edges = [] if case == "coverage" else [(0, 1)]
    if case == "disconnected":
        edges += [(2, 3)]
    if case == "nonoriginal":
        edges += [(0, 3)]
    monkeypatch.setattr(
        objects.SteinerProblem,
        "get_solution",
        lambda *a, **k: BudgetSolution(
            objective=1 if case == "unclaimed" else 2,
            connected_terminals=1 if case == "unclaimed" else 2,
            gap=0,
            runtime=0,
            selected_edges=edges,
            original_selected_edges=edges,
        ),
    )
    if case in {"cost", "coverage", "nonoriginal"}:
        with pytest.raises(
            RuntimeError,
            match={
                "cost": "original-edge budget",
                "coverage": "overstates",
                "nonoriginal": "non-original",
            }[case],
        ):
            problem.get_solution()
    else:
        solution = problem.get_solution()
        assert solution.edges == [(0, 1)] and solution.objective == 1
        assert solution.gap == (math.inf if case == "unclaimed" else 0)
