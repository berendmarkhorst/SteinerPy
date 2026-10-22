"""Public results must describe the selected original edges, including at limits."""

import math

import networkx as nx
import pytest

from benchmarks.pace2018 import validate_solution
from steinerpy import SteinerProblem
from steinerpy import mathematical_model as mm, objects


def inject_incumbent(monkeypatch, solver, objective, edges, gap):
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "0")
    monkeypatch.setattr(mm, "_check_gurobipy", lambda: None)
    suffix = "_gurobi" if solver == "gurobi" else ""
    monkeypatch.setattr(objects, "build_model" + suffix, lambda *a, **k: (None,) * 5)
    monkeypatch.setattr(
        objects,
        "run_model" + suffix,
        lambda *a, **k: (
            gap,
            0.01,
            objective,
            edges,
            "incomplete" if math.isinf(gap) else "optimal",
        ),
    )


@pytest.mark.parametrize("solver", ["highs", "gurobi"])
def test_remove_pace_style_disconnected_cycle_and_preserve_lower_bound(
    monkeypatch, solver
):
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1500435)
    graph.add_weighted_edges_from([(2, 3, 4), (3, 4, 4), (4, 5, 4), (5, 2, 4)])
    reported, lower = 1500451, 1500408.5
    inject_incumbent(
        monkeypatch, solver, reported, list(graph.edges), (reported - lower) / reported
    )

    solution = SteinerProblem(graph, [[0, 1]], preprocess=False).get_solution(
        solver=solver
    )

    assert solution.edges == [(0, 1)]
    assert solution.selected_edges == [(0, 1)]
    assert solution.objective == 1500435
    assert solution.gap == pytest.approx((1500435 - lower) / 1500435)
    checked = validate_solution(
        graph,
        [0, 1],
        solution.edges,
        solution.objective,
        solution.gap,
        1500422,
        1500422,
    )
    assert checked["status"] == "feasible"


@pytest.mark.parametrize(
    "reported,actual,gap",
    [
        (164149.99954, 164150, 0),
        (231028455.99675694, 231028456, 6.254943004591835e-5),
        (5615.999924575609, 5616, math.inf),
        (42767.9999915956, 42768, 2.3381771319586545e-5),
    ],
)
def test_pace_numeric_discrepancies_use_edge_cost_without_inventing_proof(
    monkeypatch, reported, actual, gap
):
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=actual)
    inject_incumbent(monkeypatch, "gurobi", reported, [(0, 1)], gap)
    solution = SteinerProblem(graph, [[0, 1]], preprocess=False).get_solution(
        solver="gurobi"
    )

    assert solution.objective == actual
    assert solution.gap > 0
    if math.isfinite(gap):
        lower = reported * (1 - gap)
        assert solution.gap == pytest.approx((actual - lower) / actual)
    else:
        assert math.isinf(solution.gap)
    result = validate_solution(
        graph, [0, 1], solution.edges, solution.objective, solution.gap, actual, actual
    )
    assert result["reference_optimal"] and not result["solver_proven_optimal"]


def test_cleanup_keeps_forest_requirements_and_degree_limit(monkeypatch):
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 2, 0), (4, 5, 2), (6, 7, 1)]
    )
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (6, 7)]
    inject_incumbent(monkeypatch, "highs", 6, edges, math.inf)
    solution = SteinerProblem(
        graph, [[0, 1, 3], [4, 5]], preprocess=False, max_degree=2
    ).get_solution()
    selected = nx.Graph(solution.edges)
    assert nx.is_forest(selected)
    assert max(dict(selected.degree()).values()) <= 2
    assert nx.has_path(selected, 0, 3) and nx.has_path(selected, 4, 5)
    assert not selected.has_edge(0, 2)  # Cheaper chord was never selected.
    assert solution.objective == 5 and math.isinf(solution.gap)


def test_cleanup_keeps_directed_reachability(monkeypatch):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 2), (1, 2, 3)])
    inject_incumbent(monkeypatch, "highs", 4.999999, list(graph.edges), math.inf)
    solution = objects.DirectedSteinerProblem(graph, 0, [2]).get_solution()
    assert solution.edges == [(0, 1), (1, 2)]
    assert solution.objective == 5 and math.isinf(solution.gap)


def test_cleanup_does_not_mask_an_inconsistent_bound(monkeypatch):
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2), (2, 3, 1)])
    inject_incumbent(monkeypatch, "highs", 3, list(graph.edges), 0)
    with pytest.raises(RuntimeError, match="lower bound"):
        SteinerProblem(graph, [[0, 1]], preprocess=False).get_solution()


def test_finalization_restores_fixed_edges_and_their_cost():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 2, "weight")
    problem = SteinerProblem(graph, [[0, 2]], preprocess=True)
    edges, original, objective, gap = problem._finalize_exact_incumbent([], 0, 0)
    assert edges == []
    assert set(map(frozenset, original)) == set(map(frozenset, graph.edges))
    assert objective == 4 and gap == 0
