"""Numerical reconciliation must preserve feasible solutions without false proofs."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy._feature_composition import FeatureComposition
from steinerpy._selection_model import SelectionModel
from .test_directed_group_steiner import solver


def problem(mode):
    options = dict(terminals=[0, 2], node_costs={1: 1})
    if mode == "prizes":
        options["node_prizes"] = {1: 1, 3: 5}
    elif mode == "rewards":
        options.update(node_weights={0: 1, 1: -1, 2: 3}, max_degree=2)
    elif mode == "zero":
        options.update(node_weights={0: 0}, max_degree=2)
    else:
        options.update(budget=2, node_budget=1)
    graph = nx.path_graph(3)
    graph.add_node(3)
    configured = Problem(graph, **options)
    assert isinstance(configured.implementation, FeatureComposition)
    return configured


def perturb_objective(monkeypatch, delta):
    original = SelectionModel.objective_value
    monkeypatch.setattr(
        SelectionModel, "objective_value", lambda model: original(model) + delta
    )


@pytest.mark.parametrize(
    "mode,expected", [("prizes", 8), ("rewards", 3), ("zero", 0), ("coverage", 2)]
)
@pytest.mark.parametrize("delta", [-2.1e-7, 2.1e-7])
def test_roundoff_keeps_valid_original_solution_without_proof(
    monkeypatch, solver, mode, expected, delta
):
    perturb_objective(monkeypatch, delta)
    p = problem(mode)
    solution = p.get_solution(solver=solver)
    assert solution.objective == expected
    assert set(solution.selected_nodes) == {0, 1, 2}
    assert {frozenset(e) for e in solution.edges} == {
        frozenset((0, 1)),
        frozenset((1, 2)),
    }
    assert solution.total_cost == 3
    assert math.isinf(solution.gap)
    p.implementation.decode(solution.edges, solution.selected_nodes)


@pytest.mark.parametrize("mode", ["prizes", "coverage"])
@pytest.mark.parametrize(
    "delta", [2e-6, -2e-6, 0.01, -0.01, math.nan, math.inf, -math.inf]
)
def test_larger_and_nonfinite_discrepancies_still_fail(
    monkeypatch, solver, mode, delta
):
    perturb_objective(monkeypatch, delta)
    with pytest.raises(RuntimeError, match="objective disagrees"):
        problem(mode).get_solution(solver=solver)


@pytest.mark.parametrize("mode", ["prizes", "rewards", "zero", "coverage"])
def test_consistent_solve_keeps_its_optimality_claim(solver, mode):
    assert problem(mode).get_solution(solver=solver).gap == 0


@pytest.mark.parametrize("mode", ["prizes", "coverage"])
def test_unproved_selection_is_not_enumerated_as_optimal(monkeypatch, solver, mode):
    perturb_objective(monkeypatch, 2.1e-7)
    pool = problem(mode).get_optimal_solutions(solver=solver)
    assert len(pool) == 0 and not pool.exhausted


def test_invalid_original_selection_is_still_rejected(monkeypatch, solver):
    perturb_objective(monkeypatch, 2.1e-7)
    monkeypatch.setattr(SelectionModel, "selection", lambda model: ([], [0, 2]))
    with pytest.raises(RuntimeError, match="disconnected"):
        problem("prizes").get_solution(solver=solver)


def test_issue97_reproduction_in_a_single_threaded_process():
    # A fresh process avoids HiGHS' global thread scheduler from other tests.
    script = """
import json, math
from benchmarks.profile_composition import make_case
from steinerpy import Problem
graph, options = make_case('prize-forest', 43)
solution = Problem(graph, preprocess=False, **options).get_solution(
    time_limit=5, solver='highs', threads=1)
print(json.dumps(dict(objective=solution.objective, finite_gap=math.isfinite(solution.gap))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["objective"] == 24
