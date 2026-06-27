"""Tests for the Max-Weight Connected Subgraph with vertex budget (MWCSPB, Ch. 5.6)."""

import importlib

import networkx as nx
import pytest

from steinerpy import BudgetedMaxWeightConnectedSubgraph


def _require_gurobi():
    if importlib.util.find_spec("gurobipy") is None:
        pytest.skip("gurobipy is not installed.")
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        gp.Model(env=env).dispose()
        env.dispose()
    except Exception:
        pytest.skip("Gurobi license not available.")


def _instance():
    """Path r - a - b. Reaching the high-weight b (10) requires connector a."""
    G = nx.Graph()
    G.add_edge("r", "a", weight=1)
    G.add_edge("a", "b", weight=1)
    node_weights = {"r": 5, "a": 1, "b": 10}
    node_costs = {"r": 0, "a": 1, "b": 5}
    return G, node_weights, node_costs


SOLVERS = ["highs", "gurobi"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_budget_allows_full_subgraph(solver):
    if solver == "gurobi":
        _require_gurobi()
    G, w, c = _instance()
    sol = BudgetedMaxWeightConnectedSubgraph(
        G, w, c, node_budget=6, root="r"
    ).get_solution(solver=solver)
    assert sol.objective == pytest.approx(16.0)
    assert set(sol.selected_nodes) == {"r", "a", "b"}
    spent = sum(c[v] for v in sol.selected_nodes)
    assert spent <= 6 + 1e-9


@pytest.mark.parametrize("solver", SOLVERS)
def test_tight_budget_excludes_expensive_node(solver):
    if solver == "gurobi":
        _require_gurobi()
    G, w, c = _instance()
    sol = BudgetedMaxWeightConnectedSubgraph(
        G, w, c, node_budget=5, root="r"
    ).get_solution(solver=solver)
    # b (cost 5) plus its connector a (cost 1) = 6 > 5, so b is dropped.
    assert sol.objective == pytest.approx(6.0)
    assert set(sol.selected_nodes) == {"r", "a"}
    assert sum(c[v] for v in sol.selected_nodes) <= 5 + 1e-9


@pytest.mark.parametrize("solver", SOLVERS)
def test_zero_budget_keeps_only_root(solver):
    if solver == "gurobi":
        _require_gurobi()
    G, w, c = _instance()
    sol = BudgetedMaxWeightConnectedSubgraph(
        G, w, c, node_budget=0, root="r"
    ).get_solution(solver=solver)
    assert sol.objective == pytest.approx(5.0)
    assert set(sol.selected_nodes) == {"r"}


def test_highs_and_gurobi_match():
    _require_gurobi()
    G, w, c = _instance()
    highs = BudgetedMaxWeightConnectedSubgraph(G, w, c, node_budget=6, root="r").get_solution(solver="highs")
    gurobi = BudgetedMaxWeightConnectedSubgraph(G, w, c, node_budget=6, root="r").get_solution(solver="gurobi")
    assert highs.objective == pytest.approx(gurobi.objective)


def test_unknown_solver_rejected():
    G, w, c = _instance()
    with pytest.raises(ValueError):
        BudgetedMaxWeightConnectedSubgraph(G, w, c, node_budget=6, root="r").get_solution(solver="cplex")
