"""Regression tests for time-limited and otherwise incomplete exact solves."""

import importlib.util
import math
from types import SimpleNamespace

import highspy as hp
import networkx as nx
import pytest

import steinerpy.mathematical_model as mm
import steinerpy.objects as objects
from steinerpy import (
    BudgetedMaxWeightConnectedSubgraph,
    NodeWeightedSteinerProblem,
    PrizeCollectingProblem,
    SteinerProblem,
)


def _cycle_problem(n=20):
    graph = nx.cycle_graph(n)
    nx.set_edge_attributes(graph, 1, "weight")
    return SteinerProblem(graph, [[0, n // 4, n // 2]], preprocess=False)


def _require_gurobi():
    if importlib.util.find_spec("gurobipy") is None:
        pytest.skip("gurobipy is not installed")
    try:
        import gurobipy as gp

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)
        model.dispose()
        env.dispose()
    except Exception:
        pytest.skip("Gurobi license is not available")


def test_tiny_positive_limit_has_no_bogus_core_incumbent(monkeypatch):
    """A positive limit can expire before the first MIP solve starts."""
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "0")
    problem = _cycle_problem()
    model, x, _y1, y2, z = mm.build_model(problem, time_limit=1e-9)

    gap, _runtime, objective, edges, status = mm.run_model(
        model, problem, x, y2, z, return_status=True
    )

    assert status == "incomplete"
    assert math.isinf(gap) and math.isinf(objective)
    assert edges == []


def test_deadline_after_violated_cut_discards_previous_incumbent(monkeypatch):
    """The pre-cut incumbent is invalid once its violated cut has been added."""
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "0")
    problem = _cycle_problem(8)
    model, x, _y1, y2, z = mm.build_model(problem, time_limit=1.0)

    real_separate = mm.find_violated_cuts
    separated = []

    def recording_separation(*args, **kwargs):
        cuts = real_separate(*args, **kwargs)
        separated.append(cuts)
        return cuts

    # start=0; permit the first solve at t=0; expire before the re-solve at t=2.
    times = iter((0.0, 0.0, 2.0, 2.0))
    monkeypatch.setattr(mm, "time", SimpleNamespace(time=lambda: next(times)))
    monkeypatch.setattr(mm, "find_violated_cuts", recording_separation)

    gap, _runtime, objective, edges, status = mm.run_model(
        model, problem, x, y2, z, return_status=True
    )

    assert separated and separated[0]
    assert status == "incomplete"
    assert math.isinf(gap) and math.isinf(objective)
    assert edges == []


def test_public_get_solution_rejects_disconnected_incomplete_result(
    monkeypatch,
):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")
    problem = SteinerProblem(graph, [[0, 2]], preprocess=False)
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "0")
    monkeypatch.setattr(
        objects,
        "run_model",
        lambda *args, **kwargs: (math.inf, 0.01, 0.0, [], "incomplete"),
    )

    with pytest.raises(RuntimeError, match="does not connect"):
        problem.get_solution(decompose=False)


def test_public_get_solution_accepts_valid_unproven_incumbent(monkeypatch):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")
    problem = SteinerProblem(graph, [[0, 2]], preprocess=False)
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "0")
    monkeypatch.setattr(
        objects,
        "run_model",
        lambda *args, **kwargs: (
            math.inf,
            0.01,
            2.0,
            [(0, 1), (1, 2)],
            "incomplete",
        ),
    )

    solution = problem.get_solution(decompose=False)

    assert solution.edges == [(0, 1), (1, 2)]
    assert solution.objective == pytest.approx(2.0)
    assert math.isinf(solution.gap)


def test_preprocessed_tiny_limit_reports_no_valid_solution(monkeypatch):
    """Preprocessing/back-mapping must not turn a missing incumbent into a tree."""
    graph = nx.complete_graph(14)
    nx.set_edge_attributes(graph, 1, "weight")
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "0")
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "0")
    problem = SteinerProblem(
        graph,
        [[0, 4, 9]],
        preprocess=True,
        heavy=False,
        contract_terminals=False,
        bound_based=False,
    )

    with pytest.raises(RuntimeError, match="before finding"):
        problem.get_solution(time_limit=1e-9, decompose=False)


def test_core_infeasible_status_is_distinct(monkeypatch):
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "0")
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1)
    graph.add_edge(2, 3, weight=1)
    problem = SteinerProblem(graph, [[0, 3]], preprocess=False)
    model, x, _y1, y2, z = mm.build_model(problem, time_limit=5.0)

    gap, _runtime, objective, edges, status = mm.run_model(
        model, problem, x, y2, z, return_status=True
    )

    assert status == "infeasible"
    assert math.isinf(gap) and math.isinf(objective)
    assert edges == []


def test_specialized_highs_runners_reject_no_incumbent():
    graph = nx.cycle_graph(20)
    nx.set_edge_attributes(graph, 1, "weight")

    pc = PrizeCollectingProblem(graph, [[0, 10]], {v: 1 for v in graph}, penalty_cost=1)
    model, x, _y1, _y2, _z, _f, node_vars, penalty_vars = (
        mm.build_prize_collecting_model(pc, time_limit=1e-9)
    )
    pc_result = mm.run_prize_collecting_model(
        model, pc, x, node_vars, penalty_vars, return_status=True
    )
    assert pc_result[-1] == "incomplete"
    assert math.isinf(pc_result[0]) and math.isinf(pc_result[2])
    assert pc_result[3:6] == ([], [], {})

    budget = SteinerProblem(graph, [[0, 10]], preprocess=False, budget=5)
    model, x, _y1, _y2, _z, _f, penalty_vars = mm.build_budget_model(
        budget, time_limit=1e-9
    )
    budget_result = mm.run_budget_model(
        model, budget, x, penalty_vars, return_status=True
    )
    assert budget_result[-1] == "incomplete"
    assert math.isinf(budget_result[0])
    assert budget_result[2:5] == (0, [], {})

    mwcsb = BudgetedMaxWeightConnectedSubgraph(
        graph,
        {v: 1 for v in graph},
        {v: 1 for v in graph},
        node_budget=10,
        root=0,
    )
    model, _x, y1, _y2, _z, node_vars = mm.build_mwcsb_model(mwcsb, time_limit=1e-9)
    mwcsb_result = mm.run_mwcsb_model(model, mwcsb, y1, node_vars, return_status=True)
    assert mwcsb_result[-1] == "incomplete"
    assert math.isinf(mwcsb_result[0]) and mwcsb_result[2] == -math.inf
    assert mwcsb_result[3:5] == ([], [])


def test_specialized_infeasible_and_valid_unproven_statuses(monkeypatch):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")

    # Root cost alone exceeds the budget: the full-flow MWCSPB model is
    # genuinely infeasible, rather than merely lacking an incumbent.
    infeasible = BudgetedMaxWeightConnectedSubgraph(
        graph,
        {0: 5, 1: 1, 2: 1},
        {0: 2, 1: 1, 2: 1},
        node_budget=1,
        root=0,
    )
    model, _x, y1, _y2, _z, node_vars = mm.build_mwcsb_model(infeasible, time_limit=5)
    result = mm.run_mwcsb_model(model, infeasible, y1, node_vars, return_status=True)
    assert result[-1] == "infeasible"
    assert result[2] == -math.inf

    # Solve a feasible specialized model, then force only the reported solver
    # termination to kTimeLimit. Values remain a valid incumbent, but gap=0 is
    # forbidden because optimality is no longer reported as proved.
    feasible = BudgetedMaxWeightConnectedSubgraph(
        graph,
        {0: 5, 1: 1, 2: 10},
        {0: 0, 1: 1, 2: 1},
        node_budget=2,
        root=0,
    )
    model, _x, y1, _y2, _z, node_vars = mm.build_mwcsb_model(feasible, time_limit=5)
    model.minimize(
        sum(
            -feasible._mwcs_node_weights.get(v, 0) * node_vars[v]
            for v in feasible.nodes
        )
    )
    assert model.getSolution().value_valid
    monkeypatch.setattr(model, "getModelStatus", lambda: hp.HighsModelStatus.kTimeLimit)
    result = mm.run_mwcsb_model(model, feasible, y1, node_vars, return_status=True)
    assert result[-1] == "incomplete"
    assert math.isinf(result[0])
    assert set(result[4]) == {0, 1, 2}


def test_prize_and_budget_valid_unproven_incumbents(monkeypatch):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")

    pc = PrizeCollectingProblem(graph, [[0, 2]], {0: 2, 1: 1, 2: 2}, penalty_cost=10)
    model, x, _y1, _y2, _z, _f, node_vars, penalty_vars = (
        mm.build_prize_collecting_model(pc, time_limit=5)
    )
    first = mm.run_prize_collecting_model(
        model, pc, x, node_vars, penalty_vars, return_status=True
    )
    assert first[-1] == "optimal"
    monkeypatch.setattr(model, "getModelStatus", lambda: hp.HighsModelStatus.kTimeLimit)
    result = mm.run_prize_collecting_model(
        model, pc, x, node_vars, penalty_vars, return_status=True
    )
    assert result[-1] == "incomplete"
    assert math.isinf(result[0]) and math.isfinite(result[2])
    assert result[3]

    budget = SteinerProblem(graph, [[0, 2]], preprocess=False, budget=2)
    model, x, _y1, _y2, _z, _f, penalty_vars = mm.build_budget_model(
        budget, time_limit=5
    )
    first = mm.run_budget_model(model, budget, x, penalty_vars, return_status=True)
    assert first[-1] == "optimal"
    monkeypatch.setattr(model, "getModelStatus", lambda: hp.HighsModelStatus.kTimeLimit)
    result = mm.run_budget_model(model, budget, x, penalty_vars, return_status=True)
    assert result[-1] == "incomplete"
    assert math.isinf(result[0])
    assert result[2] == 2 and result[3]


def test_sap_tiny_positive_limit_and_valid_primal_fallback(monkeypatch):
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "0")
    graph = nx.DiGraph()
    graph.add_edge("r", "a", weight=1)
    graph.add_edge("a", "t", weight=1)
    view = objects.DirectedSteinerProblem(graph, root="r", terminals=["t"])

    no_incumbent = mm.solve_sap_highs(view, time_limit=1e-9, return_status=True)
    assert no_incumbent[-1] == "incomplete"
    assert math.isinf(no_incumbent[0]) and math.isinf(no_incumbent[2])
    assert no_incumbent[3] == []

    fallback = mm.solve_sap_highs(
        view,
        time_limit=1e-9,
        primal=[("r", "a"), ("a", "t")],
        return_status=True,
    )
    assert fallback[-1] == "incomplete"
    assert math.isinf(fallback[0])
    assert fallback[2] == pytest.approx(2.0)
    assert fallback[3] == [("r", "a"), ("a", "t")]


def test_sap_deadline_after_violated_cut_discards_incumbent(monkeypatch):
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "0")
    graph = nx.DiGraph()
    graph.add_edge("r", "a", weight=1)
    graph.add_edge("a", "t", weight=1)
    view = objects.DirectedSteinerProblem(graph, root="r", terminals=["t"])
    times = iter((0.0, 0.0, 2.0, 2.0))
    monkeypatch.setattr(mm, "time", SimpleNamespace(time=lambda: next(times)))

    result = mm.solve_sap_highs(view, time_limit=1.0, return_status=True)

    assert result[-1] == "incomplete"
    assert math.isinf(result[0]) and math.isinf(result[2])
    assert result[3] == []


def test_public_specialized_paths_reject_missing_incumbents(monkeypatch):
    graph = nx.cycle_graph(30)
    nx.set_edge_attributes(graph, 1, "weight")

    pc = PrizeCollectingProblem(graph, [[0, 15]], {v: 1 for v in graph}, penalty_cost=1)
    with pytest.raises(RuntimeError, match="before finding"):
        pc.get_solution(time_limit=1e-9)

    budget = SteinerProblem(graph, [[0, 15]], preprocess=False, budget=5)
    with pytest.raises(RuntimeError, match="before finding"):
        budget.get_solution(time_limit=1e-9)

    mwcsb = BudgetedMaxWeightConnectedSubgraph(
        graph,
        {v: 1 for v in graph},
        {v: 1 for v in graph},
        node_budget=15,
        root=0,
    )
    with pytest.raises(RuntimeError, match="before finding"):
        mwcsb.get_solution(time_limit=1e-9)

    node_weighted = NodeWeightedSteinerProblem(graph, [[0, 15]], {v: 1 for v in graph})
    with pytest.raises(RuntimeError, match="before finding"):
        node_weighted.get_solution(time_limit=1e-9)


@pytest.mark.parametrize("runner", ["core", "mwcsb"])
def test_gurobi_tiny_positive_limit_is_status_safe(runner):
    _require_gurobi()
    graph = nx.cycle_graph(30)
    nx.set_edge_attributes(graph, 1, "weight")

    if runner == "core":
        problem = SteinerProblem(graph, [[0, 10, 20]], preprocess=False)
        model, x, _y1, y2, z = mm.build_model_gurobi(problem, time_limit=1e-9)
        result = mm.run_model_gurobi(model, problem, x, y2, z, return_status=True)
        objective, edges = result[2], result[3]
    else:
        problem = BudgetedMaxWeightConnectedSubgraph(
            graph,
            {v: 1 for v in graph},
            {v: 1 for v in graph},
            node_budget=15,
            root=0,
        )
        model, _x, y1, _y2, _z, node_vars = mm.build_mwcsb_model_gurobi(
            problem, time_limit=1e-9
        )
        result = mm.run_mwcsb_model_gurobi(
            model, problem, y1, node_vars, return_status=True
        )
        objective, edges = result[2], result[3]

    assert result[-1] in {"incomplete", "optimal"}
    if result[-1] == "incomplete":
        assert math.isinf(result[0])
    if not math.isfinite(objective):
        assert edges == []


def test_gurobi_mwcsb_infeasible_and_valid_unproven(monkeypatch):
    _require_gurobi()
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")

    infeasible = BudgetedMaxWeightConnectedSubgraph(
        graph,
        {0: 5, 1: 1, 2: 1},
        {0: 2, 1: 1, 2: 1},
        node_budget=1,
        root=0,
    )
    model, _x, y1, _y2, _z, node_vars = mm.build_mwcsb_model_gurobi(
        infeasible, time_limit=5
    )
    result = mm.run_mwcsb_model_gurobi(
        model, infeasible, y1, node_vars, return_status=True
    )
    assert result[-1] == "infeasible"
    assert result[2] == -math.inf and result[3:5] == ([], [])

    feasible = BudgetedMaxWeightConnectedSubgraph(
        graph,
        {0: 5, 1: 1, 2: 10},
        {0: 0, 1: 1, 2: 1},
        node_budget=2,
        root=0,
    )
    model, _x, y1, _y2, _z, node_vars = mm.build_mwcsb_model_gurobi(
        feasible, time_limit=5
    )
    monkeypatch.setattr(
        mm,
        "_gurobi_outcome",
        lambda solved_model, _grb: (
            "incomplete",
            solved_model.SolCount > 0,
        ),
    )
    result = mm.run_mwcsb_model_gurobi(
        model, feasible, y1, node_vars, return_status=True
    )
    assert result[-1] == "incomplete"
    assert math.isinf(result[0]) and math.isfinite(result[2])
    assert set(result[4]) == {0, 1, 2}
