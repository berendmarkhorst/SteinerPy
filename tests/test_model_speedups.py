"""Tests for the model-building/separation speedups.

Covers:
* nested cuts (Koch & Martin 1998) in the separation loop preserve the optimum
  and can be disabled via ``STEINERPY_NESTED_CUTS``;
* the flow variables of the flow-based models are continuous (integrality of
  the flow follows from the integral arc/connection variables), so they no
  longer inflate the integer search space.
"""

import random

import highspy as hp
import networkx as nx
import pytest

from steinerpy import SteinerProblem
from steinerpy.mathematical_model import (
    _HighsCutPool,
    build_budget_model,
    build_prize_collecting_model,
    _cut_purge_age,
    _lp_cut_rounds,
    _nested_cut_rounds,
    solve_sap_highs,
)


def _random_instance(n, m, k, seed):
    rng = random.Random(seed)
    while True:
        G = nx.gnm_random_graph(n, m, seed=seed)
        if nx.is_connected(G):
            break
        seed += 10007
    for u, v in G.edges():
        G[u][v]["weight"] = rng.randint(1, 10)
    terminals = rng.sample(sorted(G.nodes()), k)
    return G, terminals


def _n_continuous_columns(model):
    lp = model.getLp()
    return sum(1 for i in lp.integrality_ if i == hp.HighsVarType.kContinuous)


def test_nested_cut_rounds_env(monkeypatch):
    monkeypatch.setenv("STEINERPY_NESTED_CUTS", "0")
    assert _nested_cut_rounds() == 0
    monkeypatch.setenv("STEINERPY_NESTED_CUTS", "7")
    assert _nested_cut_rounds() == 7
    monkeypatch.setenv("STEINERPY_NESTED_CUTS", "not-a-number")
    assert _nested_cut_rounds() == 1  # default
    monkeypatch.delenv("STEINERPY_NESTED_CUTS")
    assert _nested_cut_rounds() == 1


def test_lp_cut_rounds_env(monkeypatch):
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "0")
    assert _lp_cut_rounds() == 0
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "12")
    assert _lp_cut_rounds() == 12
    monkeypatch.delenv("STEINERPY_LP_CUT_ROUNDS")
    assert _lp_cut_rounds() == 50  # default


def test_cut_purge_age_env(monkeypatch):
    monkeypatch.setenv("STEINERPY_CUT_PURGE_AGE", "0")
    assert _cut_purge_age() == 0
    monkeypatch.setenv("STEINERPY_CUT_PURGE_AGE", "3")
    assert _cut_purge_age() == 3
    monkeypatch.setenv("STEINERPY_CUT_PURGE_AGE", "not-a-number")
    assert _cut_purge_age() == 0
    monkeypatch.delenv("STEINERPY_CUT_PURGE_AGE")
    assert _cut_purge_age() == 0


def test_cut_pool_purges_reindexes_and_reintroduces():
    model = hp.Highs()
    model.setOptionValue("output_flag", False)
    x = model.addVariable(0, 1)
    model.addConstr(x >= 0)  # permanent structural row
    pool = _HighsCutPool(model, purge_age=1)
    assert pool.add(("slack",), x, 0.0)
    assert pool.add(("binding",), x, 1.0)
    assert not pool.add(("binding",), x, 1.0)

    model.minimize(-x)
    assert pool.age_and_purge() == 1
    assert model.getNumRow() == 2
    assert pool._records[("binding",)].row_id == 1
    assert pool.stats["cuts_purged"] == 1

    assert pool.add(("slack",), x, 0.0)
    assert pool.stats["cuts_reintroduced"] == 1


def test_cut_purge_preserves_sap_optimum(monkeypatch):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("r", "a", 1), ("a", "t", 1), ("r", "t", 5)])
    view = SteinerProblem(graph, [["r", "t"]], preprocess=False)

    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "3")
    monkeypatch.setenv("STEINERPY_CUT_PURGE_AGE", "1")
    gap, _runtime, objective, selected = solve_sap_highs(view, time_limit=30)

    assert gap == pytest.approx(0.0, abs=1e-6)
    assert objective == pytest.approx(2.0)
    assert set(selected) == {("r", "a"), ("a", "t")}
    assert view.cut_stats["purge_age"] == 1
    assert view.cut_stats["cuts_added"] > 0


def test_cut_purge_preserves_forest_optimum(monkeypatch):
    graph, terminals = _random_instance(14, 28, 6, seed=0)
    groups = [terminals[:3], terminals[3:]]
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "0")
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "5")

    monkeypatch.setenv("STEINERPY_CUT_PURGE_AGE", "0")
    baseline_problem = SteinerProblem(graph, groups, preprocess=False)
    baseline = baseline_problem.get_solution(time_limit=60)

    monkeypatch.setenv("STEINERPY_CUT_PURGE_AGE", "3")
    purged_problem = SteinerProblem(graph, groups, preprocess=False)
    purged = purged_problem.get_solution(time_limit=60)

    assert baseline.gap == pytest.approx(0.0, abs=1e-6)
    assert purged.gap == pytest.approx(0.0, abs=1e-6)
    assert purged.objective == pytest.approx(baseline.objective)
    assert purged_problem.cut_stats["purge_age"] == 3
    assert purged_problem.cut_stats["cuts_added"] > 0
    assert purged_problem.cut_stats["cuts_purged"] > 0


@pytest.mark.parametrize("groups", [1, 2])
def test_lp_cut_phase_preserves_optimum(monkeypatch, groups):
    G, terminals = _random_instance(40, 100, 4 * groups, seed=17 + groups)
    tg = [terminals[i * 4 : (i + 1) * 4] for i in range(groups)]

    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "0")
    base = SteinerProblem(G, tg).get_solution(time_limit=120)
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", "50")
    lp_first = SteinerProblem(G, tg).get_solution(time_limit=120)

    assert base.gap == pytest.approx(0.0, abs=1e-6)
    assert lp_first.gap == pytest.approx(0.0, abs=1e-6)
    assert lp_first.objective == pytest.approx(base.objective)


def test_lp_cut_phase_with_dual_ascent_warm_start():
    # Exercises the reapply_start path: dual ascent supplies a MIP warm start
    # that run_model must re-apply after the LP phase.
    G, terminals = _random_instance(40, 100, 5, seed=23)
    sol = SteinerProblem(G, [terminals], dual_ascent=True).get_solution(time_limit=120)
    ref = SteinerProblem(G, [terminals]).get_solution(time_limit=120)
    assert sol.gap == pytest.approx(0.0, abs=1e-6)
    assert sol.objective == pytest.approx(ref.objective)


@pytest.mark.parametrize("groups", [1, 2])
def test_nested_cuts_preserve_optimum(monkeypatch, groups):
    G, terminals = _random_instance(40, 100, 4 * groups, seed=7 + groups)
    tg = [terminals[i * 4 : (i + 1) * 4] for i in range(groups)]

    monkeypatch.setenv("STEINERPY_NESTED_CUTS", "0")
    base = SteinerProblem(G, tg).get_solution(time_limit=120)
    monkeypatch.setenv("STEINERPY_NESTED_CUTS", "3")
    nested = SteinerProblem(G, tg).get_solution(time_limit=120)

    assert base.gap == pytest.approx(0.0, abs=1e-6)
    assert nested.gap == pytest.approx(0.0, abs=1e-6)
    assert nested.objective == pytest.approx(base.objective)


def test_budget_flow_variables_are_continuous():
    G, terminals = _random_instance(20, 45, 5, seed=3)
    prob = SteinerProblem(G, [terminals], budget=10, preprocess=False)
    model, x, y1, y2, z, f, penalty_vars = build_budget_model(prob, time_limit=10)
    # Exactly the flow columns are continuous; everything else stays integer.
    assert len(f) > 0
    assert _n_continuous_columns(model) == len(f)


def test_prize_collecting_flow_variables_are_continuous():
    G, terminals = _random_instance(20, 45, 5, seed=4)
    from steinerpy import PrizeCollectingProblem

    prob = PrizeCollectingProblem(
        G, [terminals], node_prizes={t: 5 for t in terminals}, penalty_cost=10
    )
    model, x, y1, y2, z, f, node_vars, penalty_vars = build_prize_collecting_model(
        prob, time_limit=10
    )
    assert len(f) > 0
    assert _n_continuous_columns(model) == len(f)
