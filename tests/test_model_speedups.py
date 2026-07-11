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
    build_budget_model,
    build_prize_collecting_model,
    _nested_cut_rounds,
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
    assert _nested_cut_rounds() == 3  # default
    monkeypatch.delenv("STEINERPY_NESTED_CUTS")
    assert _nested_cut_rounds() == 3


@pytest.mark.parametrize("groups", [1, 2])
def test_nested_cuts_preserve_optimum(monkeypatch, groups):
    G, terminals = _random_instance(40, 100, 4 * groups, seed=7 + groups)
    tg = [terminals[i * 4:(i + 1) * 4] for i in range(groups)]

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
