"""Tests for the Dreyfus–Wagner few-terminal dynamic program."""
import math
import random

import networkx as nx
import pytest

from steinerpy import GroupSteinerProblem, RectilinearSteinerProblem, SteinerProblem
from steinerpy._fastgraph import HAS_SCIPY
from steinerpy.dreyfus_wagner import dw_max_terminals

if HAS_SCIPY:
    from steinerpy.dreyfus_wagner import dreyfus_wagner

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="requires scipy")


def _random_instance(n, m, k, seed, wmax=10, wmin=1):
    rng = random.Random(seed)
    while True:
        G = nx.gnm_random_graph(n, m, seed=seed)
        if nx.is_connected(G):
            break
        seed += 10007
    for u, v in G.edges():
        G[u][v]["weight"] = rng.randint(wmin, wmax)
    return G, rng.sample(sorted(G.nodes()), k)


def _edge_cost(G, edges, weight="weight"):
    return sum(G[u][v].get(weight, 1) for u, v in edges)


def _spans(edges, terminals):
    H = nx.Graph(edges)
    H.add_nodes_from(terminals)
    return any(set(terminals) <= c for c in nx.connected_components(H))


def test_dw_max_terminals_env(monkeypatch):
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "0")
    assert dw_max_terminals() == 0
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "13")
    assert dw_max_terminals() == 13
    monkeypatch.delenv("STEINERPY_DW_MAX_TERMINALS")
    assert dw_max_terminals() == 10  # default


def test_trivial_cases():
    G = nx.path_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 2
    assert dreyfus_wagner(G, []) == (0.0, [])
    assert dreyfus_wagner(G, [2]) == (0.0, [])
    # two terminals: shortest path
    cost, edges = dreyfus_wagner(G, [0, 3])
    assert cost == 6
    assert sorted(tuple(sorted(e)) for e in edges) == [(0, 1), (1, 2), (2, 3)]


def test_star_with_steiner_point():
    # Optimal tree must use the non-terminal hub.
    G = nx.star_graph(4)  # hub 0, leaves 1..4
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    G.add_edge(1, 2, weight=5)
    cost, edges = dreyfus_wagner(G, [1, 2, 3])
    assert cost == 3
    assert _spans(edges, [1, 2, 3])


def test_zero_weight_edges():
    # Zero-cost connector edges (as used by the group-Steiner transform).
    G = nx.path_graph(3)
    G[0][1]["weight"] = 0
    G[1][2]["weight"] = 0
    cost, edges = dreyfus_wagner(G, [0, 2])
    assert cost == 0.0
    assert _spans(edges, [0, 2])


def test_disconnected_terminals():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(2, 3, weight=1)
    cost, edges = dreyfus_wagner(G, [0, 3])
    assert math.isinf(cost) and edges == []


@pytest.mark.parametrize("seed", range(8))
def test_matches_ilp_on_random_instances(monkeypatch, seed):
    n = 25 + 5 * seed
    G, terminals = _random_instance(n, 3 * n, 3 + seed % 5, seed=seed)

    cost, edges = dreyfus_wagner(G, terminals)
    assert _spans(edges, terminals)
    assert cost == pytest.approx(_edge_cost(G, edges))

    # ILP reference with the DP auto-selection disabled.
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "0")
    ref = SteinerProblem(G, [terminals]).get_solution(time_limit=120)
    assert ref.gap == pytest.approx(0.0, abs=1e-6)
    assert cost == pytest.approx(ref.objective)


def test_auto_selected_in_get_solution_with_preprocessing():
    G, terminals = _random_instance(60, 150, 6, seed=101)
    # With the full default reduction stack this instance is solved outright in
    # preprocessing; use the structural reductions only so the DP is the one
    # doing the solving (still exercising the degree-2 back-mapping).
    prob = SteinerProblem(G, [terminals], heavy=False, contract_terminals=False,
                          bound_based=False)
    assert prob._dw_eligible()
    sol = prob.get_solution(time_limit=120)
    assert sol.gap == 0.0
    # The mapped-back solution must span the terminals in the ORIGINAL graph
    # at the reported cost.
    assert _spans(sol.edges, terminals)
    assert sol.objective == pytest.approx(_edge_cost(G, sol.edges))

    # And the full default stack (which may pre-solve the instance entirely)
    # must agree on the objective.
    full = SteinerProblem(G, [terminals]).get_solution(time_limit=120)
    assert full.objective == pytest.approx(sol.objective)
    assert _spans(full.edges, terminals)


def test_not_selected_above_terminal_cap(monkeypatch):
    G, terminals = _random_instance(30, 80, 6, seed=7)
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "5")
    assert not SteinerProblem(G, [terminals], preprocess=False)._dw_eligible()
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "6")
    assert SteinerProblem(G, [terminals], preprocess=False)._dw_eligible()


def test_not_eligible_for_forest_or_modifiers():
    G, terminals = _random_instance(30, 80, 6, seed=8)
    half = len(terminals) // 2
    assert not SteinerProblem(
        G, [terminals[:half], terminals[half:]], preprocess=False
    )._dw_eligible()
    assert not SteinerProblem(
        G, [terminals], preprocess=False, max_degree=3
    )._dw_eligible()
    assert not SteinerProblem(
        G, [terminals], preprocess=False, budget=10
    )._dw_eligible()


def test_group_steiner_via_dw():
    # Group-Steiner transform introduces zero-cost super-terminal connectors;
    # the DP must return the same objective as the ILP.
    G, nodes = _random_instance(30, 80, 6, seed=9)
    groups = [nodes[:3], nodes[3:]]
    sol = GroupSteinerProblem(G, groups).get_solution(time_limit=120)
    import os
    os.environ["STEINERPY_DW_MAX_TERMINALS"] = "0"
    try:
        ref = GroupSteinerProblem(G, groups).get_solution(time_limit=120)
    finally:
        del os.environ["STEINERPY_DW_MAX_TERMINALS"]
    assert sol.objective == pytest.approx(ref.objective)


def test_rectilinear_via_dw():
    points = [(0, 0), (4, 0), (0, 4), (3, 3)]
    sol = RectilinearSteinerProblem(points).get_solution(time_limit=120)
    import os
    os.environ["STEINERPY_DW_MAX_TERMINALS"] = "0"
    try:
        ref = RectilinearSteinerProblem(points).get_solution(time_limit=120)
    finally:
        del os.environ["STEINERPY_DW_MAX_TERMINALS"]
    assert sol.objective == pytest.approx(ref.objective)
