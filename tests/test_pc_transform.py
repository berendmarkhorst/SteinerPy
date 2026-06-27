"""Tests for the PCSTP / MWCSP -> SAP transformation and exact/heuristic solving.

The classic forgo-prize PCSTP objective is ``sum_{e in S} c(e) + sum_{v not in S}
p(v)``; the MWCSP maximises ``sum_{v in S} w(v)`` over a connected subgraph. We
verify the opt-in transform path (``pc_transform=True`` and ``exact=False``)
against an **independent brute-force oracle** (enumerate connected vertex subsets)
rather than the penalty ILP, whose semantics differ.

References:
  Rehfeldt & Koch, "On the exact solution of prize-collecting Steiner tree
  problems", ZIB 20-11 (2020): Transformation 2 (PCSTP -> SAP, with
  cost-shifting) and Sec. 2.2 (MWCSP -> PCSTP).
"""

import logging
import random
from itertools import combinations

import networkx as nx
import pytest

from steinerpy import PrizeCollectingProblem, MaxWeightConnectedSubgraph
from steinerpy.pc_transform import (
    is_proper_potential_terminal,
    transform_pcstp_to_sap,
    transform_mwcsp_to_pcstp,
    map_sap_solution_to_pcstp,
    best_trivial_pcstp,
    pcstp_steiner_candidate,
    refine_pcstp_tree,
)

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Brute-force oracles (small instances only)
# ---------------------------------------------------------------------------

def brute_pcstp(graph, prizes, weight="weight"):
    """Optimal classic PCSTP cost: min over connected vertex subsets S of
    ``MST(G[S]) + sum_{v not in S} p(v)``; the empty tree is allowed."""
    nodes = list(graph.nodes())
    total_p = sum(p for p in prizes.values() if p > 0)
    best = total_p  # empty tree forgoes everything
    for r in range(1, len(nodes) + 1):
        for S in combinations(nodes, r):
            sub = graph.subgraph(set(S))
            if len(S) > 1 and not nx.is_connected(sub):
                continue
            mst = nx.minimum_spanning_tree(sub, weight=weight) if len(S) > 1 else nx.Graph()
            cost = sum(d.get(weight, 1) for _, _, d in mst.edges(data=True))
            cost += sum(prizes.get(v, 0) for v in nodes if v not in set(S) and prizes.get(v, 0) > 0)
            best = min(best, cost)
    return best


def brute_mwcsp(graph, weights):
    nodes = list(graph.nodes())
    best = max(weights.get(v, 0) for v in nodes)  # best single vertex
    for r in range(1, len(nodes) + 1):
        for S in combinations(nodes, r):
            sub = graph.subgraph(set(S))
            if len(S) > 1 and not nx.is_connected(sub):
                continue
            best = max(best, sum(weights.get(v, 0) for v in S))
    return best


def random_pcstp(seed, n=7):
    rng = random.Random(seed)
    g = nx.Graph()
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n - 1):
        g.add_edge(perm[i], perm[i + 1], weight=rng.randint(1, 9))
    for _ in range(rng.randint(0, n)):
        u, v = rng.sample(range(n), 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v, weight=rng.randint(1, 9))
    prizes = {v: rng.choice([0, 0, rng.randint(1, 12)]) for v in range(n)}
    return g, prizes


def random_mwcsp(seed, n=7):
    rng = random.Random(seed)
    g = nx.Graph()
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n - 1):
        g.add_edge(perm[i], perm[i + 1], weight=0)
    for _ in range(rng.randint(0, n)):
        u, v = rng.sample(range(n), 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v, weight=0)
    w = {v: rng.randint(-8, 8) for v in range(n)}
    if all(x <= 0 for x in w.values()):
        w[0] = 5
    return g, w


# ---------------------------------------------------------------------------
# Proper / non-proper classification and arc non-negativity
# ---------------------------------------------------------------------------

def test_proper_potential_terminal_classification():
    g = nx.Graph()
    g.add_edge("a", "b", weight=5)
    g.add_edge("b", "c", weight=3)
    # b has min incident cost 3
    assert is_proper_potential_terminal(g, "b", {"b": 4}, "weight") is True   # 4 > 3
    assert is_proper_potential_terminal(g, "b", {"b": 3}, "weight") is False  # 3 == 3
    assert is_proper_potential_terminal(g, "b", {"b": 2}, "weight") is False  # 2 < 3
    assert is_proper_potential_terminal(g, "b", {"b": 0}, "weight") is False  # no prize


def test_transformed_arc_costs_nonnegative():
    for seed in range(25):
        g, prizes = random_pcstp(seed)
        ctx = transform_pcstp_to_sap(g, prizes, "weight")
        for a in ctx.sap_graph.edges():
            assert ctx.sap_graph.edges[a]["weight"] >= -1e-12, (seed, a)


def test_offset_formula():
    # offset == M - sum p(non-proper potential terminals)
    g = nx.Graph()
    g.add_edge(0, 1, weight=10)
    g.add_edge(1, 2, weight=10)
    prizes = {0: 3, 1: 2, 2: 100}  # 0 non-proper (3<10? actually min inc of 0 is 10 -> non-proper),
    ctx = transform_pcstp_to_sap(g, prizes, "weight")
    nonproper_sum = sum(prizes[t] for t in prizes
                        if prizes[t] > 0 and t not in ctx.proper_terminals)
    assert abs(ctx.offset - (ctx.big_m - nonproper_sum)) < 1e-9


# ---------------------------------------------------------------------------
# Exact PCSTP via the transform path == brute-force optimum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(30))
def test_pc_transform_exact_matches_oracle(seed):
    g, prizes = random_pcstp(seed)
    opt = brute_pcstp(g, prizes)
    sol = PrizeCollectingProblem(g.copy(), [[0]], prizes, penalty_cost=0).get_solution(pc_transform=True)
    assert abs(sol.objective - opt) < 1e-6, (seed, sol.objective, opt)
    # Every reported edge exists in the original graph; no SAP artifacts leak out.
    for (u, v) in sol.selected_edges:
        assert g.has_edge(u, v), (seed, u, v)


def test_pc_transform_known_small_instance():
    # Path 0-1-2-3 with a single valuable node 2; cheap to reach.
    g = nx.Graph()
    g.add_edge(0, 1, weight=1)
    g.add_edge(1, 2, weight=1)
    g.add_edge(2, 3, weight=1)
    prizes = {0: 1, 1: 100, 2: 10, 3: 40}
    opt = brute_pcstp(g, prizes)
    sol = PrizeCollectingProblem(g.copy(), [[2]], prizes, penalty_cost=0).get_solution(pc_transform=True)
    assert abs(sol.objective - opt) < 1e-6


def test_trivial_single_vertex_optimum():
    # A node whose prize dwarfs all reach costs but no profitable connection:
    # the optimum is the single vertex, which the SAP cannot represent directly.
    g = nx.Graph()
    g.add_edge("hub", "x", weight=100)
    g.add_edge("hub", "y", weight=100)
    prizes = {"hub": 0, "x": 50, "y": 1}
    opt = brute_pcstp(g, prizes)  # connect nothing extra: keep {x}; forgo y
    sol = PrizeCollectingProblem(g.copy(), [["x"]], prizes, penalty_cost=0).get_solution(pc_transform=True)
    assert abs(sol.objective - opt) < 1e-6


# ---------------------------------------------------------------------------
# Heuristic mode: valid optimality gap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(30))
def test_pc_transform_heuristic_valid_gap(seed):
    g, prizes = random_pcstp(seed)
    opt = brute_pcstp(g, prizes)
    sol = PrizeCollectingProblem(g.copy(), [[0]], prizes, penalty_cost=0).get_solution(exact=False)
    assert sol.gap >= -1e-9                      # gap is non-negative
    assert sol.objective >= opt - 1e-6           # primal is an upper bound on the optimum (min problem)
    if abs(sol.gap) < 1e-9:                       # gap 0 certifies optimality
        assert abs(sol.objective - opt) < 1e-6


# ---------------------------------------------------------------------------
# MWCSP exact mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(30))
def test_mwcsp_transform_exact_matches_oracle(seed):
    g, w = random_mwcsp(seed)
    opt = brute_mwcsp(g, w)
    sol = MaxWeightConnectedSubgraph(g.copy(), w).get_solution(pc_transform=True)
    assert abs(sol.objective - opt) < 1e-6, (seed, sol.objective, opt, sol.selected_nodes)


def test_mwcsp_transform_excludes_negative_node():
    g = nx.Graph()
    g.add_edge("A", "B", weight=0)
    g.add_edge("B", "C", weight=0)
    g.add_edge("C", "D", weight=0)
    w = {"A": 2, "B": 3, "C": 1, "D": -10}
    sol = MaxWeightConnectedSubgraph(g.copy(), w).get_solution(pc_transform=True)
    assert "D" not in sol.selected_nodes
    assert abs(sol.objective - 6.0) < 1e-6       # A+B+C


# ---------------------------------------------------------------------------
# best_trivial_pcstp helper
# ---------------------------------------------------------------------------

def test_best_trivial_pcstp():
    nodes, obj = best_trivial_pcstp({0: 5, 1: 3, 2: 0})
    assert nodes == [0] and abs(obj - 3) < 1e-9   # keep best node 0, forgo 1 (3)
    nodes, obj = best_trivial_pcstp({0: 0, 1: 0})
    assert nodes == [] and abs(obj) < 1e-9        # empty tree


def test_best_trivial_pcstp_empty_prizes():
    # no prizes at all -> empty tree, zero objective
    assert best_trivial_pcstp({}) == ([], 0.0)


# ---------------------------------------------------------------------------
# transform_mwcsp_to_pcstp edge cases
# ---------------------------------------------------------------------------

def test_transform_mwcsp_requires_node_weights():
    with pytest.raises(ValueError):
        transform_mwcsp_to_pcstp(nx.Graph(), {})


# ---------------------------------------------------------------------------
# pcstp_steiner_candidate
# ---------------------------------------------------------------------------

def test_pcstp_steiner_candidate_spans_prize_nodes():
    g = nx.Graph()
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "C", weight=1)
    edges, nodes = pcstp_steiner_candidate(g, {"A": 5, "C": 5})
    # connecting A and C must route through B
    assert set(nodes) == {"A", "B", "C"}
    assert {frozenset(e) for e in edges} == {frozenset(("A", "B")),
                                             frozenset(("B", "C"))}


def test_pcstp_steiner_candidate_fewer_than_two_prizes():
    g = nx.Graph()
    g.add_edge("A", "B", weight=1)
    assert pcstp_steiner_candidate(g, {"A": 5}) is None
    assert pcstp_steiner_candidate(g, {}) is None


def test_pcstp_steiner_candidate_rejects_negative_edges():
    # Mehlhorn (shortest paths) needs nonnegative costs -> bail out
    g = nx.Graph()
    g.add_edge("A", "B", weight=-1)
    assert pcstp_steiner_candidate(g, {"A": 5, "B": 5}) is None


def test_pcstp_steiner_candidate_swallows_steiner_tree_errors(monkeypatch):
    # The networkx call is wrapped defensively; a failure yields None.
    import networkx.algorithms.approximation as approx

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(approx, "steiner_tree", boom)
    g = nx.Graph()
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "C", weight=1)
    assert pcstp_steiner_candidate(g, {"A": 5, "C": 5}) is None


# ---------------------------------------------------------------------------
# refine_pcstp_tree
# ---------------------------------------------------------------------------

def test_refine_pcstp_tree_no_profitable_vertex():
    # No positive prizes and empty seed tree -> empty tree, objective == sum(prizes).
    g = nx.Graph()
    g.add_edge("A", "B", weight=1)
    edges, nodes, obj = refine_pcstp_tree(g, [], [], {"A": 0, "B": 0})
    assert edges == [] and nodes == [] and abs(obj) < 1e-9


def test_refine_pcstp_tree_inserts_profitable_prize():
    # Start from a single-node tree; the profitable prize at C (5 > path cost 2)
    # should be inserted and the objective should drop accordingly.
    g = nx.Graph()
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "C", weight=1)
    edges, nodes, obj = refine_pcstp_tree(g, [], ["A"], {"A": 5, "C": 5})
    assert set(nodes) == {"A", "B", "C"}
    # cost = 2 edges, all prize collected (total 10) -> 2 + (10 - 10) = 2
    assert abs(obj - 2.0) < 1e-9


def test_refine_pcstp_tree_prunes_unprofitable_leaf():
    # Tree A-B-C where C's prize (1) is below its connection cost (5): prune C.
    g = nx.Graph()
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "C", weight=5)
    edges, nodes, obj = refine_pcstp_tree(
        g, [("A", "B"), ("B", "C")], ["A", "B", "C"], {"A": 10, "B": 10, "C": 1}
    )
    assert "C" not in nodes
    # cost 1 (A-B) + forgone prize C (1) = 2
    assert abs(obj - 2.0) < 1e-9
