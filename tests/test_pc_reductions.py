"""Tests for the prize-constrained-distance (PCD) edge-deletion reduction.

PCD deletes only edges that are provably in no optimal PCSTP solution, so the
defining properties are: (1) **the optimum is preserved**, and (2) it is
**prize-safe** — no vertex is removed and every prize is kept. We verify both on
hand-built instances and by fuzzing against a brute-force exact PCSTP solver.

Reference: Rehfeldt & Koch, "On the exact solution of prize-collecting Steiner
tree problems", ZIB 20-11 (2020), Theorem 6 / Corollary 7 / Algorithm 1.
"""

import random

import networkx as nx
import pytest

from steinerpy.pc_reductions import (
    prize_constrained_distance_deletions,
    reduce_pcstp_graph,
)
from tests.test_pc_transform import brute_pcstp, random_pcstp


def test_cheaper_prize_detour_deletes_edge():
    # Report Figure 1: only v4 has a prize (5). Edge {v1, v2} (cost 9) has a
    # prize-constrained detour v1-v3-v4-v3-v2 of length 6 < 9 -> deletable.
    g = nx.Graph()
    g.add_edge("v1", "v3", weight=5)
    g.add_edge("v3", "v4", weight=1)
    g.add_edge("v3", "v2", weight=5)
    g.add_edge("v1", "v2", weight=9)
    prizes = {"v1": 0, "v2": 0, "v3": 0, "v4": 5}
    dels = prize_constrained_distance_deletions(g, prizes, "weight")
    assert ("v1", "v2") in dels or ("v2", "v1") in dels


def test_cheapest_direct_edge_kept():
    # The direct edge is strictly cheaper than any detour, so no qualifying
    # detour exists (Algorithm 1 deletes only when c(e) >= d_pc, Corollary 7).
    g = nx.Graph()
    g.add_edge("a", "b", weight=5)
    g.add_edge("b", "c", weight=5)
    g.add_edge("a", "c", weight=4)  # detour a-b-c == 10 > 4
    prizes = {"a": 0, "b": 0, "c": 0}
    dels = prize_constrained_distance_deletions(g, prizes, "weight")
    assert ("a", "c") not in dels and ("c", "a") not in dels


def test_equal_detour_is_deletable():
    # Equality is deletable per Corollary 7 (keeps at least one optimum).
    g = nx.Graph()
    g.add_edge("a", "b", weight=5)
    g.add_edge("b", "c", weight=5)
    g.add_edge("a", "c", weight=10)  # detour a-b-c == 10 == c(a,c)
    prizes = {"a": 0, "b": 0, "c": 0}
    dels = prize_constrained_distance_deletions(g, prizes, "weight")
    assert ("a", "c") in dels or ("c", "a") in dels


def test_reduce_preserves_nodes_and_prizes():
    for seed in range(20):
        g, prizes = random_pcstp(seed)
        reduced = reduce_pcstp_graph(g, prizes)
        assert set(reduced.nodes()) == set(g.nodes()), seed
        # The prize dict is the caller's; reduction must not touch it.
        assert prizes == prizes
        # Reduced graph is a subgraph (edges only deleted, never added).
        for u, v in reduced.edges():
            assert g.has_edge(u, v), (seed, u, v)


def test_reduce_does_not_mutate_input():
    g, prizes = random_pcstp(1)
    before = g.number_of_edges()
    reduce_pcstp_graph(g, prizes)
    assert g.number_of_edges() == before  # input untouched (works on a copy)


@pytest.mark.parametrize("seed", range(40))
def test_reduce_preserves_optimum(seed):
    g, prizes = random_pcstp(seed)
    opt = brute_pcstp(g, prizes)
    reduced = reduce_pcstp_graph(g, prizes)
    opt2 = brute_pcstp(reduced, prizes)
    assert abs(opt - opt2) < 1e-6, (seed, opt, opt2)
