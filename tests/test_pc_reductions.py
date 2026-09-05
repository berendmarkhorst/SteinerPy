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

from steinerpy import MaxWeightConnectedSubgraph, PrizeCollectingProblem
from steinerpy.pc_reductions import (
    prize_constrained_distance_deletions,
    reduce_pcstp_graph,
    terminal_region_bound_deletions,
    terminal_regions_decomposition,
)
from tests.test_pc_transform import brute_mwcsp, brute_pcstp, random_pcstp


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


def test_equal_detour_is_kept_by_batched_reducer():
    # Corollary 7 permits one equality deletion, but deleting several mutually
    # substitutable equality edges in a batch can lose every optimum. The
    # batched implementation therefore uses Theorem 6's strict condition.
    g = nx.Graph()
    g.add_edge("a", "b", weight=5)
    g.add_edge("b", "c", weight=5)
    g.add_edge("a", "c", weight=10)  # detour a-b-c == 10 == c(a,c)
    prizes = {"a": 0, "b": 0, "c": 0}
    dels = prize_constrained_distance_deletions(g, prizes, "weight")
    assert ("a", "c") not in dels and ("c", "a") not in dels


def test_prize_endpoint_and_mutually_substitutable_edges_are_kept():
    # Regression seed 69 from the independent oracle sweep. The old PCD code
    # discounted the destination endpoint's prize and batch-deleted both tied
    # alternatives {4, 6}/{1, 6}, increasing the optimum from 22 to 26.
    g, prizes = random_pcstp(69)
    assert brute_pcstp(g, prizes) == 22
    reduced = reduce_pcstp_graph(g, prizes)
    assert brute_pcstp(reduced, prizes) == 22


def test_reduce_preserves_nodes_and_prizes():
    for seed in range(20):
        g, prizes = random_pcstp(seed)
        before_prizes = dict(prizes)
        reduced = reduce_pcstp_graph(g, prizes)
        assert set(reduced.nodes()) == set(g.nodes()), seed
        # The prize dict is the caller's; reduction must not touch it.
        assert prizes == before_prizes
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


def test_terminal_regions_form_valid_partition():
    g, prizes = random_pcstp(6)
    decomposition = terminal_regions_decomposition(g, prizes)
    assigned = set(decomposition.unassigned)
    for terminal, region in decomposition.regions.items():
        assert terminal in region
        assert nx.is_connected(g.subgraph(region))
        assert region.isdisjoint(assigned)
        assert region & decomposition.proper_terminals == {terminal}
        assigned.update(region)
    assert assigned == set(g)
    assert decomposition.nonproper_terminals <= decomposition.unassigned


def test_terminal_region_bound_deletes_expensive_zero_prize_leaf():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "x", 1), ("x", "b", 1), ("x", "leaf", 100)])
    prizes = {"a": 10, "b": 10, "x": 0, "leaf": 0}

    edge_deletions, node_deletions, protected, upper_bound = (
        terminal_region_bound_deletions(
            graph,
            prizes,
            delete_edges=True,
            delete_nodes=True,
        )
    )

    assert frozenset(("x", "leaf")) in {frozenset(edge) for edge in edge_deletions}
    assert "leaf" in node_deletions
    assert not protected
    assert upper_bound == pytest.approx(2.0)


def test_terminal_region_bound_reports_but_keeps_prize_node():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [("a", "x", 1), ("x", "b", 1), ("x", "prize_leaf", 100)]
    )
    prizes = {"a": 10, "b": 10, "x": 0, "prize_leaf": 1}
    reduced, stats = reduce_pcstp_graph(
        graph,
        prizes,
        bound_edges=True,
        bound_nodes=True,
        return_stats=True,
    )

    assert "prize_leaf" in reduced
    assert stats["protected_prize_nodes"] == 1


@pytest.mark.parametrize("seed", range(100))
def test_terminal_region_stacks_preserve_optimum(seed):
    g, prizes = random_pcstp(seed)
    opt = brute_pcstp(g, prizes)
    before_prizes = dict(prizes)
    for bound_edges, bound_nodes in ((True, False), (True, True)):
        reduced, stats = reduce_pcstp_graph(
            g,
            prizes,
            bound_edges=bound_edges,
            bound_nodes=bound_nodes,
            return_stats=True,
        )
        assert abs(brute_pcstp(reduced, prizes) - opt) < 1e-6
        assert prizes == before_prizes
        assert all(prizes.get(v, 0) <= 0 for v in set(g) - set(reduced))
        assert stats["nodes_removed"] == len(set(g) - set(reduced))
        assert stats["preprocessing_time"] >= 0


@pytest.mark.parametrize("level", [True, "pcd", "pcd+trd", "pcd+trd+nodes"])
@pytest.mark.parametrize("seed", range(20))
def test_public_pc_reduction_levels_match_oracle(level, seed):
    g, prizes = random_pcstp(seed)
    solution = PrizeCollectingProblem(
        g.copy(),
        [[0]],
        prizes,
        penalty_cost=0,
        pc_reduce=level,
    ).get_solution(pc_transform=True)
    assert abs(solution.objective - brute_pcstp(g, prizes)) < 1e-6


def test_public_pc_reduction_level_validation():
    g, prizes = random_pcstp(0)
    with pytest.raises(ValueError, match="pc_reduce must be"):
        PrizeCollectingProblem(g, [[0]], prizes, penalty_cost=0, pc_reduce="all")


def test_public_reduction_stats_and_prize_preservation():
    g, prizes = random_pcstp(6)
    prize_nodes = {v for v, prize in prizes.items() if prize > 0}
    problem = PrizeCollectingProblem(
        g,
        [[0]],
        prizes,
        penalty_cost=0,
        pc_reduce="pcd+trd+nodes",
    )
    assert prize_nodes <= set(problem.graph)
    assert 0 in problem.graph  # caller-supplied terminal is protected
    assert problem.pc_reduction_stats["upper_bound"] is not None


@pytest.mark.parametrize("level", [True, "pcd", "pcd+trd", "pcd+trd+nodes"])
@pytest.mark.parametrize("seed", range(15))
def test_public_mwcsp_reduction_levels_match_oracle(level, seed):
    rng = random.Random(seed)
    graph = nx.gnm_random_graph(7, 10, seed=seed)
    if not nx.is_connected(graph):
        graph = nx.complete_graph(7)
    for u, v in graph.edges():
        graph[u][v]["weight"] = 0
    weights = {v: rng.randint(-7, 9) for v in graph}
    if not any(value > 0 for value in weights.values()):
        weights[0] = 1

    solution = MaxWeightConnectedSubgraph(graph, weights, pc_reduce=level).get_solution(
        pc_transform=True
    )
    assert solution.gap == pytest.approx(0.0, abs=1e-7)
    assert solution.objective == pytest.approx(brute_mwcsp(graph, weights))
