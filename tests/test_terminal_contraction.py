"""Tests for the terminal-contraction (fixed-edge) reductions.

Covers the infrastructure (fixed-cost channel, terminal remapping, solution
back-mapping) and the two contraction tests:

* degree-1 terminal contraction — the sole incident edge is in **every**
  feasible solution;
* adjacent-terminal contraction — a terminal-terminal edge that is cheapest at
  one endpoint is in **at least one** optimal solution.

The randomized sweeps compare full-default solves (contraction on) against
``preprocess=False`` solves and a brute-force oracle, and check that the mapped
solution spans the terminals in the *original* graph at exactly the reported
objective.
"""
import itertools
import math
import random

import networkx as nx
import pytest

from steinerpy import SteinerProblem
from steinerpy.graph_reducer import preprocess_graph


def _random_instance(n, m, k, seed, wmax=10):
    rng = random.Random(seed)
    while True:
        G = nx.gnm_random_graph(n, m, seed=seed)
        if nx.is_connected(G):
            break
        seed += 10007
    for u, v in G.edges():
        G[u][v]["weight"] = rng.randint(1, wmax)
    return G, rng.sample(sorted(G.nodes()), k)


def _edge_cost(G, edges, weight="weight"):
    return sum(G[u][v].get(weight, 1) for u, v in edges)


def _spans(edges, terminals):
    H = nx.Graph(edges)
    H.add_nodes_from(terminals)
    return any(set(terminals) <= c for c in nx.connected_components(H))


def _brute_force(G, terminals, weight="weight"):
    """Exact optimum by enumerating edge subsets (tiny graphs only)."""
    best = math.inf
    E = list(G.edges())
    for r in range(len(E) + 1):
        for sub in itertools.combinations(E, r):
            H = nx.Graph()
            H.add_edges_from(sub)
            H.add_nodes_from(terminals)
            if any(set(terminals) <= c for c in nx.connected_components(H)):
                best = min(best, _edge_cost(G, sub, weight))
        if best < math.inf and r >= len(terminals) + 2:
            break  # small optimization: no optimal tree needs many more edges
    return best


def test_degree_one_terminal_chain_fully_contracts():
    G = nx.path_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 2
    reduced, tracker = preprocess_graph(G, [[0, 3]], contract=True)
    assert reduced.number_of_edges() == 0
    assert reduced.number_of_nodes() == 1
    assert tracker.fixed_cost == 6
    assert len(tracker.fixed_edges) == 3
    assert tracker.resolve_terminal(0) == tracker.resolve_terminal(3)


def test_adjacent_terminal_contraction():
    # Terminals 0 and 1 joined by the cheapest edge at 0 -> contract; the
    # third terminal keeps the problem alive.
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=5)
    G.add_edge(1, 2, weight=5)
    G.add_edge(2, 3, weight=1)
    reduced, tracker = preprocess_graph(G, [[0, 1, 3]], contract=True)
    assert tracker.fixed_cost > 0
    merged = set(tracker.terminal_merges)
    assert merged  # at least one terminal was merged away


def test_contract_terminals_flag_disables():
    G = nx.path_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 2
    p = SteinerProblem(G, [[0, 3]], contract_terminals=False)
    assert p.reduction_tracker.fixed_cost == 0
    assert p.reduction_tracker.terminal_merges == {}
    sol = p.get_solution(time_limit=30)
    assert sol.objective == pytest.approx(6)


def test_forest_multi_group_untouched():
    G, terminals = _random_instance(30, 80, 6, seed=5)
    half = len(terminals) // 2
    groups = [terminals[:half], terminals[half:]]
    p = SteinerProblem(G, groups)
    assert p.reduction_tracker.fixed_cost == 0  # contraction is tree-only
    sol = p.get_solution(time_limit=120)
    ref = SteinerProblem(G, groups, preprocess=False).get_solution(time_limit=120)
    assert sol.objective == pytest.approx(ref.objective)


@pytest.mark.parametrize("seed", range(10))
def test_matches_unreduced_ilp_on_random_instances(seed):
    n = 20 + 4 * seed
    G, terminals = _random_instance(n, int(2.5 * n), 3 + seed % 5, seed=seed)
    sol = SteinerProblem(G, [terminals]).get_solution(time_limit=120)
    ref = SteinerProblem(G, [terminals], preprocess=False).get_solution(time_limit=120)
    assert sol.gap == pytest.approx(0.0, abs=1e-6)
    assert sol.objective == pytest.approx(ref.objective)
    # The mapped-back solution must span the terminals in the ORIGINAL graph
    # at exactly the reported objective.
    assert _spans(sol.edges, terminals)
    assert _edge_cost(G, sol.edges) == pytest.approx(sol.objective)


@pytest.mark.parametrize("seed", range(15))
def test_matches_brute_force_on_tiny_instances(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(5, 8)
    G = nx.gnp_random_graph(n, rng.uniform(0.4, 0.8), seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.randint(1, 6)
    k = rng.randint(2, min(4, n))
    terminals = rng.sample(sorted(G.nodes()), k)
    if not all(nx.has_path(G, terminals[0], t) for t in terminals[1:]
               if terminals[0] in G and t in G):
        pytest.skip("terminals disconnected")
    want = _brute_force(G, terminals)
    sol = SteinerProblem(G, [terminals]).get_solution(time_limit=60)
    assert sol.objective == pytest.approx(want)
    assert _spans(sol.edges, terminals)
    assert _edge_cost(G, sol.edges) == pytest.approx(sol.objective)


@pytest.mark.parametrize("kwargs", [
    dict(da_reduce=True),
    dict(dual_ascent=True),
    dict(heavy=False),
])
def test_composes_with_other_accelerators(kwargs):
    G, terminals = _random_instance(40, 100, 6, seed=42)
    sol = SteinerProblem(G, [terminals], **kwargs).get_solution(time_limit=120)
    ref = SteinerProblem(G, [terminals], preprocess=False).get_solution(time_limit=120)
    assert sol.objective == pytest.approx(ref.objective)
    assert _spans(sol.edges, terminals)
    assert _edge_cost(G, sol.edges) == pytest.approx(sol.objective)


def test_heuristic_mode_gap_valid_with_contraction():
    G, terminals = _random_instance(40, 100, 6, seed=77)
    heur = SteinerProblem(G, [terminals]).get_solution(exact=False)
    opt = SteinerProblem(G, [terminals], preprocess=False).get_solution(time_limit=120)
    assert heur.objective >= opt.objective - 1e-6      # upper bound
    assert heur.gap >= -1e-9                           # valid gap
    # certified: objective - gap-implied lower bound <= objective
    assert heur.objective * (1 - heur.gap) <= opt.objective + 1e-6
    assert _spans(heur.edges, terminals)


def test_solution_mapping_with_synthetic_fixed_edge():
    # A degree-2 chain hanging off a terminal: the degree-2 contraction first
    # synthesises edge (0, 2), which then becomes a *fixed* edge when terminal
    # 0 (degree 1 in the reduced graph) is contracted — the fixed-edge
    # expansion must recover both original edges.
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)   # 0 (terminal) - 1 (steiner, degree 2)
    G.add_edge(1, 2, weight=1)   # 1 - 2
    G.add_edge(2, 3, weight=3)   # 2 - 3
    G.add_edge(2, 4, weight=3)
    G.add_edge(3, 4, weight=3)
    sol = SteinerProblem(G, [[0, 3, 4]]).get_solution(time_limit=30)
    ref = SteinerProblem(G, [[0, 3, 4]], preprocess=False).get_solution(time_limit=30)
    assert sol.objective == pytest.approx(ref.objective)
    assert _spans(sol.edges, [0, 3, 4])
    assert _edge_cost(G, sol.edges) == pytest.approx(sol.objective)


# ---------------------------------------------------------------------------
# NV / SL / BND tests (Polzin & Vahdati 1998; Rehfeldt master thesis §2.2)
# ---------------------------------------------------------------------------

def test_nv_contracts_nearest_vertex():
    # Terminal 0's cheapest edge goes to Steiner vertex 1, which is right next
    # to terminal 2: c(e') + d(v', t2) = 1 + 1 = 2 <= c(e'') = 5 -> contract.
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)   # e' = (t0, v')
    G.add_edge(0, 3, weight=5)   # e'' (second cheapest at t0)
    G.add_edge(1, 2, weight=1)   # v' - t2
    G.add_edge(3, 2, weight=5)
    reduced, tracker = preprocess_graph(
        G, [[0, 2]], special_distance=False, long_edge=False, contract=True)
    assert tracker.fixed_cost == 2  # both edges of the cheap path get fixed
    assert reduced.number_of_edges() == 0


def test_sl_promotes_new_terminal():
    # Two far-apart terminals joined by a single cheap "bridge" between two
    # Steiner vertices: the bridge is each region's only viable link, so SL
    # (or the cascade it enables) must fix it although neither endpoint is a
    # terminal. Side edges are expensive so NV alone cannot resolve them.
    G = nx.Graph()
    G.add_edge("t1", "a", weight=10)
    G.add_edge("t1", "a2", weight=10)
    G.add_edge("a", "a2", weight=10)
    G.add_edge("a", "b", weight=1)    # the bridge (only link between regions)
    G.add_edge("b", "t2", weight=10)
    G.add_edge("b", "b2", weight=10)
    G.add_edge("t2", "b2", weight=10)
    sol = SteinerProblem(G, [["t1", "t2"]]).get_solution(time_limit=30)
    ref = SteinerProblem(G, [["t1", "t2"]], preprocess=False).get_solution(time_limit=30)
    assert sol.objective == pytest.approx(ref.objective) == 21
    assert _spans(sol.edges, ["t1", "t2"])
    assert _edge_cost(G, sol.edges) == pytest.approx(sol.objective)


def test_bnd_deletes_far_vertex():
    from steinerpy.graph_reducer import (
        bound_based_deletions, _voronoi2, _terminal_mst,
    )
    # Triangle of terminals with cheap direct edges; vertex 9 hangs far away —
    # any tree through it costs at least 2*20 more than the SPH tree.
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(0, 9, weight=20)
    G.add_edge(2, 9, weight=20)
    terminals = {0, 1, 2}
    vor = _voronoi2(G, terminals, "weight")
    tmst = _terminal_mst(G, terminals, vor[0], vor[1], "weight")
    nodes, edges = bound_based_deletions(G, terminals, "weight", vor, tmst)
    assert 9 in nodes


@pytest.mark.parametrize("seed", range(6))
def test_nv_sl_bnd_preserve_optimum_random(seed):
    G, terminals = _random_instance(35, 90, 4 + seed % 4, seed=200 + seed)
    sol = SteinerProblem(G, [terminals]).get_solution(time_limit=120)
    ref = SteinerProblem(G, [terminals], preprocess=False).get_solution(time_limit=120)
    assert sol.objective == pytest.approx(ref.objective)
    assert _spans(sol.edges, terminals)
    assert _edge_cost(G, sol.edges) == pytest.approx(sol.objective)


def test_bound_based_flag_disables():
    G, terminals = _random_instance(30, 80, 5, seed=300)
    p = SteinerProblem(G, [terminals], bound_based=False)
    sol = p.get_solution(time_limit=60)
    ref = SteinerProblem(G, [terminals], preprocess=False).get_solution(time_limit=60)
    assert sol.objective == pytest.approx(ref.objective)
