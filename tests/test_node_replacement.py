"""Tests for degree-k node replacement / pseudo-elimination.

Rehfeldt & Koch, "Implications, conflicts, and reductions for Steiner trees",
Math. Programming B 197 (2023), Proposition 4: a non-terminal that provably has
degree <= 2 in at least one minimum Steiner tree is deleted and each neighbour
pair bridged by the two-edge path cost, recorded like a degree-2 contraction.

The soundness contract is the same as for the other reductions: the optimum
*value* is preserved (the criterion is non-strict, so at ties a different —
equal-cost — optimum may survive), and reduced solutions back-map to valid
original edges of optimal total weight.
"""

import random
from itertools import combinations

import networkx as nx
import pytest

from steinerpy.graph_reducer import (
    preprocess_graph,
    map_solution_to_original,
    _pseudo_eliminate_pass,
    _NTD_MAX_DEGREE,
    ReductionTracker,
)
from tests.test_heavy_reductions import (
    _optimal_cost,
    _random_graph,
    _groups_connected,
)


def _optimal_edge_set(G, groups, weight="weight"):
    """Brute-force optimal edge subset (small instances only)."""
    edges = [(u, v, d.get(weight, 1)) for u, v, d in G.edges(data=True)]
    m = len(edges)
    best, bestes = float("inf"), []
    for r in range(0, m + 1):
        for combo in combinations(range(m), r):
            cost = sum(edges[i][2] for i in combo)
            if cost >= best:
                continue
            es = [(edges[i][0], edges[i][1]) for i in combo]
            if _groups_connected(es, groups):
                best, bestes = cost, es
    return best, bestes


# ---------------------------------------------------------------------------
# Hand-built instances
# ---------------------------------------------------------------------------

def test_eliminates_center_when_terminals_are_cheaply_connected():
    """Triangle path of terminals plus a degree-3 center: the two largest
    terminal-MST weights (1+1) are below the center's incident sum (3), so the
    center is pseudo-eliminated; all replacement edges merge or drop away."""
    G = nx.Graph()
    G.add_edge("t1", "t2", weight=1)
    G.add_edge("t2", "t3", weight=1)
    for t in ("t1", "t2", "t3"):
        G.add_edge("v", t, weight=1)
    groups = [["t1", "t2", "t3"]]
    before = _optimal_cost(G, groups)
    R, _tracker = preprocess_graph(G, groups, "weight", special_distance=True,
                                   long_edge=True, replace_nodes=True)
    assert "v" not in R
    assert {tuple(sorted(e)) for e in R.edges()} == {("t1", "t2"), ("t2", "t3")}
    assert _optimal_cost(R, groups) == before == 2


def test_keeps_center_that_every_optimum_needs():
    """Pure terminal star: the center is the unique Steiner point of the unique
    optimum, and the criterion (2 largest MST weights 10+10 > incident sum 15)
    correctly refuses to eliminate it."""
    G = nx.Graph()
    for t in ("a", "b", "c"):
        G.add_edge("v", t, weight=5)
    groups = [["a", "b", "c"]]
    R, _tracker = preprocess_graph(G, groups, "weight", special_distance=True,
                                   long_edge=True, replace_nodes=True)
    assert "v" in R
    assert _optimal_cost(R, groups) == 15


def test_two_terminals_make_every_small_degree_nonterminal_eliminable():
    """|T| = 2 leaves the criterion range empty (an optimal tree is a path, so
    no node has solution degree 3), and the replacement edges reproduce the
    two-edge paths through the eliminated node — including an 'add' and two
    'improve' candidates whose back-mapping expands to original edges."""
    G = nx.Graph()
    G.add_edge("t1", "v", weight=1)
    G.add_edge("t2", "v", weight=1)
    G.add_edge("u", "v", weight=1)
    G.add_edge("u", "t1", weight=10)
    G.add_edge("u", "t2", weight=10)
    groups = [["t1", "t2"]]
    before = _optimal_cost(G, groups)
    R, tracker = preprocess_graph(G, groups, "weight", special_distance=False,
                                  long_edge=False, replace_nodes=True)
    assert "v" not in R
    assert _optimal_cost(R, groups) == before == 2
    # The surviving (t1, t2) edge is the recorded (v, t1, t2) replacement.
    _best, bestes = _optimal_edge_set(R, groups)
    mapped = map_solution_to_original(bestes, tracker, R)
    assert {tuple(sorted(e)) for e in mapped} == {("t1", "v"), ("t2", "v")}
    assert all(G.has_edge(a, b) for a, b in mapped)


def test_backmap_dedups_shared_original_edge_at_cost_tie():
    """Non-strict criterion tie: the reduced optimum uses two replacement edges
    of one eliminated node whose expansions share an original edge; the mapped
    solution must contain it once and still have optimal cost."""
    G = nx.Graph()
    G.add_edge("t1", "v", weight=1)
    G.add_edge("t2", "v", weight=1)
    G.add_edge("u", "v", weight=0)
    groups = [["t1", "t2", "u"]]
    before = _optimal_cost(G, groups)
    R, tracker = preprocess_graph(G, groups, "weight", special_distance=False,
                                  long_edge=False, replace_nodes=True)
    assert "v" not in R
    assert _optimal_cost(R, groups) == before == 2
    _best, bestes = _optimal_edge_set(R, groups)
    assert len(bestes) == 2  # both replacement edges are needed
    mapped = map_solution_to_original(bestes, tracker, R)
    # (t1,v)+(v,u) and (t2,v)+(v,u) share (v,u): deduped to three edges.
    assert len(mapped) == len({frozenset(e) for e in mapped}) == 3
    assert all(G.has_edge(a, b) for a, b in mapped)
    assert sum(G[a][b]["weight"] for a, b in mapped) == before


def test_degree_four_growth_guard_blocks_pure_edge_blowup():
    """A degree-4 candidate whose six neighbour pairs would all be new edges is
    skipped (6 > deg); once three of the pairs are existing (improvable) edges
    the same node is eliminated."""
    assert _NTD_MAX_DEGREE >= 4

    def star():
        g = nx.Graph()
        for u in ("u1", "u2", "u3", "u4"):
            g.add_edge("v", u, weight=1)
        return g

    terminals = {"t1", "t2"}  # |T| = 2: the criterion range is empty
    # Fabricated Voronoi/bottleneck data: every neighbour is far from the
    # terminals, so no candidate is dropped by the SD prefilter.
    vor = ({u: 100.0 for u in ("u1", "u2", "u3", "u4")},
           {u: "t1" for u in ("u1", "u2", "u3", "u4")}, {}, {})
    bott = {"t1": {"t1": 0.0}}

    G = star()
    touched = _pseudo_eliminate_pass(G, terminals, "weight", ReductionTracker(),
                                     vor, bott, [1.0])
    assert "v" in G and not touched  # all six candidates would be additions

    G = star()
    for a, b in (("u1", "u2"), ("u1", "u3"), ("u1", "u4")):
        G.add_edge(a, b, weight=5)
    tracker = ReductionTracker()
    touched = _pseudo_eliminate_pass(G, terminals, "weight", tracker,
                                     vor, bott, [1.0])
    assert "v" not in G and touched == {"u1", "u2", "u3", "u4"}
    # Eliminating v improves the three u1-edges to 2 and adds the other three
    # pairs at weight 2, all recorded; the pass then legitimately cascades and
    # eliminates u1 as well (degree 3, all candidates merge into the cheaper
    # weight-2 edges, so nothing further is recorded).
    assert len(tracker.degree_two_contractions) == 6
    assert "u1" not in G
    for a, b in combinations(("u2", "u3", "u4"), 2):
        assert G[a][b]["weight"] == 2
        assert G[a][b]["edge_type"] == "contracted"


# ---------------------------------------------------------------------------
# Fuzzed soundness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(40))
def test_replacement_preserves_optimum_and_backmaps(seed):
    n = random.Random(seed).randint(4, 8)
    G = _random_graph(n, 0.45, seed * 5 + 2)
    if G.number_of_edges() > 14:
        pytest.skip("instance too large for brute force")
    nodes = list(G.nodes())
    rng = random.Random(seed + 11)
    groups = [rng.sample(nodes, rng.randint(2, min(4, n)))]
    before = _optimal_cost(G, groups)
    R, tracker = preprocess_graph(G, groups, "weight", special_distance=True,
                                  long_edge=True, replace_nodes=True)
    assert _optimal_cost(R, groups) == before
    _best, bestes = _optimal_edge_set(R, groups)
    mapped = map_solution_to_original(bestes, tracker, R)
    assert all(G.has_edge(a, b) for a, b in mapped)
    assert _groups_connected(mapped, groups)
    assert len(mapped) == len({frozenset(e) for e in mapped})  # deduped
    assert sum(G[a][b]["weight"] for a, b in mapped) == before


@pytest.mark.parametrize("seed", range(15))
def test_replacement_gated_off_for_forests(seed):
    """With multiple terminal groups the replacement test must never fire:
    preprocessing with and without the flag yields the identical graph."""
    n = random.Random(seed).randint(5, 8)
    G = _random_graph(n, 0.45, seed * 3 + 1)
    nodes = list(G.nodes())
    rng = random.Random(seed + 7)
    sh = rng.sample(nodes, min(n, 6))
    h = max(2, len(sh) // 2)
    groups = [sh[:h], sh[h:]]
    if len(groups[1]) < 2:
        pytest.skip("degenerate group")
    R_on, _ = preprocess_graph(G, groups, "weight", special_distance=True,
                               long_edge=True, replace_nodes=True)
    R_off, _ = preprocess_graph(G, groups, "weight", special_distance=True,
                                long_edge=True, replace_nodes=False)
    assert set(R_on.nodes()) == set(R_off.nodes())
    assert {frozenset(e) for e in R_on.edges()} == {frozenset(e) for e in R_off.edges()}
    for u, v in R_on.edges():
        assert R_on[u][v]["weight"] == R_off[u][v]["weight"]


def test_preprocess_does_not_mutate_input():
    G = _random_graph(9, 0.4, 123)
    nodes0 = set(G.nodes())
    edges0 = {frozenset(e): d["weight"] for *e, d in G.edges(data=True)}
    attrs0 = {frozenset((u, v)): dict(d) for u, v, d in G.edges(data=True)}
    preprocess_graph(G, [[0, 3, 7]], "weight", special_distance=True,
                     long_edge=True, replace_nodes=True)
    assert set(G.nodes()) == nodes0
    assert {frozenset(e): d["weight"] for *e, d in G.edges(data=True)} == edges0
    assert {frozenset((u, v)): dict(d) for u, v, d in G.edges(data=True)} == attrs0
