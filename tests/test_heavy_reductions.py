"""Tests for the heavy edge-deletion reductions (Special Distance / long-edge).

These complement ``test_graph_reducer.py`` (degree-1/2 structural reductions).
The reductions only *delete* edges that are provably in no optimal solution, so
the defining property we test is: **the optimum is preserved**. We verify this
both on hand-built instances and by fuzzing against a brute-force exact solver.

References:
  Rehfeldt & Koch, "Implications, conflicts, and reductions for Steiner trees",
  Math. Programming B 197 (2023) — Special Distance test (Theorem 1).
  Ljubic, "Solving Steiner trees: recent advances...", Networks 77 (2021), Sec. 4.
"""

import random
from itertools import combinations

import networkx as nx
import pytest

from steinerpy.graph_reducer import (
    special_distance_deletions,
    long_edge_deletions,
    heavy_edge_deletions,
    preprocess_graph,
    map_solution_to_original,
    _voronoi,
    _voronoi2,
    _mehlhorn_terminal_mst,
    _sd_bound,
)


# ---------------------------------------------------------------------------
# Brute-force exact Steiner tree / forest (edge-subset enumeration, small only)
# ---------------------------------------------------------------------------

def _groups_connected(edges, groups):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    for g in groups:
        if len(g) <= 1:
            continue
        r0 = find(g[0])
        if any(find(t) != r0 for t in g[1:]):
            return False
    return True


def _optimal_cost(G, groups, weight="weight"):
    edges = [(u, v, d.get(weight, 1)) for u, v, d in G.edges(data=True)]
    m = len(edges)
    best = float("inf")
    for r in range(0, m + 1):
        for combo in combinations(range(m), r):
            cost = sum(edges[i][2] for i in combo)
            if cost >= best:
                continue
            es = [(edges[i][0], edges[i][1]) for i in combo]
            if _groups_connected(es, groups):
                best = cost
    return best


def _random_graph(n, p, seed, wmax=6):
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(1, n):
        G.add_edge(perm[i], perm[rng.randrange(i)], weight=rng.randint(1, wmax))
    for i in range(n):
        for j in range(i + 1, n):
            if not G.has_edge(i, j) and rng.random() < p:
                G.add_edge(i, j, weight=rng.randint(1, wmax))
    return G


# ---------------------------------------------------------------------------
# Hand-built instances
# ---------------------------------------------------------------------------

def test_long_edge_deletes_cheaper_detour():
    """Triangle a-b-c: the long a-c edge (5) loses to the detour a-b-c (2)."""
    G = nx.Graph()
    G.add_edge("a", "b", weight=1)
    G.add_edge("b", "c", weight=1)
    G.add_edge("a", "c", weight=5)
    dels = long_edge_deletions(G, "weight")
    assert {tuple(sorted(e)) for e in dels} == {("a", "c")}


def test_long_edge_keeps_tight_edge():
    """No deletion when the detour is not strictly cheaper (equal cost)."""
    G = nx.Graph()
    G.add_edge("a", "b", weight=1)
    G.add_edge("b", "c", weight=1)
    G.add_edge("a", "c", weight=2)  # equals detour -> NOT deleted (strict test)
    assert long_edge_deletions(G, "weight") == set()


def test_special_distance_deletes_via_terminal_bottleneck():
    """A long chord across terminals is removed by the Special Distance test
    even though every single sub-edge is cheaper than the chord."""
    G = nx.Graph()
    # path of unit edges through terminals t0..t3
    terms = ["t0", "t1", "t2", "t3"]
    for i in range(3):
        G.add_edge(terms[i], terms[i + 1], weight=1)
    # a chord t0-t3 of cost 3: special distance (terminal bottleneck) is 1 < 3
    G.add_edge("t0", "t3", weight=3)
    dels = special_distance_deletions(G, set(terms), "weight")
    assert {"t0", "t3"} in [set(e) for e in dels]


def test_special_distance_skipped_for_forest_via_heavy():
    """heavy_edge_deletions must NOT apply the terminal-hopping SD test across
    multiple groups (it would be unsound); long-edge still applies."""
    G = nx.Graph()
    G.add_edge("a", "b", weight=1)
    G.add_edge("b", "c", weight=1)
    G.add_edge("a", "c", weight=5)
    groups = [["a", "b"], ["c", "b"]]  # two groups -> forest
    dels = heavy_edge_deletions(G, groups, "weight",
                                special_distance=True, long_edge=True)
    # long-edge still removes the cheaper-detour chord; nothing unsound added
    assert {tuple(sorted(e)) for e in dels} == {("a", "c")}


# ---------------------------------------------------------------------------
# Fuzzed soundness: optimum preserved by the pure deletions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(40))
def test_deletions_preserve_tree_optimum(seed):
    G = _random_graph(random.Random(seed).randint(4, 7), 0.4, seed)
    if G.number_of_edges() > 14:
        pytest.skip("instance too large for brute force")
    nodes = list(G.nodes())
    rng = random.Random(seed + 100)
    groups = [rng.sample(nodes, rng.randint(2, min(4, len(nodes))))]
    before = _optimal_cost(G, groups)
    R = G.copy()
    changed = True
    while changed:
        changed = False
        for u, v in heavy_edge_deletions(R, groups, "weight",
                                         special_distance=True, long_edge=True):
            if R.has_edge(u, v):
                R.remove_edge(u, v)
                changed = True
    assert _optimal_cost(R, groups) == before


@pytest.mark.parametrize("seed", range(40))
def test_deletions_preserve_forest_optimum(seed):
    n = random.Random(seed).randint(5, 7)
    G = _random_graph(n, 0.45, seed * 3 + 1)
    if G.number_of_edges() > 14:
        pytest.skip("instance too large for brute force")
    nodes = list(G.nodes())
    rng = random.Random(seed + 7)
    sh = rng.sample(nodes, min(n, 6))
    h = max(2, len(sh) // 2)
    g1, g2 = sh[:h], sh[h:]
    if len(g2) < 2:
        pytest.skip("degenerate group")
    groups = [g1, g2]
    before = _optimal_cost(G, groups)
    R = G.copy()
    changed = True
    while changed:
        changed = False
        for u, v in heavy_edge_deletions(R, groups, "weight",
                                         special_distance=True, long_edge=True):
            if R.has_edge(u, v):
                R.remove_edge(u, v)
                changed = True
    assert _optimal_cost(R, groups) == before


@pytest.mark.parametrize("seed", range(30))
def test_preprocess_with_heavy_preserves_optimum_and_backmaps(seed):
    """Full preprocess_graph (degree + heavy + contraction) keeps the optimum
    value, and the solution back-maps to valid original-graph edges."""
    n = random.Random(seed).randint(4, 7)
    G = _random_graph(n, 0.4, seed * 5 + 2)
    if G.number_of_edges() > 13:
        pytest.skip("instance too large for brute force")
    nodes = list(G.nodes())
    rng = random.Random(seed + 11)
    single = rng.random() < 0.5
    if single:
        groups = [rng.sample(nodes, rng.randint(2, min(4, n)))]
    else:
        sh = rng.sample(nodes, min(n, 6))
        h = max(2, len(sh) // 2)
        groups = [sh[:h], sh[h:]]
        if len(groups[1]) < 2:
            pytest.skip("degenerate group")
    before = _optimal_cost(G, groups)
    R, tracker = preprocess_graph(G, groups, "weight",
                                  special_distance=True, long_edge=True)
    # value preserved on the reduced graph
    assert _optimal_cost(R, groups) == before
    # back-mapping yields existing, group-connecting edges of optimal weight
    # (find an optimal edge set on R, map it back)
    redges = [(u, v, d.get("weight", 1)) for u, v, d in R.edges(data=True)]
    m = len(redges)
    best, bestes = float("inf"), None
    for r in range(0, m + 1):
        for combo in combinations(range(m), r):
            cost = sum(redges[i][2] for i in combo)
            if cost >= best:
                continue
            es = [(redges[i][0], redges[i][1]) for i in combo]
            if _groups_connected(es, groups):
                best, bestes = cost, es
    mapped = map_solution_to_original(bestes, tracker, R)
    assert all(G.has_edge(u, v) for u, v in mapped)
    assert _groups_connected(mapped, groups)
    uniq = {tuple(sorted(e)) for e in mapped}
    assert sum(G[u][v]["weight"] for u, v in uniq) == before


# ---------------------------------------------------------------------------
# Accelerated internals: Voronoi diagram + Mehlhorn terminal MST
# ---------------------------------------------------------------------------

def test_voronoi_nearest_terminal_is_exact():
    """The single multi-source Dijkstra must agree with per-terminal distances."""
    G = _random_graph(12, 0.4, 77)
    terminals = {0, 5, 9}
    dist, base = _voronoi(G, terminals, "weight")
    for v in G.nodes():
        exact = min(
            nx.shortest_path_length(G, t, v, weight="weight") for t in terminals
        )
        assert dist[v] == pytest.approx(exact)
        # base must be a terminal that realises that distance
        assert nx.shortest_path_length(G, base[v], v, weight="weight") == pytest.approx(exact)


def test_mehlhorn_bottleneck_matches_complete_network():
    """Mehlhorn's Voronoi-boundary MST must give the same terminal bottleneck
    distances as the exact complete distance-network MST (minimax distances are
    invariant across spanning trees)."""
    G = _random_graph(14, 0.45, 31)
    terminals = {0, 3, 7, 11}
    dist, base = _voronoi(G, terminals, "weight")
    fast = _mehlhorn_terminal_mst(G, terminals, dist, base, "weight")

    # Reference: complete distance network on terminals, its MST, bottleneck.
    K = nx.Graph()
    for a in terminals:
        for b in terminals:
            if a < b:
                K.add_edge(a, b, weight=nx.shortest_path_length(G, a, b, weight="weight"))
    mst = nx.minimum_spanning_tree(K, weight="weight")
    for a in terminals:
        for b in terminals:
            if a == b:
                continue
            # bottleneck on the reference MST path
            path = nx.shortest_path(mst, a, b)
            ref = max(mst[u][v]["weight"] for u, v in zip(path, path[1:]))
            assert fast[a][b] == pytest.approx(ref)


def test_voronoi2_labels_are_exact_order_statistics():
    """The two-label multi-source Dijkstra must yield the smallest and second
    smallest per-terminal distances (the second over terminals distinct from
    the chosen nearest one — which equals the second order statistic regardless
    of tie-breaking)."""
    G = _random_graph(12, 0.4, 77)
    terminals = {0, 5, 9}
    d1, b1, d2, b2 = _voronoi2(G, terminals, "weight")
    for v in G.nodes():
        dists = sorted(
            (nx.shortest_path_length(G, t, v, weight="weight"), t) for t in terminals
        )
        assert d1[v] == pytest.approx(dists[0][0])
        assert nx.shortest_path_length(G, b1[v], v, weight="weight") == pytest.approx(d1[v])
        assert d2[v] == pytest.approx(dists[1][0])
        assert b2[v] != b1[v]
        assert nx.shortest_path_length(G, b2[v], v, weight="weight") == pytest.approx(d2[v])


def test_voronoi2_first_label_matches_voronoi():
    G = _random_graph(15, 0.35, 5)
    terminals = {1, 6, 11}
    dist, _base = _voronoi(G, terminals, "weight")
    d1, _b1, _d2, _b2 = _voronoi2(G, terminals, "weight")
    assert d1 == dist


def test_second_nearest_terminal_certifies_extra_sd_deletion():
    """An edge only the two-label SD bound can delete: the endpoints' nearest
    terminals are separated by a large bottleneck, but the second-nearest
    terminal of one endpoint is cheaply connected."""
    G = nx.Graph()
    G.add_edge("u", "ta", weight=1)
    G.add_edge("v", "tb", weight=1)
    G.add_edge("u", "tc", weight=2)
    G.add_edge("tc", "tb", weight=1)
    G.add_edge("u", "v", weight=2.5)
    terminals = {"ta", "tb", "tc"}

    vor = _voronoi2(G, terminals, "weight")
    bott = _mehlhorn_terminal_mst(G, terminals, vor[0], vor[1], "weight")
    # Nearest-only bound (classic test): max(1, b_T(ta, tb) = 3, 1) = 3 — keeps.
    nearest_only = (vor[0], vor[1], {}, {})
    assert _sd_bound("u", "v", nearest_only, bott) == pytest.approx(3)
    # Two-label bound routes u -> tc -> tb -> v: max(2, 1, 1) = 2 — deletes.
    assert _sd_bound("u", "v", vor, bott) == pytest.approx(2)

    dels = {tuple(sorted(e)) for e in special_distance_deletions(G, terminals, "weight")}
    assert ("u", "v") in dels
    # ...and the long-edge test alone would NOT delete it (detour costs 4).
    assert ("u", "v") not in {tuple(sorted(e)) for e in long_edge_deletions(G, "weight")}


@pytest.mark.parametrize("seed", range(25))
def test_accelerated_long_edge_matches_definition(seed):
    """Per-vertex long-edge deletion must equal the definition: delete {v,w} iff
    the shortest v-w distance is strictly below c({v,w})."""
    G = _random_graph(random.Random(seed).randint(5, 9), 0.4, seed * 2 + 1)
    got = {tuple(sorted(e)) for e in long_edge_deletions(G, "weight")}
    expected = set()
    for u, v, d in G.edges(data=True):
        if nx.shortest_path_length(G, u, v, weight="weight") < d["weight"] - 1e-9:
            expected.add(tuple(sorted((u, v))))
    assert got == expected
