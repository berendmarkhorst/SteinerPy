"""Tests for the Directed Prize-Collecting Steiner Tree Problem.

The classic forgo-prize objective on a digraph is ``sum_{a in S} c(a) +
sum_{v not in S} p(v)`` where ``S`` is a directed arborescence — rooted anywhere
(unrooted variant) or at a mandatory root (rooted variant, the G-Retriever
case).  We verify the directed transform path (exact and ``exact=False``)
against an **independent brute-force oracle** that enumerates vertex subsets
and computes a minimum spanning arborescence (Edmonds) per candidate root,
following the pattern of ``tests/test_pc_transform.py``.
"""

import logging
import random
from itertools import combinations

import networkx as nx
import pytest

from steinerpy import DirectedPrizeCollectingProblem
from steinerpy.pc_transform import (
    transform_directed_pcstp_to_sap,
    map_sap_solution_to_pcstp,
)

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Brute-force oracle (small instances only)
# ---------------------------------------------------------------------------

def _min_arborescence_cost(sub, s, weight="weight"):
    """Cost of a minimum spanning arborescence of ``sub`` rooted at ``s``,
    or None when no arborescence rooted at ``s`` spans ``sub``."""
    if set(nx.descendants(sub, s)) | {s} != set(sub.nodes()):
        return None
    # Removing the in-arcs of s forces every spanning arborescence to root at s.
    pruned = nx.DiGraph()
    pruned.add_nodes_from(sub.nodes())
    for u, v, d in sub.edges(data=True):
        if v == s:
            continue
        pruned.add_edge(u, v, weight=d.get(weight, 1))
    try:
        arb = nx.minimum_spanning_arborescence(pruned, attr="weight")
    except nx.NetworkXException:
        return None
    return sum(d["weight"] for _, _, d in arb.edges(data=True))


def brute_directed_pcstp(g, prizes, root=None, weight="weight"):
    """Optimal directed forgo-prize PCSTP cost by enumerating vertex subsets
    and candidate roots.  ``root=None`` allows any root and the empty tree."""
    nodes = list(g.nodes())
    total_p = sum(p for p in prizes.values() if p > 0)
    if root is None:
        # Empty tree and every single-vertex tree.
        best = total_p
        for v in nodes:
            best = min(best, total_p - max(0, prizes.get(v, 0)))
    else:
        best = total_p - max(0, prizes.get(root, 0))  # root-only tree
    for r in range(2, len(nodes) + 1):
        for S in combinations(nodes, r):
            if root is not None and root not in S:
                continue
            sub = g.subgraph(set(S))
            roots_to_try = [root] if root is not None else list(S)
            for s in roots_to_try:
                cost = _min_arborescence_cost(sub, s, weight)
                if cost is None:
                    continue
                cost += sum(prizes.get(v, 0) for v in nodes
                            if v not in set(S) and prizes.get(v, 0) > 0)
                best = min(best, cost)
    return best


def random_directed_pcstp(seed, n=6):
    rng = random.Random(seed)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n - 1):
        g.add_edge(perm[i], perm[i + 1], weight=rng.randint(1, 9))
    for _ in range(rng.randint(0, 2 * n)):
        u, v = rng.sample(range(n), 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v, weight=rng.randint(1, 9))
    prizes = {v: rng.choice([0, 0, rng.randint(1, 12)]) for v in range(n)}
    return g, prizes


# ---------------------------------------------------------------------------
# Exact solve == brute-force optimum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(15))
def test_directed_pc_exact_unrooted_matches_oracle(seed):
    g, prizes = random_directed_pcstp(seed)
    opt = brute_directed_pcstp(g, prizes)
    sol = DirectedPrizeCollectingProblem(g.copy(), prizes).get_solution()
    assert abs(sol.objective - opt) < 1e-6, (seed, sol.objective, opt)
    assert sol.gap == 0.0
    # Orientation is preserved: every reported edge is an arc of the DiGraph.
    for (u, v) in sol.selected_edges:
        assert g.has_edge(u, v), (seed, u, v)


@pytest.mark.parametrize("seed", range(15))
def test_directed_pc_exact_rooted_matches_oracle(seed):
    g, prizes = random_directed_pcstp(seed)
    root = 0
    opt = brute_directed_pcstp(g, prizes, root=root)
    sol = DirectedPrizeCollectingProblem(g.copy(), prizes, root=root).get_solution()
    assert abs(sol.objective - opt) < 1e-6, (seed, sol.objective, opt)
    assert sol.gap == 0.0
    assert root in sol.selected_nodes
    for (u, v) in sol.selected_edges:
        assert g.has_edge(u, v), (seed, u, v)


def test_directed_pc_picks_cheaper_arc_direction():
    # Anti-parallel arcs with asymmetric costs: the tree must use (a, b).
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=1)
    g.add_edge("b", "a", weight=6)
    prizes = {"a": 3, "b": 10}
    sol = DirectedPrizeCollectingProblem(g.copy(), prizes).get_solution()
    assert abs(sol.objective - 1.0) < 1e-6  # arc cost 1, no prize forgone
    assert sol.selected_edges == [("a", "b")]


def test_directed_pc_prizeless_root_vertex():
    # The optimal arborescence roots at a prize-less source: anchoring only at
    # prize nodes would miss it (b and c have no outgoing arcs).
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=1)
    g.add_edge("a", "c", weight=1)
    prizes = {"a": 0, "b": 10, "c": 10}
    opt = brute_directed_pcstp(g, prizes)
    sol = DirectedPrizeCollectingProblem(g.copy(), prizes).get_solution()
    assert abs(sol.objective - opt) < 1e-6
    assert abs(sol.objective - 2.0) < 1e-6  # both arcs, both prizes collected
    assert set(sol.selected_nodes) == {"a", "b", "c"}


def test_directed_pc_rooted_vs_unrooted():
    # Reaching the only prize is unprofitable: rooted keeps the bare root and
    # forgoes the prize; unrooted keeps the prize vertex alone at cost 0.
    g = nx.DiGraph()
    g.add_edge("r", "a", weight=2)
    prizes = {"r": 0, "a": 1}
    rooted = DirectedPrizeCollectingProblem(g.copy(), prizes, root="r").get_solution()
    assert abs(rooted.objective - 1.0) < 1e-6
    assert rooted.selected_nodes == ["r"] and rooted.selected_edges == []
    unrooted = DirectedPrizeCollectingProblem(g.copy(), prizes).get_solution()
    assert abs(unrooted.objective) < 1e-6
    assert unrooted.selected_nodes == ["a"]


def test_directed_pc_rooted_through_steiner_vertex():
    # G-Retriever-style: the root reaches the prizes only through a prize-less
    # intermediate, which must be paid for as a connector.
    g = nx.DiGraph()
    g.add_edge("q", "s", weight=1)
    g.add_edge("s", "e1", weight=1)
    g.add_edge("s", "e2", weight=1)
    prizes = {"q": 0, "s": 0, "e1": 5, "e2": 5}
    sol = DirectedPrizeCollectingProblem(g.copy(), prizes, root="q").get_solution()
    assert abs(sol.objective - 3.0) < 1e-6  # three arcs, no prize forgone
    assert set(sol.selected_nodes) == {"q", "s", "e1", "e2"}


# ---------------------------------------------------------------------------
# Heuristic mode: valid optimality gap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(15))
def test_directed_pc_heuristic_valid_gap_unrooted(seed):
    g, prizes = random_directed_pcstp(seed)
    opt = brute_directed_pcstp(g, prizes)
    sol = DirectedPrizeCollectingProblem(g.copy(), prizes).get_solution(exact=False)
    assert sol.gap >= -1e-9
    assert sol.objective >= opt - 1e-6      # primal is an upper bound (min problem)
    if abs(sol.gap) < 1e-9:                  # gap 0 certifies optimality
        assert abs(sol.objective - opt) < 1e-6


@pytest.mark.parametrize("seed", range(15))
def test_directed_pc_heuristic_valid_gap_rooted(seed):
    g, prizes = random_directed_pcstp(seed)
    opt = brute_directed_pcstp(g, prizes, root=0)
    sol = DirectedPrizeCollectingProblem(g.copy(), prizes, root=0).get_solution(exact=False)
    assert sol.gap >= -1e-9
    assert sol.objective >= opt - 1e-6
    if abs(sol.gap) < 1e-9:
        assert abs(sol.objective - opt) < 1e-6


# ---------------------------------------------------------------------------
# Transformation-level checks
# ---------------------------------------------------------------------------

def test_directed_transform_unrooted_structure():
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=4)
    prizes = {0: 0, 1: 7}
    ctx = transform_directed_pcstp_to_sap(g, prizes)
    assert ctx.directed and ctx.pcstp_root is None
    assert abs(ctx.offset - ctx.big_m) < 1e-9
    # One anchor arc per original vertex; the original arc keeps its cost.
    anchors = [a for a, k in ctx.aux_arc_kind.items() if k == "root"]
    assert len(anchors) == g.number_of_nodes()
    assert ctx.sap_graph.edges[(0, 1)]["weight"] == 4
    # Only the prize node gets a gadget terminal.
    assert len(ctx.terminals) == 1
    # All arc costs are non-negative (no cost-shifting is performed).
    for a in ctx.sap_graph.edges():
        assert ctx.sap_graph.edges[a]["weight"] >= 0


def test_directed_transform_rooted_structure():
    g = nx.DiGraph()
    g.add_edge("r", "t", weight=2)
    prizes = {"r": 0, "t": 5}
    ctx = transform_directed_pcstp_to_sap(g, prizes, root="r")
    assert ctx.directed and ctx.pcstp_root == "r"
    assert ctx.root == "r"                    # the SAP roots at the real vertex
    assert ctx.offset == 0.0
    assert not any(k == "root" for k in ctx.aux_arc_kind.values())


def test_directed_transform_rejects_bad_input():
    with pytest.raises(ValueError):
        transform_directed_pcstp_to_sap(nx.Graph(), {})
    g = nx.DiGraph()
    g.add_node(0)
    with pytest.raises(ValueError):
        transform_directed_pcstp_to_sap(g, {}, root=99)


def test_directed_backmap_preserves_orientation():
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=1)
    prizes = {"a": 4, "b": 4}
    ctx = transform_directed_pcstp_to_sap(g, prizes)
    from steinerpy.pc_transform import _root_label, _term_label
    sap_arcs = [(_root_label(), "a"), ("a", "b"),
                ("a", _term_label("a")), ("b", _term_label("b"))]
    edges, nodes, obj = map_sap_solution_to_pcstp(ctx, sap_arcs)
    assert edges == [("a", "b")]              # not collapsed to an undirected edge
    assert nodes == ["a", "b"]
    assert abs(obj - 1.0) < 1e-9              # arc cost 1, both prizes collected


# ---------------------------------------------------------------------------
# Constructor validation and trivial instances
# ---------------------------------------------------------------------------

def test_directed_pc_rejects_undirected_graph():
    g = nx.Graph()
    g.add_edge(0, 1, weight=1)
    with pytest.raises(ValueError):
        DirectedPrizeCollectingProblem(g, {0: 1})


def test_directed_pc_rejects_unknown_root():
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=1)
    with pytest.raises(ValueError):
        DirectedPrizeCollectingProblem(g, {0: 1}, root=99)


def test_directed_pc_rejects_empty_graph():
    with pytest.raises(ValueError):
        DirectedPrizeCollectingProblem(nx.DiGraph(), {})


def test_directed_pc_rejects_undirected_only_options():
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=1)
    with pytest.raises(ValueError):
        DirectedPrizeCollectingProblem(g, {1: 1}, pc_reduce=True)
    with pytest.raises(ValueError):
        DirectedPrizeCollectingProblem(g, {1: 1}, budget=5)
    with pytest.raises(ValueError):
        DirectedPrizeCollectingProblem(g, {1: 1}, max_degree=2)


def test_directed_pc_rejects_penalty_ilp_path():
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=1)
    problem = DirectedPrizeCollectingProblem(g, {1: 1})
    with pytest.raises(NotImplementedError):
        problem.get_solution(pc_transform=False)


def test_directed_pc_no_prizes_trivial():
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=1)
    sol = DirectedPrizeCollectingProblem(g.copy(), {}).get_solution()
    assert sol.objective == 0.0 and sol.selected_edges == [] and sol.gap == 0.0
    sol = DirectedPrizeCollectingProblem(g.copy(), {}, root=0).get_solution()
    assert sol.objective == 0.0 and sol.selected_nodes == [0]
