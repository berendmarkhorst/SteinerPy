"""Tests for the ``enumeration_safe`` preprocessing mode (steinerpy#43).

The default reduction pipeline only preserves the optimal *value*: several
tests pick one witness among possibly several tied-optimal alternatives
(degree-2 contraction against a same-cost parallel edge, node replacement,
adjacent-terminal/Nearest-Vertex/Short-Links terminal contraction) and
structurally erase the rest. ``enumeration_safe=True`` restricts preprocessing
to reductions proven to preserve the *complete set* of optima, so it can be
safely combined with :meth:`BaseSteinerProblem.get_optimal_solutions`.
"""

import networkx as nx
import pytest

from steinerpy import SteinerProblem
from steinerpy.graph_reducer import preprocess_graph


def diamond_graph():
    """Two disjoint, equal-cost A->D paths: A-B-C-D and A-E-F-D, cost 3 each."""
    g = nx.Graph()
    for u, v in [("A", "B"), ("B", "C"), ("C", "D"), ("A", "E"), ("E", "F"), ("F", "D")]:
        g.add_edge(u, v, weight=1)
    return g


def test_degree_two_tie_is_skipped_under_enumeration_safe():
    """Both A-D paths collapse to one edge by default (steinerpy#41's bug);
    enumeration_safe leaves the tied alternative (node F) in the graph."""
    G = diamond_graph()

    reduced, _tracker = preprocess_graph(G, [["A", "D"]], contract=True)
    assert set(reduced.nodes()) == {"D"}  # fully collapsed, one path lost

    reduced_safe, _tracker = preprocess_graph(
        G, [["A", "D"]], contract=True, enumeration_safe=True
    )
    assert set(reduced_safe.nodes()) == {"A", "D", "F"}
    edges = {tuple(sorted((u, v))): d["weight"] for u, v, d in reduced_safe.edges(data=True)}
    assert edges == {("A", "D"): 3, ("A", "F"): 2, ("D", "F"): 1}


def test_strict_new_weight_still_contracts_under_enumeration_safe():
    """A strictly cheaper/costlier alternative is not a tie, so it is
    unaffected: only the exact-tie case is special-cased."""
    G = nx.Graph()
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "C", weight=1)
    G.add_edge("A", "C", weight=100)  # strictly worse parallel edge
    reduced, _tracker = preprocess_graph(G, [["A", "C"]], enumeration_safe=True)
    assert set(reduced.nodes()) == {"A", "C"}
    assert reduced["A"]["C"]["weight"] == 2


def test_node_replacement_disabled_under_enumeration_safe():
    """A degree-3 center provably eliminable at a tie (Prop. 4's non-strict
    certificate) is kept in place instead of removed."""
    G = nx.Graph()
    G.add_edge("t1", "t2", weight=1)
    G.add_edge("t2", "t3", weight=1)
    for t in ("t1", "t2", "t3"):
        G.add_edge("v", t, weight=1)
    groups = [["t1", "t2", "t3"]]

    reduced, _tracker = preprocess_graph(
        G, groups, special_distance=True, long_edge=True, replace_nodes=True
    )
    assert "v" not in reduced  # default: eliminated

    reduced_safe, _tracker = preprocess_graph(
        G, groups, special_distance=True, long_edge=True, replace_nodes=True,
        enumeration_safe=True,
    )
    assert "v" in reduced_safe  # enumeration-safe: kept


def test_nontrivial_terminal_contraction_disabled_under_enumeration_safe():
    """Adjacent-terminal contraction (a tie-prone witness pick) is skipped;
    no terminal in this triangle is degree-1, so nothing else fires either."""
    G = nx.Graph()
    G.add_edge("t1", "t2", weight=1)
    G.add_edge("t2", "t3", weight=5)
    G.add_edge("t1", "t3", weight=5)
    groups = [["t1", "t2", "t3"]]

    reduced, tracker = preprocess_graph(G, groups, contract=True)
    assert tracker.fixed_cost > 0  # default: contracted away

    reduced_safe, tracker_safe = preprocess_graph(
        G, groups, contract=True, enumeration_safe=True
    )
    assert tracker_safe.fixed_cost == 0
    assert tracker_safe.terminal_merges == {}
    assert set(reduced_safe.nodes()) == {"t1", "t2", "t3"}


def test_degree_one_terminal_contraction_still_fires_under_enumeration_safe():
    """The degree-1 terminal case is forced (its sole edge has no alternative
    to lose), so it still fires even under enumeration_safe."""
    G = nx.path_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 2
    reduced, tracker = preprocess_graph(
        G, [[0, 3]], contract=True, enumeration_safe=True
    )
    assert reduced.number_of_edges() == 0
    assert reduced.number_of_nodes() == 1
    assert tracker.fixed_cost == 6


def test_sound_edge_deletions_unaffected_by_enumeration_safe():
    """Long-edge deletion is a strict-inequality test (never erases a tied
    optimum), so enumeration_safe leaves it untouched. All three nodes are
    terminals so the redundant edge survives structural (degree-2) reduction
    long enough for the long-edge test to be the one that removes it."""
    G = nx.Graph()
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "C", weight=1)
    G.add_edge("A", "C", weight=100)  # strictly dominated by the A-B-C detour
    groups = [["A", "B", "C"]]
    reduced_default = preprocess_graph(G, groups, long_edge=True)[0]
    reduced_safe = preprocess_graph(G, groups, long_edge=True, enumeration_safe=True)[0]
    assert not reduced_default.has_edge("A", "C")
    assert not reduced_safe.has_edge("A", "C")


def test_get_optimal_solutions_finds_both_ties_with_enumeration_safe_preprocessing():
    """End-to-end: preprocessing reduces the graph but every tied optimum is
    still enumerable and back-maps to valid original-graph edges."""
    G = diamond_graph()
    problem = SteinerProblem(G, [["A", "D"]], preprocess=True, enumeration_safe=True)
    pool = problem.get_optimal_solutions(limit=10)

    assert pool.exhausted is True
    assert len(pool) == 2
    for sol in pool:
        assert sol.objective == 3.0
        assert sol.was_preprocessed is True
        for u, v in sol.edges:
            assert G.has_edge(u, v)

    edge_sets = {frozenset(frozenset(e) for e in sol.edges) for sol in pool}
    assert edge_sets == {
        frozenset(frozenset(e) for e in [("A", "B"), ("B", "C"), ("C", "D")]),
        frozenset(frozenset(e) for e in [("A", "E"), ("E", "F"), ("F", "D")]),
    }


def test_get_optimal_solutions_still_rejects_plain_preprocess_true():
    """Without enumeration_safe, preprocess=True is still rejected: default
    reduction is not enumeration-safe."""
    G = diamond_graph()
    problem = SteinerProblem(G, [["A", "D"]])  # preprocess=True, enumeration_safe=False
    with pytest.raises(ValueError, match="preprocess"):
        problem.get_optimal_solutions()
