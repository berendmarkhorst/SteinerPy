"""Tests for graph reduction and solution backmapping (steinerpy.graph_reducer).

Regression coverage for issue #20: when degree-2 reduction collapses a chain in
several steps, the solution must be mapped back to *original* graph edges, not to
intermediate contracted edges that no longer exist in the original graph.
"""

import networkx as nx

from steinerpy import SteinerProblem
from steinerpy.graph_reducer import (
    ReductionTracker,
    map_solution_to_original,
    preprocess_graph,
)


def _norm(edges):
    """Normalize a list of edges to a set of sorted tuples for comparison."""
    return {tuple(sorted(e)) for e in edges}


def _chain(n, weight=1):
    """Build an integer path graph 0-1-...-(n-1) with the given edge weight."""
    G = nx.Graph()
    for i in range(n - 1):
        G.add_edge(i, i + 1, weight=weight)
    return G


def test_backmapping_four_node_chain():
    """Issue #20: a chain collapsed in two contraction steps expands fully."""
    G = _chain(4)  # 0-1-2-3
    reduced, tracker = preprocess_graph(G, [[0, 3]], "weight")

    # Both interior nodes are gone; a single contracted edge connects 0 and 3.
    assert reduced.number_of_nodes() == 2
    assert reduced.number_of_edges() == 1

    mapped = map_solution_to_original([(0, 3)], tracker, reduced)
    assert _norm(mapped) == {(0, 1), (1, 2), (2, 3)}
    # Every mapped edge must exist in the original graph (the actual bug symptom).
    for u, v in mapped:
        assert G.has_edge(u, v)


def test_backmapping_six_node_chain():
    """A longer chain expands to all original edges."""
    G = _chain(6)  # 0-1-2-3-4-5
    reduced, tracker = preprocess_graph(G, [[0, 5]], "weight")
    assert reduced.number_of_nodes() == 2

    mapped = map_solution_to_original(list(reduced.edges()), tracker, reduced)
    assert _norm(mapped) == {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}
    for u, v in mapped:
        assert G.has_edge(u, v)


def test_backmapping_handles_reversed_edge_orientation():
    """A contracted edge given in reversed orientation still expands correctly."""
    G = _chain(4)
    reduced, tracker = preprocess_graph(G, [[0, 3]], "weight")
    mapped = map_solution_to_original([(3, 0)], tracker, reduced)
    assert _norm(mapped) == {(0, 1), (1, 2), (2, 3)}


def test_backmapping_pass_through_without_reductions():
    """With an empty tracker, edges are returned unchanged (preprocess=False path)."""
    G = _chain(3)  # 0-1-2
    tracker = ReductionTracker()
    edges = [(0, 1), (1, 2)]
    assert map_solution_to_original(edges, tracker, G) == edges


def test_backmapping_does_not_expand_original_edge_with_stale_contraction():
    """Issue #20 safety: an original edge that shares endpoints with a recorded
    contraction must NOT be expanded."""
    # Chain 0-1-2-3 plus a cheap direct edge (0, 3). The direct edge is cheaper
    # than the length-3 chain, so it survives as an 'original' edge even though a
    # {0, 2} contraction was recorded while collapsing the chain.
    G = _chain(4)
    G.add_edge(0, 3, weight=1)
    reduced, tracker = preprocess_graph(G, [[0, 3]], "weight")

    assert reduced.has_edge(0, 3)
    assert reduced[0][3].get("edge_type") != "contracted"

    mapped = map_solution_to_original([(0, 3)], tracker, reduced)
    assert _norm(mapped) == {(0, 3)}  # untouched, NOT expanded into the chain


def test_steiner_problem_backmapping_end_to_end():
    """Issue #20 regression: preprocess=True yields the same valid edges as False."""
    G = _chain(4)
    G.edges[0, 1]["weight"] = 1
    G.edges[1, 2]["weight"] = 100
    G.edges[2, 3]["weight"] = 30

    sol_pre = SteinerProblem(
        G, [[0, 3]], weight="weight", preprocess=True
    ).get_solution(time_limit=30)
    sol_raw = SteinerProblem(
        G, [[0, 3]], weight="weight", preprocess=False
    ).get_solution(time_limit=30)

    assert _norm(sol_pre.edges) == _norm(sol_raw.edges) == {(0, 1), (1, 2), (2, 3)}
    for u, v in sol_pre.edges:
        assert G.has_edge(u, v)
