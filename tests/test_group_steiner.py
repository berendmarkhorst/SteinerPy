"""Tests for the Group Steiner Tree variant (thesis Ch. 5.7)."""

import networkx as nx
import pytest

from steinerpy import GroupSteinerProblem


def _path_graph():
    G = nx.Graph()
    for u, v in [("A", "B"), ("B", "C"), ("C", "D")]:
        G.add_edge(u, v, weight=1)
    return G


def test_group_steiner_picks_cheapest_representatives():
    G = _path_graph()
    # Group 1 = {A, B}, group 2 = {C, D}. Cheapest connection is the single edge B-C.
    sol = GroupSteinerProblem(G, [["A", "B"], ["C", "D"]]).get_solution()
    assert sol.objective == pytest.approx(1.0)
    chosen = {n for e in sol.edges for n in e}
    assert chosen & {"A", "B"}  # at least one vertex from group 1
    assert chosen & {"C", "D"}  # at least one vertex from group 2
    # Connector super-terminals must not leak into the reported solution.
    assert not any(str(n).startswith("__group_") for n in chosen)


def test_group_steiner_singletons_equals_steiner_tree():
    G = _path_graph()
    # Singleton groups reduce to a plain Steiner tree connecting A and D (cost 3).
    sol = GroupSteinerProblem(G, [["A"], ["D"]]).get_solution()
    assert sol.objective == pytest.approx(3.0)


def test_empty_group_rejected():
    G = _path_graph()
    with pytest.raises(ValueError):
        GroupSteinerProblem(G, [["A"], []])


def test_group_vertex_not_in_graph_rejected():
    G = _path_graph()
    with pytest.raises(ValueError):
        GroupSteinerProblem(G, [["A"], ["Z"]])


def test_label_collision_is_avoided():
    # A node literally named like the default super-terminal must not clash.
    G = nx.Graph()
    G.add_edge("__group_0__", "B", weight=1)
    G.add_edge("B", "C", weight=1)
    sol = GroupSteinerProblem(G, [["__group_0__"], ["C"]]).get_solution()
    assert sol.objective == pytest.approx(2.0)


def test_directed_graph_rejected():
    # DirectedGroupSteinerProblem is the directed variant (see
    # test_directed_group_steiner.py).
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1)
    with pytest.raises(ValueError):
        GroupSteinerProblem(G, [["A"], ["B"]])
