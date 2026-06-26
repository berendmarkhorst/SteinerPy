"""Tests for the Hop-Constrained Directed Steiner Tree variant (thesis Ch. 5.8)."""

import math

import networkx as nx
import pytest

from steinerpy import HopConstrainedSteinerProblem


def _two_route_graph():
    """Root r reaches terminal t either directly (1 hop, cost 10) or via a->b
    (3 hops, cost 3)."""
    DG = nx.DiGraph()
    DG.add_edge("r", "t", weight=10)
    DG.add_edge("r", "a", weight=1)
    DG.add_edge("a", "b", weight=1)
    DG.add_edge("b", "t", weight=1)
    return DG


def test_slack_hop_limit_takes_cheap_long_path():
    DG = _two_route_graph()
    sol = HopConstrainedSteinerProblem(DG, root="r", terminals=["t"], hop_limit=3).get_solution()
    assert sol.objective == pytest.approx(3.0)
    assert len(sol.edges) == 3
    assert len(sol.edges) <= 3


def test_tight_hop_limit_forces_expensive_short_path():
    DG = _two_route_graph()
    sol = HopConstrainedSteinerProblem(DG, root="r", terminals=["t"], hop_limit=1).get_solution()
    assert sol.objective == pytest.approx(10.0)
    assert len(sol.edges) == 1


def test_infeasible_when_hop_limit_too_small():
    DG = nx.DiGraph()
    DG.add_edge("r", "a", weight=1)
    DG.add_edge("a", "b", weight=1)
    DG.add_edge("b", "t", weight=1)  # only a 3-arc route exists
    sol = HopConstrainedSteinerProblem(DG, root="r", terminals=["t"], hop_limit=2).get_solution()
    assert math.isinf(sol.objective)


def test_terminal_outgoing_arcs_removed():
    DG = _two_route_graph()
    DG.add_edge("t", "a", weight=0)  # an outgoing arc from the terminal
    prob = HopConstrainedSteinerProblem(DG, root="r", terminals=["t"], hop_limit=3)
    assert ("t", "a") not in prob.graph.edges()


def test_requires_directed_graph():
    G = nx.Graph()
    G.add_edge("r", "t", weight=1)
    with pytest.raises(ValueError):
        HopConstrainedSteinerProblem(G, root="r", terminals=["t"], hop_limit=1)
