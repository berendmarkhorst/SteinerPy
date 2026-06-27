"""Tests for the Partial / Full Terminal Steiner Tree variants (thesis Ch. 5.1)."""

import math

import networkx as nx
import pytest

from steinerpy import (
    SteinerProblem,
    PartialTerminalSteinerProblem,
    FullTerminalSteinerProblem,
)


def _star_graph():
    """A-M-B-M-C star plus the A-B-C path edges; terminals A, B, C.

    Plain Steiner tree connects them via the path A-B-C (cost 2, B internal); forcing
    every terminal to be a leaf instead requires the star through M (cost 3).
    """
    G = nx.Graph()
    for u, v in [("A", "M"), ("B", "M"), ("C", "M"), ("A", "B"), ("B", "C")]:
        G.add_edge(u, v, weight=1)
    return G


def test_plain_steiner_uses_internal_terminal():
    G = _star_graph()
    sol = SteinerProblem(G, [["A", "B", "C"]]).get_solution()
    assert sol.objective == pytest.approx(2.0)


def test_full_terminal_forces_leaves():
    G = _star_graph()
    sol = FullTerminalSteinerProblem(G, [["A", "B", "C"]]).get_solution()
    # Star through M: A-M, B-M, C-M, total 3.
    assert sol.objective == pytest.approx(3.0)
    # Every terminal is a leaf (degree exactly 1 in the solution).
    deg = {n: 0 for n in ("A", "B", "C")}
    for u, v in sol.edges:
        for x in (u, v):
            if x in deg:
                deg[x] += 1
    assert all(d == 1 for d in deg.values())


def test_partial_terminal_allows_chosen_internal():
    G = _star_graph()
    # Only A and C must be leaves; B may be internal -> path A-B-C (cost 2).
    sol = PartialTerminalSteinerProblem(
        G, [["A", "B", "C"]], partial_terminals=["A", "C"]
    ).get_solution()
    assert sol.objective == pytest.approx(2.0)
    deg = {"A": 0, "C": 0}
    for u, v in sol.edges:
        for x in (u, v):
            if x in deg:
                deg[x] += 1
    assert deg["A"] == 1 and deg["C"] == 1  # the partial terminals are leaves


def test_unknown_partial_terminal_raises():
    G = _star_graph()
    with pytest.raises(ValueError):
        PartialTerminalSteinerProblem(G, [["A", "B", "C"]], partial_terminals=["Z"])


def test_directed_graph_rejected():
    DG = nx.DiGraph()
    DG.add_edge("A", "B", weight=1)
    with pytest.raises(ValueError):
        PartialTerminalSteinerProblem(DG, [["A", "B"]], partial_terminals=["A"])


def test_infeasible_full_terminal_triangle():
    # Three mutually adjacent terminals, no Steiner node: no tree has all as leaves.
    G = nx.Graph()
    for u, v in [("A", "B"), ("B", "C"), ("A", "C")]:
        G.add_edge(u, v, weight=1)
    with pytest.raises(RuntimeError):
        FullTerminalSteinerProblem(G, [["A", "B", "C"]]).get_solution()
