"""Tests for the Rectilinear Steiner Minimum Tree variant (thesis Ch. 5.4)."""

import pytest

from steinerpy import RectilinearSteinerProblem, RectilinearSolution
from steinerpy.rectilinear import hanan_grid


def test_hanan_grid_dimensions():
    G, terminals = hanan_grid([(0, 0), (2, 0), (0, 2)])
    # 2 distinct x's and 2 distinct y's -> 4 grid nodes.
    assert G.number_of_nodes() == 4
    assert set(terminals) == {(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)}
    # Edge weights are the L1 distances between adjacent grid nodes.
    assert G.edges[(0.0, 0.0), (2.0, 0.0)]["weight"] == pytest.approx(2.0)


def test_l_shaped_triple_length():
    # The rectilinear Steiner tree of an L-triple is the two legs: 2 + 2 = 4.
    sol = RectilinearSteinerProblem([(0, 0), (2, 0), (0, 2)]).get_solution()
    assert isinstance(sol, RectilinearSolution)
    assert sol.objective == pytest.approx(4.0)
    # Every segment is axis-aligned (shares an x- or a y-coordinate).
    for (xa, ya), (xb, yb) in sol.segments:
        assert xa == xb or ya == yb


def test_collinear_points_length():
    sol = RectilinearSteinerProblem([(0, 0), (1, 0), (3, 0)]).get_solution()
    assert sol.objective == pytest.approx(3.0)


def test_square_corners_length():
    # Four corners of a 2x2 square: rectilinear Steiner minimum tree length is 6.
    sol = RectilinearSteinerProblem([(0, 0), (2, 0), (0, 2), (2, 2)]).get_solution()
    assert sol.objective == pytest.approx(6.0)


def test_empty_points_rejected():
    with pytest.raises(ValueError):
        hanan_grid([])
