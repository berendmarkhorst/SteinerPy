"""Feature inference and normalized connectivity checks."""

import networkx as nx
import pytest

from steinerpy import (
    Problem,
    SteinerProblem,
    DirectedSteinerProblem,
    DirectedGroupSteinerProblem,
    HopConstrainedSteinerProblem,
)


def test_flat_and_nested_terminals_and_legacy_alias():
    graph = nx.path_graph(5)
    nx.set_edge_attributes(graph, 1, "weight")
    solutions = [
        Problem(graph, terminals=[0, 4]).get_solution(),
        Problem(graph, terminals=[[0, 4]]).get_solution(),
        Problem(graph, terminal_groups=[[0, 4]]).get_solution(),
        Problem(graph, [[0, 4]]).get_solution(),
    ]
    assert all(sol.objective == 4 for sol in solutions)
    assert isinstance(Problem(graph, terminals=[0, 4]).implementation, SteinerProblem)


def test_forest_does_not_require_connecting_sets():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2), (2, 3, 3)])
    assert Problem(graph, terminals=[[0, 1], [2, 3]]).get_solution().objective == 5


def test_directed_terminal_aliases_and_root_validation():
    graph = nx.DiGraph()
    graph.add_edge((0, 1), "end", weight=2)
    for kwargs in ({"terminals": ["end"]}, {"terminal_groups": [["end"]]}):
        assert Problem(graph, root=(0, 1), **kwargs).get_solution().objective == 2
    with pytest.raises(TypeError, match="root"):
        Problem(graph, root=None, terminals=["end"])
    with pytest.raises(ValueError, match="root.*not in the graph"):
        Problem(graph, root="missing", terminals=["end"])


def test_forest_shares_edges():
    graph = nx.path_graph(5)
    nx.set_edge_attributes(graph, 1, "weight")
    sol = Problem(graph, terminals=[[0, 3], [1, 4]], preprocess=False).get_solution()
    assert sol.objective == 4
    assert len(sol.edges) == 4


def test_tuple_nodes_are_atomic():
    graph = nx.Graph()
    graph.add_edge((0, 1), (2, 3), weight=2)
    assert Problem(graph, terminals=[(0, 1), (2, 3)]).get_solution().objective == 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"terminals": [0, [1]]}, "mix"),
        ({"terminals": []}, "empty"),
        ({"terminals": [[]]}, "empty"),
        ({"terminals": [0, 3]}, "not in the graph"),
        ({"terminals": [0], "terminal_groups": [[0]]}, "not both"),
        ({"groups": [[0], []]}, "at least one"),
        ({"groups": [[0], [3]]}, "not in the graph"),
    ],
)
def test_connectivity_validation(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        Problem(nx.path_graph(3), **kwargs)


def test_infer_directed_and_hop_and_group():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=2)
    for kwargs, expected in [
        ({"terminals": [1]}, DirectedSteinerProblem),
        ({"terminals": [1], "hop_limit": 1}, HopConstrainedSteinerProblem),
        ({"groups": [[1]]}, DirectedGroupSteinerProblem),
    ]:
        problem = Problem(graph, root=0, **kwargs)
        assert isinstance(problem.implementation, expected)
        assert problem.get_solution().objective == 2
    with pytest.raises(TypeError, match="root"):
        Problem(graph, terminals=[1])
    with pytest.raises(ValueError, match="one connected tree"):
        Problem(graph, root=0, terminals=[[0], [1]])
