"""Central validation must not transform or mutate normalized input."""

import networkx as nx

from steinerpy import Problem, objects
from steinerpy._problem_spec import normalize_problem
from steinerpy._problem_validation import validate_problem


def test_validation_is_reusable_without_transforming_input():
    graph = nx.path_graph(3)
    options = {"budget": 3, "max_degree": 2, "preprocess": False}
    spec = normalize_problem(graph, None, [[0, 2]], None, options, Problem.variants)
    first = validate_problem(spec, "steiner", objects.SteinerProblem)
    second = validate_problem(spec, "steiner", objects.SteinerProblem)
    assert first == second
    first["terminal_groups"][0].clear()
    assert second["terminal_groups"] == [[0, 2]]
    assert spec.terminal_groups == ((0, 2),)
    assert options == {"budget": 3, "max_degree": 2, "preprocess": False}
    assert list(graph.edges) == [(0, 1), (1, 2)]
