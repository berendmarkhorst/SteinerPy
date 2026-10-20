"""Legacy adapters either use safe transformations or reject unmapped modifiers."""

import random

import networkx as nx
import pytest

from steinerpy import (
    GroupSteinerProblem,
    DirectedGroupSteinerProblem,
    NodeWeightedSteinerProblem,
    Problem,
)
from steinerpy import objects
from .test_directed_group_steiner import solver
from .test_composition import exhaustive_cost
from .test_problem_validation import forbid_construction


@pytest.mark.parametrize("adapter", ["group", "directed_group", "node_weighted"])
def test_modular_parallel_edges_fail_before_adapter_construction(
    adapter, forbid_construction
):
    graph = (
        nx.MultiDiGraph([(0, 1)])
        if adapter == "directed_group"
        else nx.MultiGraph([(0, 1)])
    )
    if adapter == "node_weighted":
        options = {"variant": adapter, "terminals": [0, 1], "node_weights": {}}
    else:
        options = {
            "groups": [[0], [1]],
            **({"root": 0} if adapter == "directed_group" else {}),
        }
    with pytest.raises(ValueError, match="simple"):
        Problem(graph, **options)


def test_modular_group_preprocessing_fails_before_adapter_construction(
    forbid_construction,
):
    with pytest.raises(ValueError, match="preprocess=False"):
        Problem(nx.path_graph(2), groups=[[0], [1]], preprocess=True)


@pytest.mark.parametrize("adapter", ["group", "node_weighted"])
@pytest.mark.parametrize("modifier", ["budget", "hop_limit", "max_degree"])
def test_legacy_modifiers_fail_before_transformation(adapter, modifier, monkeypatch):
    graph = nx.path_graph(3)
    before = graph.copy()

    def forbidden(*a, **k):
        pytest.fail("Unmapped modifiers must be rejected before transformation")

    monkeypatch.setattr(objects, "_attach_group_terminals", forbidden)
    monkeypatch.setattr(objects, "node_split_graph", forbidden)
    with pytest.raises(TypeError, match="cannot yet be combined"):
        if adapter == "group":
            GroupSteinerProblem(graph, [[0], [2]], **{modifier: 2})
        else:
            NodeWeightedSteinerProblem(graph, [[0, 2]], {}, **{modifier: 2})
    assert nx.utils.graphs_equal(graph, before)


def test_undirected_group_connectors_cannot_create_free_shortcuts(solver):
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1), (1, 2, 100), (2, 3, 1)])
    sol = GroupSteinerProblem(graph, [[0], [1, 2], [3]]).get_solution(solver=solver)
    assert sol.objective == 102 and sol.gap == 0
    assert nx.is_connected(nx.Graph(sol.edges))


@pytest.mark.parametrize("seed", range(12))
def test_legacy_and_modular_groups_match_original_oracle(seed, solver):
    rng = random.Random(seed)
    graph = nx.cycle_graph(5)
    graph.add_edge(0, 2)
    nx.set_edge_attributes(graph, {e: rng.randint(0, 8) for e in graph.edges}, "cost")
    groups = [rng.sample(list(graph), rng.randint(1, 3)) for _ in range(3)]
    expected = exhaustive_cost(graph, groups, {}, weight="cost")
    for problem in (
        GroupSteinerProblem(graph, groups, weight="cost"),
        Problem(graph, groups=groups, weight="cost"),
    ):
        s = problem.get_solution(solver=solver)
        assert s.objective == expected and s.gap == 0
        if s.edges:
            selected = nx.Graph(s.edges)
            assert nx.is_connected(selected)
            assert all(not set(selected).isdisjoint(g) for g in groups)


def test_legacy_node_cost_default_and_preprocess_false_are_preserved(solver):
    g = nx.path_graph(3)
    s = NodeWeightedSteinerProblem(
        g, [[0, 2]], {0: 2, 1: 3, 2: 4}, preprocess=False
    ).get_solution(solver=solver)
    assert s.objective == 9 and s.gap == 0
    assert len(s.edges) == 2


@pytest.mark.parametrize("adapter", ["group", "node_weighted", "directed_group"])
def test_parallel_edges_are_rejected_before_identity_is_lost(adapter):
    g = nx.MultiDiGraph() if adapter == "directed_group" else nx.MultiGraph()
    g.add_edge(0, 1, weight=1)
    g.add_edge(0, 1, weight=9)
    with pytest.raises(ValueError, match="simple"):
        if adapter == "group":
            GroupSteinerProblem(g, [[0], [1]])
        elif adapter == "directed_group":
            DirectedGroupSteinerProblem(g, [[1]], root=0)
        else:
            NodeWeightedSteinerProblem(g, [[0, 1]], {})


@pytest.mark.parametrize("adapter", ["group", "node_weighted"])
def test_explicit_preprocessing_is_rejected(adapter):
    with pytest.raises(ValueError, match="preprocess=False"):
        if adapter == "group":
            GroupSteinerProblem(nx.path_graph(2), [[0], [1]], preprocess=True)
        else:
            NodeWeightedSteinerProblem(nx.path_graph(2), [[0, 1]], {}, preprocess=True)
