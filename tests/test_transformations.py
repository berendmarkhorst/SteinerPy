"""Reusable node splitting preserves original identities and cost conventions."""

import itertools

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy._transformations import split_node_costs
from steinerpy.objects import node_split_graph


@pytest.mark.parametrize("default", [0, 1])
def test_split_ports_costs_provenance_and_input_ownership(default):
    graph = nx.Graph()
    graph.add_edge(1, "1", cost=3)
    graph.add_edge("1", (2, 3))
    graph.add_node("isolated")
    costs = {1: 4, "1": 5, "isolated": 2}
    compiled = split_node_costs(graph, costs, "cost", default_edge_cost=default)
    assert graph.number_of_nodes() == 4 and graph.number_of_edges() == 2
    assert "cost" not in graph["1"][(2, 3)]
    assert costs == {1: 4, "1": 5, "isolated": 2}
    assert len(compiled.graph) == 8
    assert len(set(label for pair in compiled.split.values() for label in pair)) == 8
    for node, pair in compiled.split.items():
        assert compiled.node_arcs[pair] == node
        assert compiled.graph.edges[pair]["cost"] == costs.get(node, 0)
    for u, v, data in graph.edges(data=True):
        for a, b in [(u, v), (v, u)]:
            arc = (compiled.split[a][1], compiled.split[b][0])
            assert compiled.edge_arcs[arc] == (u, v)
            assert compiled.graph.edges[arc]["cost"] == data.get("cost", default)
    graph.clear()
    costs.clear()
    assert len(compiled.split) == 4
    assert compiled.graph.edges[compiled.split[1]]["cost"] == 4


def test_custom_split_labels_are_snapshotted():
    graph = nx.path_graph(2)
    labels = {0: ["a", "b"], 1: ["c", "d"]}
    compiled = split_node_costs(graph, {}, split=labels)
    labels[0][0] = "changed"
    labels.clear()
    assert compiled.split == {0: ("a", "b"), 1: ("c", "d")}
    assert compiled.edge_arcs == {("b", "c"): (0, 1), ("d", "a"): (0, 1)}


@pytest.mark.parametrize(
    "labels, message",
    [
        ({0: (0, 1)}, "pair per node"),
        ({0: (0,), 1: (1, 2)}, "pair per node"),
        ({0: (0, 1), 1: (1, 2)}, "unique"),
    ],
)
def test_invalid_split_labels_are_rejected(labels, message):
    with pytest.raises(ValueError, match=message):
        split_node_costs(nx.path_graph(2), {}, split=labels)


def test_split_rejects_directed_input_before_changing_it():
    graph = nx.path_graph(2, create_using=nx.DiGraph)
    with pytest.raises(ValueError, match="undirected"):
        split_node_costs(graph, {})
    assert list(graph.edges) == [(0, 1)]


def test_legacy_adapter_preserves_labels_and_missing_edge_cost_default():
    graph = nx.path_graph(2)
    transformed, groups, mapping = node_split_graph(graph, [[0, 1]], {0: 2, 1: 3})
    assert groups == [["0_in", "1_in"]]
    assert mapping == {"0_in": 0, "0_out": 0, "1_in": 1, "1_out": 1}
    assert transformed["0_out"]["1_in"]["weight"] == 0
    assert transformed["0_in"]["0_out"]["weight"] == 2


@pytest.fixture(scope="module", params=["highs", "gurobi"])
def solver(request):
    if request.param == "gurobi":
        gp = pytest.importorskip("gurobipy")
        try:
            with gp.Env(params={"OutputFlag": 0}) as env:
                with gp.Model(env=env):
                    pass
        except gp.GurobiError:
            pytest.skip("Gurobi license is not available")
    return request.param


@pytest.mark.parametrize("label", ["mixed", "tuple"])
def test_legacy_node_identity_collisions_do_not_merge_vertices(label, solver):
    middle = "1" if label == "mixed" else ("1",)
    first = 1 if label == "mixed" else "('1',)"
    graph = nx.path_graph([first, middle, "end"])
    nx.set_edge_attributes(graph, 1, "cost")
    costs = {first: 2, middle: 3, "end": 4}
    transformed, groups, mapping = node_split_graph(
        graph, [[first, "end"]], costs, "cost"
    )
    assert len(transformed) == 6
    assert len(mapping) == 6 and set(mapping.values()) == set(graph)
    assert groups[0][0] != groups[0][1]
    solution = Problem(
        graph,
        variant="node_weighted",
        terminals=[first, "end"],
        node_weights=costs,
        weight="cost",
    ).get_solution(solver=solver)
    assert solution.objective == 11
    assert set(solution.selected_nodes) == set(graph)
    assert solution.gap == 0


@pytest.mark.parametrize("seed", range(4))
def test_composed_splitting_and_group_sinks_match_original_graph_oracle(seed, solver):
    graph = nx.cycle_graph(4)
    graph = nx.relabel_nodes(graph, {0: 1, 1: "1", 2: (2, 3), 3: "end"})
    for index, edge in enumerate(graph.edges):
        graph.edges[edge]["cost"] = (index + seed) % 4 + 1
    costs = {node: (index + seed) % 3 for index, node in enumerate(graph)}
    groups = [[1, "1"], [(2, 3), "end"]]
    best = float("inf")
    for count in range(1, len(graph) + 1):
        for chosen in itertools.combinations(graph, count):
            if "end" not in chosen or any(
                not set(chosen).intersection(g) for g in groups
            ):
                continue
            original = graph.subgraph(chosen)
            if nx.is_connected(original):
                tree = nx.minimum_spanning_tree(original, weight="cost")
                best = min(
                    best,
                    sum(costs[node] for node in chosen)
                    + sum(data["cost"] for _, _, data in tree.edges(data=True)),
                )
    solution = Problem(
        graph, groups=groups, terminals=["end"], node_costs=costs, weight="cost"
    ).get_solution(solver=solver)
    assert solution.objective == best and solution.gap == 0
    selected = nx.Graph(solution.edges)
    selected.add_nodes_from(solution.selected_nodes)
    assert nx.is_connected(selected) and "end" in selected
    assert all(set(selected).intersection(group) for group in groups)
    assert (
        sum(graph.edges[e]["cost"] for e in solution.edges)
        + sum(costs[node] for node in solution.selected_nodes)
        == best
    )
