"""Independent original-graph checks for modular connectivity and costs."""

import importlib.util
import itertools
import math
import random

import networkx as nx
import pytest

from steinerpy import (
    Problem,
    SteinerProblem,
    DirectedSteinerProblem,
    DirectedGroupSteinerProblem,
    HopConstrainedSteinerProblem,
    NodeWeightedSolution,
)
from steinerpy import _composition as composition


def exhaustive_cost(
    graph, groups, costs, terminals=(), weight="weight", hop_limit=None
):
    """Enumerate vertex sets and spanning trees, independently of compilation."""
    best = math.inf
    nodes = list(graph)
    for count in range(1, len(nodes) + 1):
        if hop_limit is not None and count - 1 > hop_limit:
            continue
        for subset in itertools.combinations(nodes, count):
            chosen = set(subset)
            if not set(terminals) <= chosen or any(
                not chosen.intersection(g) for g in groups
            ):
                continue
            induced = graph.subgraph(subset)
            if not nx.is_connected(induced):
                continue
            tree = nx.minimum_spanning_tree(induced, weight=weight)
            cost = sum(data.get(weight, 1) for _, _, data in tree.edges(data=True))
            cost += sum(costs.get(node, 0) for node in subset)
            best = min(best, cost)
    return best


def assert_original_solution(
    graph, groups, costs, solution, terminals=(), weight="weight"
):
    selected = nx.Graph()
    selected.add_nodes_from(solution.selected_nodes)
    selected.add_edges_from(solution.edges)
    assert nx.is_connected(selected)
    assert nx.is_tree(selected)
    assert set(terminals) <= set(selected)
    assert all(set(group).intersection(selected) for group in groups)
    assert all(graph.has_edge(u, v) for u, v in solution.edges)
    cost = sum(graph.edges[e].get(weight, 1) for e in solution.edges)
    cost += sum(costs.get(node, 0) for node in solution.selected_nodes)
    assert cost == pytest.approx(solution.objective)
    assert solution.gap == 0
    assert not solution.was_preprocessed


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
    assert Problem(graph, root=0, terminals=[[0], [1]]).get_solution().objective == 2


def test_node_cost_changes_group_representatives():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("A", "C", 1), ("B", "C", 3)])
    groups = [["A", "B"], ["C"]]
    cheap_edges = Problem(graph, groups=groups, node_costs={}).get_solution()
    paid_nodes = Problem(graph, groups=groups, node_costs={"A": 10}).get_solution()
    assert cheap_edges.objective == 1
    assert paid_nodes.objective == 3
    assert set(paid_nodes.selected_nodes) == {"B", "C"}
    assert isinstance(paid_nodes, NodeWeightedSolution)


@pytest.mark.parametrize(
    "groups, costs, expected_nodes, expected",
    [
        ([[0, 1]], {0: 7, 1: 2}, {1}, 2),
        ([[0], [0]], {0: 3}, {0}, 3),
        ([[0, 1], [1, 2]], {1: 4}, {1}, 4),
        ([[0]], {}, {0}, 0),
    ],
)
def test_isolated_and_overlapping_groups(groups, costs, expected_nodes, expected):
    graph = nx.empty_graph(3)
    sol = Problem(graph, groups=groups, node_costs=costs).get_solution()
    assert sol.objective == expected
    assert set(sol.selected_nodes) == expected_nodes
    assert sol.edges == []


def test_mandatory_terminals_and_costs_paid_once():
    graph = nx.path_graph(4)
    nx.set_edge_attributes(graph, 1, "weight")
    costs = {0: 2, 1: 3, 2: 5, 3: 7}
    sol = Problem(
        graph, terminals=[0, 0], groups=[[2, 3]], node_costs=costs
    ).get_solution()
    assert sol.objective == 12
    assert_original_solution(graph, [[2, 3]], costs, sol, [0])
    assert Problem(graph, terminals=[0], groups=[[2, 3]]).get_solution().objective == 2


def test_group_sinks_cannot_shortcut_original_paths():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1), (1, 2, 100), (2, 3, 1)])
    groups = [[0], [1, 2], [3]]
    sol = Problem(graph, groups=groups, node_costs={}).get_solution()
    assert sol.objective == 102
    assert_original_solution(graph, groups, {}, sol)


def test_disconnected_groups_are_infeasible():
    with pytest.raises(RuntimeError, match="infeasible"):
        Problem(nx.empty_graph(2), groups=[[0], [1]], node_costs={}).get_solution()


def test_virtual_source_cannot_combine_disconnected_components():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 10), (2, 3, 2)])
    groups = [[0, 2], [1, 3]]
    sol = Problem(graph, groups=groups, node_costs={}).get_solution()
    assert sol.objective == 2
    assert_original_solution(graph, groups, {}, sol)


def test_singleton_groups_equal_node_weighted_terminals():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 2, "weight")
    costs = {0: 2, 1: 3, 2: 4}
    sol = Problem(graph, groups=[[0], [2]], node_costs=costs).get_solution()
    legacy = Problem(
        graph, variant="node_weighted", terminals=[0, 2], node_weights=costs
    ).get_solution()
    assert sol.objective == legacy.objective == 13


@pytest.mark.parametrize("seed", range(30))
def test_composition_matches_original_graph_oracle(seed):
    rng = random.Random(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    for u, v in itertools.combinations(graph, 2):
        if rng.random() < 0.5:
            graph.add_edge(u, v, weight=rng.randint(0, 4))
    costs = {node: rng.randint(0, 4) for node in graph}
    groups = [rng.sample(list(graph), rng.randint(1, 3)) for _ in range(3)]
    terminals = [rng.choice(list(graph))] if seed % 2 else []
    expected = exhaustive_cost(graph, groups, costs, terminals)
    kwargs = {"terminals": terminals} if terminals else {}
    problem = Problem(graph, groups=groups, node_costs=costs, **kwargs)
    if math.isinf(expected):
        with pytest.raises(RuntimeError, match="infeasible"):
            problem.get_solution()
    else:
        sol = problem.get_solution()
        assert sol.objective == pytest.approx(expected)
        assert_original_solution(graph, groups, costs, sol, terminals)


def test_custom_weight_default_costs_labels_and_snapshot():
    labels = [1, "1", (1, 2), "__group_0__"]
    graph = nx.Graph()
    graph.add_edge(labels[0], labels[1], cost=2)
    graph.add_edge(labels[1], labels[2])
    graph.add_edge(labels[2], labels[3], cost=3)
    before = graph.copy()
    costs = {labels[1]: 4}
    groups = [[labels[0]], [labels[3]]]
    problem = Problem(
        graph, groups=groups, node_costs=costs, weight="cost", hop_limit=3
    )
    assert nx.utils.graphs_equal(graph, before)
    costs[labels[1]] = 100
    groups.clear()
    graph.clear()
    sol = problem.get_solution()
    assert sol.objective == 10
    assert_original_solution(
        before, [[labels[0]], [labels[3]]], {labels[1]: 4}, sol, weight="cost"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dual_ascent": True},
        {"heavy": True},
        {"variant": "steiner"},
    ],
)
def test_unsupported_compositions(kwargs):
    with pytest.raises(
        (TypeError, ValueError), match="composition|combined|incompatible"
    ):
        Problem(nx.path_graph(3), groups=[[0], [2]], node_costs={}, **kwargs)


@pytest.mark.parametrize("cost", [-1, math.inf, -math.inf, math.nan, "bad"])
def test_invalid_costs_rejected_before_compiling(cost):
    graph = nx.path_graph(2)
    with pytest.raises(ValueError, match="finite and non-negative"):
        Problem(graph, groups=[[0]], node_costs={1: cost})
    nx.set_edge_attributes(graph, cost, "weight")
    with pytest.raises(ValueError, match="finite and non-negative"):
        Problem(graph, groups=[[0]], node_costs={})


@pytest.mark.parametrize(
    "graph", [nx.MultiGraph([(0, 1)]), nx.Graph([(0, 0)])]
)
def test_unsupported_graphs(graph):
    with pytest.raises(ValueError, match="simple graph|self-loops"):
        Problem(graph, groups=[[0]], node_costs={})


def test_unknown_cost_node_and_features():
    with pytest.raises(ValueError, match="not in the graph"):
        Problem(nx.path_graph(2), groups=[[0]], node_costs={3: 1})
    with pytest.raises(ValueError, match="requires groups or terminals"):
        Problem(nx.path_graph(2), node_costs={})
    with pytest.raises(TypeError, match="NetworkX"):
        Problem("not a graph", terminals=[0])
    with pytest.raises(TypeError, match="node_costs must map"):
        Problem(nx.path_graph(2), groups=[[0]], node_costs=[])
    with pytest.raises(TypeError, match="nested collections"):
        Problem(nx.path_graph(2), groups=["01"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "variant": "node_weighted",
            "terminals": [0, 2],
            "node_weights": {},
            "hop_limit": 3,
        },
    ],
)
def test_legacy_transformations_reject_misapplied_constraints(kwargs):
    with pytest.raises(TypeError, match="cannot yet be combined"):
        Problem(nx.path_graph(3), **kwargs)


def test_new_path_restrictions_and_repeated_solves():
    problem = Problem(
        nx.path_graph(2), groups=[[0], [1]], node_costs={}, preprocess=False
    )
    assert problem.get_solution().objective == problem.get_solution().objective == 1
    with pytest.raises(NotImplementedError, match="heuristic-only"):
        problem.get_solution(exact=False)
    assert problem.get_optimal_solutions().exhausted
    with pytest.raises(ValueError, match="Unknown solver"):
        problem.get_solution(solver="typo")
    with pytest.raises(RuntimeError, match="incumbent"):
        problem.get_solution(time_limit=0)


def test_incomplete_feasible_incumbent_keeps_unknown_gap(monkeypatch):
    real_run = composition.mm.run_model

    def incomplete(*args, **kwargs):
        _, runtime, objective, edges, _ = real_run(*args, **kwargs)
        return 0.0, runtime, objective, edges, "incomplete"

    monkeypatch.setattr(composition.mm, "run_model", incomplete)
    sol = Problem(nx.path_graph(2), groups=[[0], [1]], node_costs={}).get_solution()
    assert sol.objective == 1
    assert math.isinf(sol.gap)


def test_invalid_decoded_incumbent_is_rejected():
    problem = Problem(nx.path_graph(2), groups=[[0], [1]], node_costs={})
    spec = problem.implementation.spec
    compiled = composition.compile_groups(spec)
    with pytest.raises(RuntimeError, match="root"):
        composition.decode_groups(spec, compiled, [], 0)
    with pytest.raises(RuntimeError, match="cover"):
        composition.decode_groups(
            spec, compiled, compiled.source_arcs + [next(iter(compiled.node_arcs))], 0
        )


def test_gurobi_composition_matches_oracle():
    if importlib.util.find_spec("gurobipy") is None:
        pytest.skip("gurobipy is not installed")
    import gurobipy as gp

    try:
        with gp.Env(params={"OutputFlag": 0}) as env:
            with gp.Model(env=env):
                pass
    except gp.GurobiError:
        pytest.skip("Gurobi license is not available")
    graph = nx.cycle_graph(4)
    costs = {0: 5, 1: 2, 3: 1}
    groups = [[0, 1], [2, 3]]
    sol = Problem(graph, groups=groups, node_costs=costs).get_solution(solver="gurobi")
    assert sol.objective == exhaustive_cost(graph, groups, costs)
    assert_original_solution(graph, groups, costs, sol)
