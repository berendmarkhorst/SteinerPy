"""Public feature combinations checked against original-edge subset oracles."""

import itertools
import math

import networkx as nx
import pytest

from steinerpy import Problem
from .test_composed_budget import optimum
from .test_composed_constraints import degree_optimum
from .test_directed_group_steiner import solver

LIMITS = [
    dict(zip(("max_degree", "hop_limit", "budget"), values))
    for values in itertools.product((None, 1, 2), (None, 2), (None, 2))
]


@pytest.mark.parametrize("limits", LIMITS)
@pytest.mark.parametrize("variant", [None, "group"])
def test_group_modifiers_do_not_require_dummy_node_costs(limits, variant, solver):
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2, weight=0)
    groups = [[0, 1], [2], [3]]
    budget = limits["budget"]
    if budget is None:
        cost = degree_optimum(
            graph, groups, {}, limits["max_degree"] or len(graph), limits["hop_limit"]
        )
        expected = None if math.isinf(cost) else cost
    else:
        expected = optimum(
            graph, groups, {}, budget, limits["max_degree"], limits["hop_limit"], []
        )
    for costs in ({}, {"node_costs": {}}, {"node_costs": None}):
        problem = Problem(graph, groups=groups, variant=variant, **limits, **costs)
        if expected is None:
            with pytest.raises(RuntimeError, match="infeasible"):
                problem.get_solution(solver=solver)
            continue
        result = problem.get_solution(solver=solver)
        assert result.gap == 0
        if budget is None:
            assert result.objective == expected
        else:
            assert (-result.objective, result.total_cost) == expected
            assert result.total_terminals == len(groups)
        selected = nx.Graph()
        selected.add_edges_from(result.edges)
        # Composed results also preserve singleton node selections.
        selected.add_nodes_from(getattr(result, "selected_nodes", ()))
        assert nx.is_connected(selected)
        assert limits["max_degree"] is None or all(
            d <= limits["max_degree"] for _, d in selected.degree()
        )
        assert limits["hop_limit"] is None or len(result.edges) <= limits["hop_limit"]


@pytest.mark.parametrize(
    "option,value", [("max_degree", -1), ("hop_limit", True), ("budget", -1)]
)
def test_group_modifier_validation_does_not_require_dummy_costs(option, value):
    with pytest.raises(ValueError, match="non-negative"):
        Problem(nx.path_graph(3), groups=[[0], [2]], **{option: value})


@pytest.mark.parametrize("limits", LIMITS)
@pytest.mark.parametrize("seed", range(4))
def test_plain_terminal_node_costs_match_original_subset_oracle(limits, seed, solver):
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2)
    nx.set_edge_attributes(
        graph, {e: 1 + (i + seed) % 4 for i, e in enumerate(graph.edges)}, "weight"
    )
    costs = {n: (n * 3 + seed) % 7 for n in graph}
    terminals = [0, 2, 3]
    budget = limits["budget"]
    if budget is None:
        cost = degree_optimum(
            graph,
            [],
            costs,
            limits["max_degree"] or len(graph),
            limits["hop_limit"],
            terminals,
        )
        expected = None if math.isinf(cost) else cost
    else:
        expected = optimum(
            graph,
            [[t] for t in terminals],
            costs,
            budget,
            limits["max_degree"],
            limits["hop_limit"],
            [terminals[0]],
        )
    problem = Problem(graph, terminals=terminals, node_costs=costs, **limits)
    if expected is None:
        with pytest.raises(RuntimeError, match="infeasible"):
            problem.get_solution(solver=solver)
        return
    result = problem.get_solution(solver=solver)
    assert result.gap == 0
    chosen = set(result.selected_nodes)
    assert terminals[0] in chosen
    selected = nx.Graph()
    selected.add_nodes_from(chosen)
    selected.add_edges_from(result.edges)
    assert nx.is_connected(selected) and set(selected) == chosen
    assert all(graph.has_edge(*e) for e in result.edges)
    edge_cost = sum(graph.edges[e]["weight"] for e in result.edges)
    total_cost = edge_cost + sum(costs[n] for n in chosen)
    assert limits["hop_limit"] is None or len(result.edges) <= limits["hop_limit"]
    assert limits["max_degree"] is None or all(
        d <= limits["max_degree"] for _, d in selected.degree()
    )
    if budget is None:
        assert set(terminals) <= chosen
        assert result.objective == total_cost == expected
    else:
        assert (-result.objective, total_cost) == expected
        assert edge_cost <= budget and result.total_cost == total_cost
        assert (
            result.objective
            == result.connected_terminals
            == len(chosen.intersection(terminals))
        )
        assert result.total_terminals == len(terminals)
        assert result.penalties == {
            f"group_0_{t}": 1 for t in terminals if t not in chosen
        }
        assert (
            problem._spec.objective(None).value(
                result.edges, chosen, penalty_count=len(result.penalties)
            )
            == result.objective
        )


@pytest.mark.parametrize("variant", [None, "steiner"])
@pytest.mark.parametrize("syntax", ["flat", "nested", "alias", "positional"])
def test_plain_terminal_cost_aliases_and_snapshot(variant, syntax, solver):
    graph = nx.Graph()
    graph.add_edges_from([(0, "1"), ("1", (2, 3))], cost=2)
    terminals = [0, (2, 3)]
    options = {
        "flat": {"terminals": terminals},
        "nested": {"terminals": [terminals]},
        "alias": {"terminal_groups": [terminals]},
    }
    costs = {0: 3, "1": 5, (2, 3): 7}
    args = ([terminals],) if syntax == "positional" else ()
    problem = Problem(
        graph,
        *args,
        variant=variant,
        node_costs=costs,
        weight="cost",
        **options.get(syntax, {}),
    )
    graph.clear()
    costs.clear()
    result = problem.get_solution(solver=solver)
    assert result.objective == 19 and result.gap == 0
    assert set(result.selected_nodes) == {0, "1", (2, 3)}


@pytest.mark.parametrize("budget", [None, 0])
def test_single_terminal_cost_is_paid_with_zero_limits(budget, solver):
    result = Problem(
        nx.empty_graph(["a", "b"]),
        terminals=["b"],
        node_costs={"b": 9},
        max_degree=0,
        hop_limit=0,
        budget=budget,
    ).get_solution(solver=solver)
    assert result.selected_nodes == ["b"] and not result.edges and result.gap == 0
    assert (result.objective if budget is None else result.total_cost) == 9
    if budget is not None:
        assert result.objective == result.total_terminals == 1


@pytest.mark.parametrize("budget", [None, 0])
def test_disconnected_terminal_cost_requirements(budget, solver):
    problem = Problem(
        nx.empty_graph(2), terminals=[0, 1], node_costs={0: 5, 1: 1}, budget=budget
    )
    if budget is None:
        with pytest.raises(RuntimeError, match="infeasible"):
            problem.get_solution(solver=solver)
    else:
        result = problem.get_solution(solver=solver)
        assert result.selected_nodes == [0] and result.objective == 1
        assert result.total_cost == 5 and result.total_terminals == 2


@pytest.mark.parametrize("budget", [None, 0])
def test_zero_cost_edges_and_repeated_terminals_with_all_limits(budget, solver):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 0, "weight")
    result = Problem(
        graph,
        variant="steiner",
        terminals=[0, 0, 2],
        node_costs={1: 4},
        max_degree=2,
        hop_limit=2,
        budget=budget,
    ).get_solution(solver=solver)
    assert result.gap == 0 and set(result.selected_nodes) == {0, 1, 2}
    assert len(result.edges) == 2
    assert (result.objective if budget is None else result.total_cost) == 4
    if budget is not None:
        assert result.objective == result.total_terminals == 2


@pytest.mark.parametrize(
    "options,match",
    [
        ({"terminals": [0, 2], "node_costs": {1: -1}}, "non-negative"),
        ({"terminals": [0, 2], "node_costs": []}, "must map"),
        ({"terminals": [0, 2], "node_costs": {9: 1}}, "not in the graph"),
        ({"terminals": [0, 2], "variant": "prize_collecting"}, "incompatible"),
    ],
)
def test_plain_node_cost_validation(options, match):
    with pytest.raises((TypeError, ValueError), match=match):
        Problem(nx.path_graph(3), **{"node_costs": {}, **options})
