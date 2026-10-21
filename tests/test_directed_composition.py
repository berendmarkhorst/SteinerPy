"""Directed combinations checked independently in original arc coordinates."""

import itertools
import math
import random

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy import _composition as composition
from steinerpy import _composition_budget as budget_module
from .test_directed_group_steiner import solver


def optimum(graph, root, groups, terminals, costs, budget, degree, hops):
    """Enumerate arc subsets, checking reachability, costs and all active limits."""
    best = None
    demands = groups if groups else [(t,) for t in dict.fromkeys([root] + terminals)]
    required = terminals if groups or budget is None else []
    for flags in itertools.product((False, True), repeat=graph.number_of_edges()):
        arcs = [a for a, flag in zip(graph.edges, flags) if flag]
        selected = nx.DiGraph()
        selected.add_node(root)
        selected.add_edges_from(arcs)
        reached = nx.descendants(selected, root) | {root}
        if set(selected) != reached or not set(required) <= reached:
            continue
        if degree is not None and any(d > degree for _, d in selected.degree()):
            continue
        if hops is not None and len(arcs) > hops:
            continue
        edge_cost = sum(graph.edges[a].get("weight", 1) for a in arcs)
        cost = edge_cost + sum(costs.get(n, 0) for n in reached)
        covered = sum(not reached.isdisjoint(g) for g in demands)
        if budget is None:
            if covered != len(demands):
                continue
            key = cost
        else:
            if edge_cost > budget:
                continue
            key = (-covered, cost)
        best = key if best is None else min(best, key)
    return best


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("with_groups", [False, True])
@pytest.mark.parametrize("degree,hops", [(None, None), (2, 2), (1, None)])
@pytest.mark.parametrize("budget", [None, 0, 4])
def test_directed_combinations_match_original_arc_oracle(
    seed, with_groups, degree, hops, budget, solver
):
    rng = random.Random(seed)
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (3, 1), (2, 0)])
    nx.set_edge_attributes(graph, {a: rng.randint(1, 3) for a in graph.edges}, "weight")
    costs = {n: rng.randint(0, 5) for n in graph}
    groups = [[1, 2], [2, 3], [0, 3]] if with_groups else []
    terminals = [3] if with_groups else [2, 3]
    options = {"groups": groups} if with_groups else {}
    problem = Problem(
        graph,
        root=0,
        terminals=terminals,
        node_costs=costs,
        max_degree=degree,
        hop_limit=hops,
        budget=budget,
        **options,
    )
    expected = optimum(graph, 0, groups, terminals, costs, budget, degree, hops)
    if expected is None:
        with pytest.raises(RuntimeError, match="infeasible"):
            problem.get_solution(solver=solver)
        return
    result = problem.get_solution(solver=solver)
    chosen = set(result.selected_nodes)
    selected = nx.DiGraph()
    selected.add_nodes_from(chosen)
    selected.add_edges_from(result.edges)
    assert 0 in chosen and nx.descendants(selected, 0) | {0} == chosen
    assert set(selected) == chosen and all(graph.has_edge(*a) for a in result.edges)
    assert degree is None or all(d <= degree for _, d in selected.degree())
    assert hops is None or len(result.edges) <= hops
    edge_cost = sum(graph.edges[a]["weight"] for a in result.edges)
    total_cost = edge_cost + sum(costs[n] for n in chosen)
    assert result.gap == 0
    if budget is None:
        assert result.objective == total_cost == expected
    else:
        assert (-result.objective, result.total_cost) == expected
        assert result.total_cost == total_cost and edge_cost <= budget
        demands = groups if with_groups else [[0], [2], [3]]
        covered = sum(not chosen.isdisjoint(g) for g in demands)
        assert result.objective == result.connected_terminals == covered
        assert result.total_terminals == len(demands)
        assert (
            problem._spec.objective(None).value(
                result.edges, chosen, penalty_count=len(result.penalties)
            )
            == result.objective
        )
    if with_groups or budget is None:
        assert set(terminals) <= chosen


@pytest.mark.parametrize("variant", [None, "directed_group"])
@pytest.mark.parametrize("costs", [None, {}])
def test_directed_groups_with_mandatory_terminals_need_no_dummy_costs(
    variant, costs, solver
):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 1), (1, 2, 2)])
    options = {} if costs is None else {"node_costs": costs}
    result = Problem(
        graph,
        root=0,
        groups=[[1]],
        terminals=[2],
        variant=variant,
        max_degree=2,
        hop_limit=2,
        budget=3,
        **options,
    ).get_solution(solver=solver)
    assert result.objective == 1 and result.total_cost == 3 and result.gap == 0
    assert set(result.selected_nodes) == {0, 1, 2}


@pytest.mark.parametrize("budget", [None, 0])
@pytest.mark.parametrize("root_in_terminals", [False, True])
def test_root_cost_singletons_and_plain_budget_counting(
    budget, root_in_terminals, solver
):
    options = {"terminals": [0]} if root_in_terminals else {}
    result = Problem(
        nx.empty_graph(2, create_using=nx.DiGraph),
        root=0,
        node_costs={0: 7},
        budget=budget,
        max_degree=0,
        hop_limit=0,
        **options,
    ).get_solution(solver=solver)
    assert result.selected_nodes == [0] and result.edges == [] and result.gap == 0
    if budget is None:
        assert result.objective == 7
    else:
        assert result.objective == result.total_terminals == 1
        assert result.total_cost == 7


def test_zero_coverage_still_requires_and_pays_directed_root(solver):
    result = Problem(
        nx.empty_graph(2, create_using=nx.DiGraph),
        root=0,
        groups=[[1]],
        node_costs={0: 7},
        budget=0,
        max_degree=0,
        hop_limit=0,
    ).get_solution(solver=solver)
    assert result.objective == 0 and result.total_terminals == 1 and result.gap == 0
    assert result.selected_nodes == [0] and result.total_cost == 7
    assert result.penalties == {"group_0": 1}


@pytest.mark.parametrize("budget,expected", [(None, 5), (2, 13)])
def test_node_costs_change_directed_group_choice_without_using_edge_budget(
    budget, expected, solver
):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("r", "a", 1), ("r", "b", 3)])
    result = Problem(
        graph,
        root="r",
        groups=[["a", "b"]],
        node_costs={"r": 2, "a": 10},
        budget=budget,
        max_degree=1,
        hop_limit=1,
    ).get_solution(solver=solver)
    assert result.gap == 0
    assert (result.objective if budget is None else result.total_cost) == expected
    assert set(result.selected_nodes) == {"r", "b" if budget is None else "a"}
    if budget is not None:
        assert result.objective == 1


def test_directed_plain_budget_does_not_require_first_listed_terminal(solver):
    graph = nx.DiGraph()
    graph.add_edge(0, 2, weight=1)
    graph.add_node(1)
    result = Problem(
        graph, root=0, terminals=[1, 2], node_costs={0: 3}, budget=1
    ).get_solution(solver=solver)
    assert result.objective == 2 and result.total_terminals == 3
    assert result.total_cost == 4 and set(result.selected_nodes) == {0, 2}
    assert result.penalties == {"group_0_1": 1} and result.gap == 0


@pytest.mark.parametrize("budget", [None, 10])
def test_mandatory_terminal_cannot_be_reached_backwards(budget, solver):
    graph = nx.DiGraph([(1, 0)])
    with pytest.raises(RuntimeError, match="infeasible"):
        Problem(
            graph, root=0, groups=[[0]], terminals=[1], node_costs={}, budget=budget
        ).get_solution(solver=solver)


def test_asymmetric_arcs_custom_costs_and_input_snapshots(solver):
    graph = nx.DiGraph()
    graph.add_edge("r", (1, 2), cost=8)
    graph.add_edge((1, 2), "r", cost=1)
    costs = {"r": 3, (1, 2): 5}
    problem = Problem(
        graph,
        root="r",
        terminals=[[(1, 2)]],
        node_costs=costs,
        variant="directed",
        weight="cost",
        hop_limit=1,
        max_degree=1,
    )
    graph.clear()
    costs.clear()
    result = problem.get_solution(solver=solver)
    assert result.edges == [("r", (1, 2))] and result.objective == 16
    assert result.gap == 0


def test_zero_cost_original_arcs_count_but_paid_root_and_group_sinks_do_not(solver):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 0), (1, 2, 0), (0, 2, 9)])
    result = Problem(
        graph,
        root=0,
        groups=[[2]],
        terminals=[1],
        node_costs={0: 4, 1: 2, 2: 3},
        budget=0,
        hop_limit=2,
        max_degree=2,
    ).get_solution(solver=solver)
    assert result.objective == 1 and result.total_cost == 9 and result.gap == 0
    assert set(result.edges) == {(0, 1), (1, 2)}


def test_decoder_rejects_wrong_original_root_and_unpaid_budget_root():
    problem = Problem(nx.DiGraph([(0, 1), (1, 0)]), root=0, groups=[[1]], node_costs={})
    spec = problem.implementation.spec
    compiled = composition.compile_groups(spec)
    # A fabricated source entering node 1 must not pass merely weak connectivity.
    root_arc = next(a for a, n in compiled.node_arcs.items() if n == 1)
    compiled.source_arcs = [(compiled.root, root_arc[0])]
    other_arc = next(a for a, n in compiled.node_arcs.items() if n == 0)
    backward = next(a for a, e in compiled.edge_arcs.items() if e == (1, 0))
    with pytest.raises(RuntimeError, match="connected original"):
        composition.decode_groups(
            spec, compiled, compiled.source_arcs + [root_arc, backward, other_arc], None
        )
    with pytest.raises(RuntimeError, match="mandatory terminals"):
        budget_module.decode_budget(spec, compiled, compiled.source_arcs, 0)


@pytest.mark.parametrize(
    "options,error,match",
    [
        ({}, TypeError, "require root"),
        ({"root": 9}, ValueError, "not in the graph"),
        ({"root": 0, "node_costs": {1: -1}}, ValueError, "non-negative"),
        ({"root": 0, "max_degree": -1}, ValueError, "non-negative integer"),
        ({"root": 0, "hop_limit": True}, ValueError, "non-negative integer"),
        ({"root": 0, "budget": math.inf}, ValueError, "finite and non-negative"),
        ({"root": 0, "variant": "steiner"}, TypeError, "incompatible"),
    ],
)
def test_directed_composition_validation(options, error, match):
    with pytest.raises(error, match=match):
        Problem(nx.DiGraph([(0, 1)]), **{"terminals": [1], "node_costs": {}, **options})


def test_directed_multigraph_and_self_loops_remain_rejected():
    for graph in (nx.MultiDiGraph([(0, 1)]), nx.DiGraph([(0, 0)])):
        with pytest.raises(ValueError, match="simple graph|self-loops"):
            Problem(graph, root=0, terminals=[0], node_costs={})
