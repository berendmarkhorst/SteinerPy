"""Check combined forest requirements against exhaustive original selections."""

import itertools
import math
import random

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy import _selection_model as selection
from .test_directed_group_steiner import solver


def candidates(graph):
    for flags in itertools.product((False, True), repeat=len(graph)):
        nodes = [n for n, flag in zip(graph, flags) if flag]
        possible = list(graph.subgraph(nodes).edges)
        for flags in itertools.product((False, True), repeat=len(possible)):
            edges = [e for e, flag in zip(possible, flags) if flag]
            selected = graph.__class__()
            selected.add_nodes_from(nodes)
            selected.add_edges_from(edges)
            yield selected


def optimum(graph, terminals, groups, costs, budget, degree, hops, root=None):
    best = None
    roots = [root if root is not None else g[0] for g in terminals]
    for selected in candidates(graph):
        if not set(roots) <= set(selected):
            continue
        reached = [
            (
                nx.descendants(selected, r) | {r}
                if graph.is_directed()
                else nx.node_connected_component(selected, r)
            )
            for r in roots
        ]
        terminal_coverage = sum(len(set(g) & c) for g, c in zip(terminals, reached))
        if (budget is None or groups) and terminal_coverage != sum(map(len, terminals)):
            continue
        attached = set().union(*reached)
        if groups:
            components = (
                [reached[0]]
                if graph.is_directed()
                else nx.connected_components(selected)
            )
            possible = [c for c in components if attached | c == set(selected)]
            if not possible:
                continue
            covered = max(sum(bool(set(g) & c) for g in groups) for c in possible)
            if budget is None and covered != len(groups):
                continue
        else:
            if attached != set(selected):
                continue
            covered = sum(
                len((set(g) | {r}) & c) for g, r, c in zip(terminals, roots, reached)
            )
        if degree is not None and any(d > degree for _, d in selected.degree()):
            continue
        if hops is not None and selected.number_of_edges() > hops:
            continue
        edge_cost = sum(graph.edges[e].get("weight", 1) for e in selected.edges)
        if budget is not None and edge_cost > budget:
            continue
        cost = edge_cost + sum(costs.get(n, 0) for n in selected)
        key = cost if budget is None else (-covered, cost)
        best = key if best is None else min(best, key)
    return best


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("with_groups", [False, True])
@pytest.mark.parametrize("budget", [None, 0, 4])
@pytest.mark.parametrize("degree,hops", [(None, None), (2, 3), (1, 1)])
@pytest.mark.parametrize("seed", range(2))
def test_forest_combinations_match_exhaustive_selections(
    directed, with_groups, budget, degree, hops, seed, solver
):
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2), (3, 1)])
    rng = random.Random(seed)
    nx.set_edge_attributes(graph, {e: rng.randrange(3) for e in graph.edges}, "weight")
    costs = {n: rng.randrange(4) for n in graph}
    terminals = [[0, 2], [1, 3]]
    groups = [[0, 1], [2, 3], [3]] if with_groups else []
    root = 0 if directed and with_groups else None
    opts = {"groups": groups} if groups else {}
    if root is not None:
        opts["root"] = root
    p = Problem(
        graph,
        terminals=terminals,
        node_costs=costs,
        budget=budget,
        max_degree=degree,
        hop_limit=hops,
        **opts,
    )
    expected = optimum(graph, terminals, groups, costs, budget, degree, hops, root)
    if expected is None:
        with pytest.raises(RuntimeError, match="infeasible"):
            p.get_solution(solver=solver)
        return
    s = p.get_solution(solver=solver)
    assert (s.objective if budget is None else (-s.objective, s.total_cost)) == expected
    assert s.gap == 0
    assert (
        p._spec.objective(None).value(
            s.edges,
            s.selected_nodes,
            penalty_count=len(s.penalties) if budget is not None else 0,
        )
        == s.objective
    )


def test_shared_node_and_edge_are_paid_once(solver):
    graph = nx.Graph()
    graph.add_edges_from([("a", "u"), ("u", "v"), ("v", "b"), ("c", "v"), ("u", "d")])
    s = Problem(
        graph, terminals=[["a", "b"], ["c", "d"]], node_costs={"u": 4, "v": 3}
    ).get_solution(solver=solver)
    assert s.objective == 12 and len(s.edges) == 5 and s.gap == 0


def test_group_choices_connect_to_each_other_while_forest_can_remain_separate(solver):
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [("a", "b", 1), ("c", "d", 1), ("x", "p", 2), ("y", "q", 6)]
    )
    s = Problem(
        graph, terminals=[["a", "b"], ["c", "d"]], groups=[["x", "y"], ["p", "q"]]
    ).get_solution(solver=solver)
    assert s.objective == 4 and set(s.selected_nodes) == {"a", "b", "c", "d", "x", "p"}


def test_budget_cannot_count_group_members_in_different_forest_components(solver):
    graph = nx.Graph([(0, 1), (2, 3)])
    s = Problem(
        graph, terminals=[[0, 1], [2, 3]], groups=[[0], [2]], budget=2
    ).get_solution(solver=solver)
    assert s.objective == 1 and s.total_cost == 2 and len(s.penalties) == 1


def test_edgeless_forests_keep_isolated_roots_and_snapshot(solver):
    graph = nx.empty_graph([1, "1"])
    costs = {1: 2, "1": 4}
    p = Problem(graph, terminals=[[1], ["1"]], node_costs=costs, budget=0)
    graph.clear()
    costs.clear()
    s = p.get_solution(solver=solver)
    assert s.objective == 2 and s.total_cost == 6 and set(s.selected_nodes) == {1, "1"}


def test_distinct_terminal_labels_with_same_text_keep_separate_penalties(solver):
    s = Problem(
        nx.empty_graph([0, 1, "1", 2]),
        terminals=[[0, 1, "1"], [2]],
        node_costs={},
        budget=0,
    ).get_solution(solver=solver)
    assert s.objective == 2 and s.total_terminals == 4 and len(s.penalties) == 2


@pytest.mark.parametrize("phase", [1, 2])
@pytest.mark.parametrize("incumbent", [False, True])
def test_incomplete_solve_phases_keep_valid_incumbents(
    phase, incumbent, solver, monkeypatch
):
    name = "_highs_outcome" if solver == "highs" else "_gurobi_outcome"
    real = getattr(selection.mm, name)
    calls = []

    def outcome(*args):
        calls.append(1)
        return ("incomplete", incumbent) if len(calls) == phase else real(*args)

    monkeypatch.setattr(selection.mm, name, outcome)
    p = Problem(nx.path_graph(3), terminals=[[0, 1], [1, 2]], node_costs={}, budget=2)
    if phase == 1 and not incumbent:
        with pytest.raises(RuntimeError, match="incumbent"):
            p.get_solution(solver=solver)
    else:
        s = p.get_solution(solver=solver)
        assert s.objective == 4 and math.isinf(s.gap)


@pytest.mark.parametrize(
    "edges,nodes,match",
    [
        ([], [0, 1], "mandatory"),
        ([(0, 2)], [0, 2], "invalid"),
        ([], [0], "root"),
        ([(0, 1)], [0, 1, 3], "disconnected"),
    ],
)
def test_decoder_rejects_invalid_original_selection(edges, nodes, match):
    p = Problem(nx.path_graph(4), terminals=[[0, 1], [1]], node_costs={})
    with pytest.raises(RuntimeError, match=match):
        p.implementation.decode(edges, nodes)
