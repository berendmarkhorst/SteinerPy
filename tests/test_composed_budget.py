"""Coverage and cost tie-breaking checked in original graph coordinates."""

import math
import random

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy import _composition_budget as budget_module
from .test_composed_constraints import candidates
from .test_directed_group_steiner import solver


def optimum(graph, groups, costs, budget, degree, hops, terminals):
    best = None
    for edges, nodes in candidates(graph):
        if not set(terminals) <= set(nodes):
            continue
        edge_cost = sum(graph.edges[e].get("weight", 1) for e in edges)
        if edge_cost > budget or (hops is not None and len(edges) > hops):
            continue
        selected = nx.Graph()
        selected.add_nodes_from(nodes)
        selected.add_edges_from(edges)
        if degree is not None and any(d > degree for _, d in selected.degree()):
            continue
        covered = sum(not set(nodes).isdisjoint(g) for g in groups)
        key = (-covered, edge_cost + sum(costs.get(n, 0) for n in nodes))
        best = key if best is None else min(best, key)
    return best


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("budget", [0, 2, 5])
@pytest.mark.parametrize("degree,hops", [(None, None), (1, None), (2, 2), (0, 0)])
def test_budget_composition_matches_two_objective_oracle(
    seed, budget, degree, hops, solver
):
    rng = random.Random(seed)
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2)
    nx.set_edge_attributes(graph, {e: rng.randint(0, 3) for e in graph.edges}, "weight")
    costs = {n: rng.randint(0, 7) for n in graph}
    groups = [[0, 1], [2], [1, 3], [2]]
    terminals = [0] if seed % 2 else []
    expected = optimum(graph, groups, costs, budget, degree, hops, terminals)
    opts = {"terminals": terminals} if terminals else {}
    p = Problem(
        graph,
        groups=groups,
        node_costs=costs,
        budget=budget,
        max_degree=degree,
        hop_limit=hops,
        **opts,
    )
    if expected is None:
        with pytest.raises(RuntimeError, match="infeasible"):
            p.get_solution(solver=solver)
        return
    s = p.get_solution(solver=solver)
    assert (-s.objective, s.total_cost) == expected and s.gap == 0
    selected = nx.Graph()
    selected.add_nodes_from(s.selected_nodes)
    selected.add_edges_from(s.edges)
    assert nx.is_connected(selected)
    assert set(terminals) <= set(selected)
    assert all(graph.has_edge(*e) for e in s.edges)
    assert sum(graph.edges[e]["weight"] for e in s.edges) <= budget
    assert hops is None or len(s.edges) <= hops
    assert degree is None or all(d <= degree for _, d in selected.degree())
    missed = {
        f"group_{i}": 1 for i, g in enumerate(groups) if set(selected).isdisjoint(g)
    }
    assert s.penalties == missed and s.total_terminals == len(groups)
    assert s.objective == s.connected_terminals == len(groups) - len(missed)
    assert s.total_cost == sum(graph.edges[e]["weight"] for e in s.edges) + sum(
        costs[n] for n in selected
    )
    assert (
        p._spec.objective(None).value(
            s.edges, s.selected_nodes, penalty_count=len(missed)
        )
        == s.objective
    )


def test_optional_groups_do_not_force_a_common_component_and_node_costs_do_not_use_edge_budget(
    solver,
):
    g = nx.empty_graph(["a", "b"])
    s = Problem(
        g, groups=[["a"], ["b"]], node_costs={"a": 100, "b": 8}, budget=0
    ).get_solution(solver=solver)
    assert s.objective == 1 and s.total_cost == 8 and s.selected_nodes == ["b"]


def test_mandatory_terminals_remain_required_under_a_budget(solver):
    p = Problem(
        nx.empty_graph(2), groups=[[0], [1]], terminals=[0, 1], node_costs={}, budget=10
    )
    with pytest.raises(RuntimeError, match="infeasible"):
        p.get_solution(solver=solver)


def test_zero_cost_singleton_budget_and_custom_cost_snapshot(solver):
    graph = nx.Graph()
    graph.add_edge(1, "1", cost=0)
    costs = {1: 0, "1": 4}
    p = Problem(
        graph,
        groups=[[1, "1"]],
        node_costs=costs,
        budget=0,
        hop_limit=0,
        max_degree=0,
        weight="cost",
    )
    graph.clear()
    costs[1] = 100
    s = p.get_solution(solver=solver)
    assert (
        s.objective == 1
        and s.total_cost == 0
        and s.selected_nodes == [1]
        and s.gap == 0
    )


@pytest.mark.parametrize("phase", ["primary", "secondary"])
@pytest.mark.parametrize("incumbent", [True, False])
def test_incomplete_budget_phases_do_not_claim_optimality(
    phase, incumbent, solver, monkeypatch
):
    name = "_highs_outcome" if solver == "highs" else "_gurobi_outcome"
    real = getattr(budget_module.mm, name)
    calls = []

    def outcome(*args):
        calls.append(1)
        if len(calls) == (1 if phase == "primary" else 2):
            return "incomplete", incumbent
        return real(*args)

    monkeypatch.setattr(budget_module.mm, name, outcome)
    p = Problem(nx.path_graph(2), groups=[[0], [1]], node_costs={0: 7, 1: 2}, budget=1)
    if phase == "primary" and not incumbent:
        with pytest.raises(RuntimeError, match="incumbent"):
            p.get_solution(solver=solver)
    else:
        s = p.get_solution(solver=solver)
        assert s.objective == 2 and math.isinf(s.gap)


@pytest.mark.parametrize("value", [-1, True, "bad", math.inf, math.nan])
def test_invalid_composed_budget_is_rejected(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        Problem(nx.path_graph(2), groups=[[0], [1]], node_costs={}, budget=value)


@pytest.mark.parametrize("case", ["cost", "coverage", "root", "mandatory", "empty"])
def test_budget_decoder_rejects_invalid_original_results(case):
    from steinerpy._composition import compile_groups

    opts = {"terminals": [0]} if case == "mandatory" else {}
    p = Problem(
        nx.path_graph(2),
        groups=[[0], [1]],
        node_costs={},
        budget=0 if case == "cost" else 1,
        **opts,
    )
    spec = p.implementation.spec
    compiled = compile_groups(spec)
    graph = compiled.graph.copy()
    for arc in compiled.source_arcs[1:]:
        graph.remove_edge(*arc)
    arcs = set()
    for terminal in compiled.terminals:
        path = nx.shortest_path(graph, compiled.root, terminal)
        arcs.update(zip(path, path[1:]))
    if case == "root":
        arcs = set()
    if case in ("mandatory", "empty"):
        arcs = {compiled.source_arcs[0]}
    claimed = 3 if case == "coverage" else 0
    if case == "empty":
        assert budget_module.decode_budget(spec, compiled, arcs, claimed)[-1] == 0
    else:
        with pytest.raises(RuntimeError):
            budget_module.decode_budget(spec, compiled, arcs, claimed)


def test_no_secondary_solve_after_deadline(solver, monkeypatch):
    times = iter((0, 2, 2))
    monkeypatch.setattr(budget_module, "perf_counter", lambda: next(times))
    s = Problem(
        nx.path_graph(2), groups=[[0], [1]], node_costs={}, budget=1
    ).get_solution(solver=solver, time_limit=1)
    assert s.objective == 2 and math.isinf(s.gap)
