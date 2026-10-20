"""Hop limits refer to original edges after node-cost and group transformations."""

import itertools
import math
import random

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy import _composition as composition
from .test_composition import exhaustive_cost, assert_original_solution
from .test_directed_group_steiner import solver


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("limit", [0, 1, 2, 4])
def test_composed_hops_match_original_graph_oracle(seed, limit, solver):
    rng = random.Random(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    for u, v in itertools.combinations(graph, 2):
        if rng.random() < 0.5:
            graph.add_edge(u, v, weight=rng.randint(0, 4))
    costs = {node: rng.randint(0, 4) for node in graph}
    groups = [rng.sample(list(graph), rng.randint(1, 3)) for _ in range(3)]
    terminals = [rng.choice(list(graph))] if seed % 2 else []
    expected = exhaustive_cost(graph, groups, costs, terminals, hop_limit=limit)
    options = {"terminals": terminals} if terminals else {}
    problem = Problem(
        graph, groups=groups, node_costs=costs, hop_limit=limit, **options
    )
    if math.isinf(expected):
        with pytest.raises(RuntimeError, match="infeasible"):
            problem.get_solution(solver=solver)
    else:
        solution = problem.get_solution(solver=solver)
        assert solution.objective == pytest.approx(expected)
        assert len(solution.edges) <= limit
        assert_original_solution(graph, groups, costs, solution, terminals)


@pytest.mark.parametrize("mode", ["node_costs", "mandatory", "both"])
@pytest.mark.parametrize("limit", [1, 2])
def test_only_original_edges_count_even_when_they_cost_zero(mode, limit, solver):
    graph = nx.Graph()
    graph.add_weighted_edges_from([("r", "a", 0), ("a", "t", 0), ("r", "t", 9)])
    costs = {"r": 2, "a": 1, "t": 3} if mode != "mandatory" else {}
    groups = [["r"], ["t"], ["r", "a"]] if mode != "mandatory" else [["t"]]
    options = {"node_costs": costs} if mode != "mandatory" else {}
    terminals = ["r"] if mode != "node_costs" else []
    if terminals:
        options["terminals"] = terminals
    problem = Problem(graph, groups=groups, hop_limit=limit, **options)
    solution = problem.get_solution(solver=solver)
    assert solution.objective == ({1: 14, 2: 6} if costs else {1: 9, 2: 0})[limit]
    assert len(solution.edges) == limit
    assert_original_solution(graph, groups, costs, solution, terminals)


def test_zero_hops_selects_and_pays_a_common_group_member(solver):
    graph = nx.empty_graph([1, "1", (1, 2), "__group_0__"])
    groups = [[1, "1"], ["1", (1, 2)], ["1", "__group_0__"]]
    costs = {1: 1, "1": 7}
    solution = Problem(
        graph, groups=groups, node_costs=costs, hop_limit=0
    ).get_solution(solver=solver)
    assert solution.objective == 7 and solution.edges == []
    assert solution.selected_nodes == ["1"]
    assert_original_solution(graph, groups, costs, solution)


def test_hop_limit_counts_total_edges_in_a_branching_solution(solver):
    graph = nx.star_graph(3)
    problem = Problem(graph, groups=[[1], [2], [3]], node_costs={}, hop_limit=2)
    with pytest.raises(RuntimeError, match="infeasible"):
        problem.get_solution(solver=solver)
    solution = Problem(
        graph, groups=[[1], [2], [3]], node_costs={}, hop_limit=3
    ).get_solution(solver=solver)
    assert len(solution.edges) == 3 and solution.objective == 3


def test_decoding_checks_original_edge_hop_limit():
    problem = Problem(
        nx.path_graph(2), groups=[[0], [1]], node_costs={0: 2, 1: 3}, hop_limit=0
    )
    spec = problem.implementation.spec
    compiled = composition.compile_groups(spec)
    arcs = set()
    for terminal in compiled.terminals:
        path = nx.shortest_path(compiled.graph, compiled.root, terminal)
        arcs.update(zip(path, path[1:]))
    with pytest.raises(RuntimeError, match="original-edge hop limit"):
        composition.decode_groups(spec, compiled, arcs, 6)


def test_unproven_composed_hop_solution_keeps_unknown_gap(solver, monkeypatch):
    name = "run_model" if solver == "highs" else "run_model_gurobi"
    original = getattr(composition.mm, name)

    def incomplete(*args, **kwargs):
        _, runtime, objective, edges, _ = original(*args, **kwargs)
        return 0, runtime, objective, edges, "incomplete"

    monkeypatch.setattr(composition.mm, name, incomplete)
    solution = Problem(
        nx.path_graph(2), groups=[[0], [1]], node_costs={0: 2, 1: 3}, hop_limit=1
    ).get_solution(solver=solver)
    assert solution.objective == 6 and math.isinf(solution.gap)
