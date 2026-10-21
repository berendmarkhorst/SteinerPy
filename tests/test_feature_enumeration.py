"""Compare the entire optimum pool, including preprocessing, to subset oracles."""

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy import _selection_model as selection
from steinerpy._feature_preprocessing import feasible_edges
from .test_reward_leaf_composition import optimum
from .test_directed_group_steiner import solver


def signature(graph, solution):
    return (
        frozenset(
            tuple(e) if graph.is_directed() else frozenset(e) for e in solution.edges
        ),
        frozenset(solution.selected_nodes),
    )


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("connectivity", ["tree", "forest", "groups", "mixed"])
@pytest.mark.parametrize("reward", [None, "prizes", "weights"])
@pytest.mark.parametrize("budget", [None, 1])
@pytest.mark.parametrize("preprocess", [False, True])
def test_all_minimal_optima_match_independent_original_selections(
    directed, connectivity, reward, budget, preprocess, solver
):
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 0), (1, 2, 1), (0, 2, 1)])
    terminals = (
        [[0, 2]]
        if connectivity == "tree"
        else [[0, 1], [2]] if connectivity in {"forest", "mixed"} else []
    )
    groups = [[0, 1], [2]] if connectivity in {"groups", "mixed"} else []
    root = 0 if directed and connectivity != "mixed" else None
    costs = {0: 1, 1: 0, 2: 1}
    prizes = {1: 1} if reward == "prizes" else None
    weights = {0: 1, 1: -1, 2: 2} if reward == "weights" else None
    opts = dict(
        node_costs=costs,
        node_budget=2,
        budget=budget,
        max_degree=2,
        hop_limit=2,
        partial_terminals=[2],
        preprocess=preprocess,
    )
    if root is not None:
        opts["root"] = root
    if terminals:
        opts["terminals"] = terminals
    if groups:
        opts["groups"] = groups
    if prizes is not None:
        opts["node_prizes"] = prizes
    if weights is not None:
        opts["node_weights"] = weights
    best, selections = optimum(
        graph,
        terminals,
        groups,
        costs,
        prizes,
        weights,
        root,
        budget,
        2,
        [2],
        2,
        2,
        None,
        all_optima=True,
    )
    minimal = {
        s
        for s in selections
        if not any(t != s and t[0] <= s[0] and t[1] <= s[1] for t in selections)
    }
    p = Problem(graph, **opts)
    pool = p.get_optimal_solutions(limit=100, solver=solver)
    assert pool.exhausted
    assert {signature(graph, s) for s in pool} == minimal
    assert len(pool) == len(minimal)
    for s in pool:
        actual = (
            (-s.objective, s.total_cost)
            if p.implementation.coverage_mode
            else -s.objective if weights is not None else s.objective
        )
        assert actual == best and s.gap == 0


@pytest.mark.parametrize("budget", [None, 1])
def test_plain_group_api_enumerates_without_dummy_features(budget, solver):
    graph = nx.cycle_graph(4)
    p = Problem(graph, groups=[[0, 1], [2, 3]], budget=budget)
    graph.clear()
    pool = p.get_optimal_solutions(solver=solver)
    assert pool.exhausted and len(pool) == 2
    assert all(s.objective == (1 if budget is None else 2) for s in pool)


def test_plain_forest_budget_api_enumerates_without_dummy_features(solver):
    p = Problem(
        nx.cycle_graph(4), terminals=[[0, 2], [1, 3]], budget=3, preprocess=False
    )
    pool = p.get_optimal_solutions(solver=solver)
    assert pool.exhausted and len(pool) == 4
    assert all(s.objective == 4 and s.total_cost == 3 for s in pool)


def test_zero_sum_reward_branch_is_not_a_distinct_minimal_optimum(solver):
    p = Problem(nx.path_graph(3), root=0, node_weights={0: 1, 1: -2, 2: 2})
    pool = p.get_optimal_solutions(solver=solver)
    assert pool.exhausted and len(pool) == 1 and pool.solutions[0].selected_nodes == [0]


def test_empty_optimum_and_edgeless_group_choices(solver):
    pool = Problem(nx.empty_graph(3), node_weights={}).get_optimal_solutions(
        solver=solver
    )
    assert pool.exhausted and len(pool) == 1 and pool.solutions[0].selected_nodes == []
    pool = Problem(
        nx.empty_graph([1, "1"]), groups=[[1, "1"]], node_costs={}
    ).get_optimal_solutions(solver=solver)
    assert pool.exhausted and len(pool) == 2


@pytest.mark.parametrize("limit", [0, 1])
def test_limit_does_not_claim_exhaustion(limit, solver):
    pool = Problem(
        nx.cycle_graph(4), terminals=[0, 2], node_costs={}
    ).get_optimal_solutions(limit=limit, solver=solver)
    assert len(pool) == limit and not pool.exhausted


@pytest.mark.parametrize("phase", [1, 2, 3])
@pytest.mark.parametrize("incumbent", [True, False])
def test_incomplete_enumeration_never_claims_exhaustion(
    phase, incumbent, solver, monkeypatch
):
    name = "_highs_outcome" if solver == "highs" else "_gurobi_outcome"
    real = getattr(selection.mm, name)
    calls = []

    def outcome(*args):
        calls.append(1)
        return ("incomplete", incumbent) if len(calls) == phase else real(*args)

    monkeypatch.setattr(selection.mm, name, outcome)
    pool = Problem(
        nx.cycle_graph(4), terminals=[0, 2], node_costs={}
    ).get_optimal_solutions(solver=solver)
    assert not pool.exhausted and len(pool) == (1 if phase == 3 else 0)


@pytest.mark.parametrize("limit", [-1, True, "2", 1.5])
def test_invalid_enumeration_limits(limit):
    with pytest.raises(ValueError, match="non-negative integer"):
        Problem(nx.path_graph(2), node_weights={}).get_optimal_solutions(limit=limit)


def test_preprocessing_only_removes_edges_forbidden_by_original_resources(solver):
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 2, 10), (1, 3, 0)])
    p = Problem(
        graph,
        terminals=[0, 2],
        node_costs={3: 10},
        node_budget=1,
        budget=2,
        preprocess=True,
        enumeration_safe=True,
    )
    assert set(feasible_edges(p._spec)) == {(0, 1), (1, 2)}
    assert graph.number_of_edges() == p._spec.graph.number_of_edges() == 4
    s = p.get_solution(solver=solver)
    assert s.objective == 2 and s.total_cost == 2 and s.gap == 0
    pool = p.get_optimal_solutions(solver=solver)
    assert pool.exhausted and len(pool) == 1


def test_preprocessing_keeps_required_unaffordable_nodes_infeasible(solver):
    p = Problem(
        nx.path_graph(2),
        terminals=[0, 1],
        node_costs={0: 5},
        node_budget=1,
        preprocess=True,
    )
    with pytest.raises(RuntimeError, match="infeasible"):
        p.get_solution(solver=solver)
    pool = p.get_optimal_solutions(solver=solver)
    assert pool.exhausted and len(pool) == 0


@pytest.mark.parametrize("preprocess", [False, True])
def test_solving_and_enumeration_share_penalty_labels(preprocess, solver):
    for terminals in ([0, 1, 2], [0, 1, "1"]):
        p = Problem(
            nx.empty_graph(terminals),
            terminals=terminals,
            node_costs={},
            budget=0,
            preprocess=preprocess,
        )
        s = p.get_solution(solver=solver)
        pool = p.get_optimal_solutions(solver=solver)
        assert s.objective == 1 and len(s.penalties) == 2
        assert (
            pool.exhausted
            and len(pool) == 1
            and pool.solutions[0].penalties == s.penalties
        )


def test_directed_groups_and_forest_need_not_share_a_global_root(solver):
    graph = nx.DiGraph([(0, 1), (2, 3), (4, 5)])
    p = Problem(graph, terminals=[[0, 1], [2, 3]], groups=[[4], [5]], preprocess=True)
    s = p.get_solution(solver=solver)
    assert s.objective == 3 and set(s.selected_nodes) == set(graph)
    pool = p.get_optimal_solutions(solver=solver)
    assert pool.exhausted and len(pool) == 1


def test_explicit_none_node_costs_do_not_drop_full_terminal_leaves(solver):
    p = Problem(
        nx.path_graph(3), terminals=[0, 1, 2], variant="full_terminal", node_costs=None
    )
    with pytest.raises(RuntimeError, match="infeasible"):
        p.get_solution(solver=solver)


def test_rectilinear_rejects_graph_features_before_composition():
    with pytest.raises(TypeError, match="Unsupported option.*node_costs"):
        Problem(variant="rectilinear", points=[(0, 0), (1, 1)], node_costs={})
