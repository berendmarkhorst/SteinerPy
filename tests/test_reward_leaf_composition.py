"""Independent subset oracle for rewards, budgets and leaf constraints."""

import math

import networkx as nx
import pytest

from steinerpy import Problem
from .test_forest_composition import candidates
from .test_directed_group_steiner import solver


def optimum(
    graph,
    terminals,
    groups,
    costs,
    prizes,
    weights,
    root,
    budget,
    node_budget,
    leaves,
    degree,
    hops,
    penalty,
    penalty_budget=None,
    all_optima=False,
):
    best = None
    selections = set()
    coverage_mode = (
        budget is not None and prizes is None and weights is None and penalty is None
    )
    optional = coverage_mode or penalty is not None
    roots = [root if root is not None else g[0] for g in terminals]
    if not roots and root is not None:
        roots = [root]
        terminals = [[root]]
    for selected in candidates(graph):
        if not set(roots) <= set(selected):
            continue

        def reach(r):
            return (
                nx.descendants(selected, r) | {r}
                if graph.is_directed()
                else nx.node_connected_component(selected, r)
            )

        reached = [reach(r) for r in roots]
        missed = sum(
            len((set(g) | {r}) - c) for g, r, c in zip(terminals, roots, reached)
        )
        if missed and (not optional or groups):
            continue
        attached = set().union(*reached)
        total = sum(len(set(g) | {r}) for g, r in zip(terminals, roots))
        if groups or not roots:
            fixed = root if root is not None else roots[0] if len(roots) == 1 else None
            components = (
                [reach(fixed)] if fixed is not None else [reach(n) for n in selected]
            )
            if not selected and not groups:
                components = [set()]
            components = [c for c in components if attached | c == set(selected)]
            if not components:
                continue
            missed = min(sum(not (set(g) & c) for g in groups) for c in components)
            total = len(groups)
            if missed and not optional:
                continue
        elif attached != set(selected):
            continue
        if penalty_budget is not None and missed > penalty_budget:
            continue
        if any(selected.degree(n) > 1 for n in leaves if n in selected):
            continue
        if degree is not None and any(d > degree for _, d in selected.degree()):
            continue
        if hops is not None and selected.number_of_edges() > hops:
            continue
        edge_cost = sum(graph.edges[e].get("weight", 1) for e in selected.edges)
        node_cost = sum(costs.get(n, 0) for n in selected)
        if budget is not None and edge_cost > budget:
            continue
        if node_budget is not None and node_cost > node_budget:
            continue
        cost = edge_cost + node_cost
        if weights is not None:
            objective = -sum(weights.get(n, 0) for n in selected)
        elif prizes is not None:
            objective = cost + sum(prizes.get(n, 0) for n in graph if n not in selected)
        else:
            objective = cost
        objective += (penalty or 0) * missed
        key = (-(total - missed), cost) if coverage_mode else objective
        if best is None or key < best:
            best, selections = key, set()
        if key == best:
            edges = frozenset(
                tuple(e) if graph.is_directed() else frozenset(e)
                for e in selected.edges
            )
            selections.add((edges, frozenset(selected)))
    return (best, selections) if all_optima else best


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize(
    "connectivity", ["tree", "forest", "groups", "mixed", "unrooted"]
)
@pytest.mark.parametrize("reward", ["prizes", "weights", "none"])
@pytest.mark.parametrize("penalty", [None, 3])
@pytest.mark.parametrize("limits", [False, True])
def test_reward_and_resource_combinations_match_original_subsets(
    directed, connectivity, reward, penalty, limits, solver
):
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (1, 2, 0), (2, 3, 2), (0, 2, 4), (3, 1, 1)]
    )
    terminals = (
        [[0, 2]]
        if connectivity == "tree"
        else [[0, 2], [1, 3]] if connectivity in {"forest", "mixed"} else []
    )
    groups = [[0, 1], [2, 3]] if connectivity in {"groups", "mixed"} else []
    costs = {0: 1, 1: 2, 2: 2, 3: 1}
    prizes = {0: 0, 1: 1, 2: 4, 3: 6} if reward == "prizes" else None
    weights = {0: -1, 1: -3, 2: 4, 3: 5} if reward == "weights" else None
    root = 0 if directed and connectivity != "unrooted" else None
    if connectivity == "unrooted" and reward == "none":
        root = 0
    budget, node_budget, degree, hops = (
        (3, 5, 2, 3) if limits else (None, None, None, None)
    )
    leaves = (
        [2] if limits and connectivity in {"tree", "forest", "groups", "mixed"} else []
    )
    opts = {
        "root": root,
        "node_costs": costs,
        "node_budget": node_budget,
        "budget": budget,
        "max_degree": degree,
        "hop_limit": hops,
        "partial_terminals": leaves,
        "penalty_cost": penalty,
    }
    if terminals:
        opts["terminals"] = terminals
    if groups:
        opts["groups"] = groups
    if prizes is not None:
        opts["node_prizes"] = prizes
    if weights is not None:
        opts["node_weights"] = weights
    expected = optimum(
        graph,
        terminals,
        groups,
        costs,
        prizes,
        weights,
        root,
        budget,
        node_budget,
        leaves,
        degree,
        hops,
        penalty,
    )
    p = Problem(graph, **opts)
    if expected is None:
        with pytest.raises(RuntimeError, match="infeasible"):
            p.get_solution(solver=solver)
        return
    s = p.get_solution(solver=solver)
    actual = (
        (-s.objective, s.total_cost)
        if p.implementation.coverage_mode
        else -s.objective if weights is not None else s.objective
    )
    assert actual == expected and s.gap == 0
    assert (
        p._spec.objective(None).value(
            s.edges, s.selected_nodes, penalty_count=len(s.penalties)
        )
        == s.objective
    )


def test_reward_cannot_be_collected_from_a_disconnected_island(solver):
    graph = nx.Graph([(0, 1), (2, 3)])
    s = Problem(
        graph, terminals=[0, 1], node_prizes={3: 1000}, node_costs={}
    ).get_solution(solver=solver)
    assert s.objective == 1001 and set(s.selected_nodes) == {0, 1}


@pytest.mark.parametrize("directed", [False, True])
def test_all_negative_unrooted_weights_can_choose_empty_solution(directed, solver):
    graph = nx.path_graph(3, create_using=nx.DiGraph if directed else nx.Graph)
    s = Problem(graph, node_weights={0: -4, 1: -2, 2: -1}).get_solution(solver=solver)
    assert s.objective == 0 and s.selected_nodes == [] and s.gap == 0


@pytest.mark.parametrize("variant", ["partial_terminal", "full_terminal"])
def test_named_leaf_variants_use_original_edge_budget(variant, solver):
    opts = {"partial_terminals": [2]} if variant == "partial_terminal" else {}
    s = Problem(
        nx.path_graph(3), terminals=[0, 2], variant=variant, budget=2, **opts
    ).get_solution(solver=solver)
    assert s.objective == 2 and s.total_cost == 2


def test_signed_connector_weight_and_independent_budgets(solver):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 10, "weight")
    opts = dict(
        root=0, node_weights={0: 5, 1: -7, 2: 10}, node_costs={0: 1, 1: 2, 2: 1}
    )
    s = Problem(graph, budget=20, node_budget=4, **opts).get_solution(solver=solver)
    assert (
        s.objective == 8 and set(s.selected_nodes) == {0, 1, 2} and s.total_cost == 24
    )
    assert (
        Problem(graph, budget=19, node_budget=4, **opts)
        .get_solution(solver=solver)
        .objective
        == 5
    )
    assert (
        Problem(graph, budget=20, node_budget=3, **opts)
        .get_solution(solver=solver)
        .objective
        == 5
    )


def test_penalty_budget_caps_missed_requirements(solver):
    graph = nx.path_graph(4)
    opts = dict(terminals=[0, 2, 3], node_prizes={}, penalty_cost=0)
    s = Problem(graph, penalty_budget=1, **opts).get_solution(solver=solver)
    assert s.objective == 2 and len(s.penalties) == 1
    assert (
        Problem(graph, penalty_budget=0, **opts).get_solution(solver=solver).objective
        == 3
    )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"node_prizes": {0: -1}}, "non-negative"),
        ({"node_weights": {0: math.nan}}, "finite"),
        ({"node_weights": {99: 1}}, "not in the graph"),
        ({"node_prizes": {}, "node_weights": {}}, "not both"),
        ({"node_prizes": None}, "must map"),
        ({"node_weights": []}, "must map"),
        ({"node_budget": 1}, "requires node_costs"),
        ({"node_budget": -1}, "node_budget"),
        ({"penalty_cost": math.inf}, "penalty_cost"),
        ({"penalty_budget": True}, "penalty_budget"),
        ({"partial_terminals": [1]}, "must be terminals"),
    ],
)
def test_new_feature_validation(options, match):
    with pytest.raises((ValueError, TypeError), match=match):
        Problem(nx.path_graph(3), terminals=[0, 2], **options)


def test_new_feature_data_is_snapshotted(solver):
    graph = nx.path_graph(3)
    prizes, costs, leaves = {2: 10}, {1: 4}, [2]
    p = Problem(
        graph,
        terminals=[0, 2],
        node_prizes=prizes,
        node_costs=costs,
        partial_terminals=leaves,
    )
    graph.clear()
    prizes.clear()
    costs.clear()
    leaves.clear()
    assert p.get_solution(solver=solver).objective == 6


@pytest.mark.parametrize("variant", [None, "group"])
def test_explicit_compatible_variant_does_not_drop_rewards(variant, solver):
    s = Problem(
        nx.path_graph(3),
        variant=variant,
        groups=[[0]],
        node_costs={},
        node_prizes={2: 10},
    ).get_solution(solver=solver)
    assert s.objective == 2 and set(s.selected_nodes) == {0, 1, 2}


def test_leaf_generator_is_consumed_only_once(solver):
    p = Problem(
        nx.path_graph(3), terminals=[0, 1, 2], partial_terminals=(n for n in [1])
    )
    with pytest.raises(RuntimeError, match="infeasible"):
        p.get_solution(solver=solver)


def test_named_partial_terminal_still_requires_leaf_list():
    with pytest.raises(TypeError, match="requires partial_terminals"):
        Problem(
            nx.path_graph(3), terminals=[0, 2], variant="partial_terminal", budget=2
        )


def test_original_objective_mismatch_is_not_reported_as_optimal(solver, monkeypatch):
    from steinerpy._selection_model import SelectionModel

    real = SelectionModel.objective_value
    monkeypatch.setattr(
        SelectionModel, "objective_value", lambda model: real(model) + 1
    )
    with pytest.raises(RuntimeError, match="objective disagrees"):
        Problem(nx.path_graph(3), terminals=[0, 2], node_prizes={}).get_solution(
            solver=solver
        )


def test_budget_tie_breaking_does_not_restart_expired_time_allowance(
    solver, monkeypatch
):
    from steinerpy import _selection_model

    clock = [0]
    monkeypatch.setattr(_selection_model, "perf_counter", lambda: clock[0])
    original = _selection_model.SelectionModel._optimize

    def optimize(model, objective, deadline):
        result = original(model, objective, deadline)
        clock[0] = 2
        return result

    monkeypatch.setattr(_selection_model.SelectionModel, "_optimize", optimize)
    s = Problem(
        nx.path_graph(3), terminals=[[0, 1], [1, 2]], node_costs={}, budget=2
    ).get_solution(solver=solver, time_limit=1)
    assert s.objective == 4 and math.isinf(s.gap)
