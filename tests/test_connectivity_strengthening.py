"""Connectivity witnesses must tighten bounds without excluding selections."""

from time import perf_counter

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy._feature_composition import FeatureComposition
from steinerpy._selection_model import SelectionModel
from .test_directed_group_steiner import solver


def test_triangle_relaxation_requires_two_edges():
    # Unit terminal flows alone allow half of each edge, costing 1.5.
    problem = Problem(nx.cycle_graph(3), terminals=[0, 1, 2], node_costs={})
    strategy = FeatureComposition(problem._spec)
    model = SelectionModel(problem._spec, "highs", 5, "", None)
    try:
        strategy.configure(model)
        model.strengthen_connectivity()
        model.model.setOptionValue("solve_relaxation", True)
        model.model.minimize(model.cost)
        assert (
            model.model.modelStatusToString(model.model.getModelStatus()) == "Optimal"
        )
        assert model.objective_value() == pytest.approx(2)
    finally:
        model.close()


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize(
    "scenario", ["cycle", "forest", "groups-forest", "optional-root", "empty"]
)
def test_feasible_original_selections_survive_including_cycles_and_separate_roots(
    directed, scenario, solver
):
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (4, 5)])
    options = dict(node_costs={n: 0 for n in graph}, preprocess=True)
    if scenario == "cycle":
        options.update(terminals=[0, 3], root=0, node_prizes={1: 2}, hop_limit=5)
        edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
        nodes = [0, 1, 2, 3]
    elif scenario in {"forest", "groups-forest"}:
        options.update(terminals=[[0, 3], [4, 5]], node_budget=0, max_degree=3)
        if scenario == "groups-forest":
            options["groups"] = [[0, 1], [2, 3]]
        edges, nodes = [(0, 1), (1, 2), (2, 3), (4, 5)], list(graph)
    else:
        options.update(node_weights={n: 0 for n in graph}, budget=0)
        edges, nodes = [], ([1] if scenario == "optional-root" else [])
    problem = Problem(graph, **options)
    strategy = FeatureComposition(problem._spec)
    strategy.decode(edges, nodes)
    model = SelectionModel(problem._spec, solver, 5, "", None)
    try:
        strategy.configure(model)
        model.strengthen_connectivity()
        selected = set(edges) if directed else {frozenset(e) for e in edges}
        for e, variable in model.edges.items():
            key = e if directed else frozenset(e)
            model.add(variable == int(key in selected))
        for n, variable in model.nodes.items():
            model.add(variable == int(n in nodes))
        status, incumbent = model.solve(0 * model.zero, perf_counter() + 5)
        assert status == "optimal" and incumbent
    finally:
        model.close()


@pytest.mark.parametrize("second", ["missing", "worse", "optimal"])
def test_adaptive_solve_retains_incumbent_values_and_clears_fallback(
    monkeypatch, second, solver
):
    problem = Problem(nx.path_graph(3), terminals=[0, 2], node_costs={1: 2})
    strategy = FeatureComposition(problem._spec)
    model = SelectionModel(problem._spec, solver, 5, "", None)
    try:
        coverage = strategy.configure(model)
        original = model._optimize
        calls = []

        def optimize(objective, deadline):
            calls.append(deadline)
            if len(calls) == 1:
                assert original(objective, deadline) == ("optimal", True)
                return "incomplete", True
            if second == "missing":
                return "incomplete", False
            return original(objective + (10 if second == "worse" else 0), deadline)

        monkeypatch.setattr(model, "_optimize", optimize)
        monkeypatch.setattr(model, "_time_limit_reached", lambda: True)
        deadline = perf_counter() + 5
        status, incumbent = model.solve(model.cost, deadline)
        assert len(calls) == 2 and calls[0] < calls[1] == deadline
        assert incumbent and status == (
            "optimal" if second == "optimal" else "incomplete"
        )
        assert model.objective_value() == pytest.approx(4)
        assert sum(model.value(v) for v in coverage) == 2
        assert strategy.decode(*model.selection())[2] == 4
        assert model._connectivity_roots is None
        monkeypatch.setattr(model, "_optimize", original)
        assert model.solve(model.cost, perf_counter() + 5) == ("optimal", True)
        assert model._fallback is None
    finally:
        model.close()


@pytest.mark.parametrize(
    "remaining,outcome,elapsed,expected",
    [
        (1, ("incomplete", False), 0, 1),
        (5, ("optimal", True), 0, 1),
        (5, ("infeasible", False), 0, 1),
        (5, ("incomplete", False), 6, 1),
        (5, ("incomplete", False), 1, 2),
    ],
)
def test_adaptive_solve_respects_deadline_and_does_not_restart_finished_solves(
    monkeypatch, remaining, outcome, elapsed, expected
):
    from steinerpy import _selection_model as selection

    model = SelectionModel.__new__(SelectionModel)
    model._connectivity_roots = {}
    model._fallback = None
    clock = [100]
    calls, strengthened = [], []
    monkeypatch.setattr(selection, "perf_counter", lambda: clock[0])

    def optimize(objective, deadline):
        calls.append(deadline)
        clock[0] += elapsed
        return outcome

    model._optimize = optimize
    model._time_limit_reached = lambda: True
    model.strengthen_connectivity = lambda: strengthened.append(True)
    assert model.solve(None, 100 + remaining) == outcome
    assert len(calls) == expected and len(strengthened) == expected - 1
    assert calls[-1] <= 100 + remaining
