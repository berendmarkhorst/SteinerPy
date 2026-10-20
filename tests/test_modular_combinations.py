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
    for values in itertools.product((None, 1), (None, 2), (None, 2))
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


@pytest.mark.parametrize("option,value", [("max_degree", -1), ("hop_limit", True), ("budget", -1)])
def test_group_modifier_validation_does_not_require_dummy_costs(option, value):
    with pytest.raises(ValueError, match="non-negative"):
        Problem(nx.path_graph(3), groups=[[0], [2]], **{option: value})
