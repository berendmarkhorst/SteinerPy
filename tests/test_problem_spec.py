"""Check normalized original-space problem data independently of dispatch."""

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy._problem_spec import normalize_problem

def normalize(graph, terminals=None, variant=None, **options):
    return normalize_problem(graph, None, terminals, variant, options, Problem.variants)

def test_normalization_retains_constraints_before_compatibility_validation():
    graph = nx.path_graph(3)
    options = dict(
        groups=[[0], [2]],
        root=0,
        max_degree=2,
        hop_limit=4,
        budget=10,
        node_budget=3,
        penalty_budget=5,
        preprocess=False,
    )
    spec = normalize_problem(graph, None, None, None, options, Problem.variants)

    assert spec.requested_variant is None
    assert (spec.root, spec.max_degree, spec.hop_limit) == (0, 2, 4)
    assert (spec.budget, spec.node_budget, spec.penalty_budget) == (10, 3, 5)
    assert spec.groups == ((0,), (2,))
    assert spec.options == {"preprocess": False}
    assert options["groups"] == [[0], [2]]
    assert options["root"] == 0

@pytest.mark.parametrize("name", ["node_weights", "node_prizes", "node_costs"])
def test_normalization_does_not_reinterpret_objective_data(name):
    costs = {0: -2, 1: 0, 2: 3}
    spec = normalize(nx.path_graph(3), [0, 2], **{name: costs}, penalty_cost=0)
    assert getattr(spec, name) is costs
    assert spec.penalty_cost == 0
    assert name not in spec.options
    # This stage stores data; support and mathematical validity belong to dispatch.
    assert costs == {0: -2, 1: 0, 2: 3}

def test_rectilinear_geometry_and_explicit_variant_remain_in_specification():
    points = [(0, 0), (1, 1)]
    spec = normalize(None, variant="rectilinear", points=points)
    assert spec.graph is None
    assert spec.points is points
    assert spec.requested_variant == "rectilinear"
