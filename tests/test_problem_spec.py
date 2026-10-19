"""Check the boundary between normalized data and implementation construction."""

from dataclasses import replace

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy._composition import GroupComposition
from steinerpy._problem_dispatch import build_problem
from steinerpy._problem_spec import normalize_problem
from steinerpy.objects import DirectedSteinerProblem, HopConstrainedSteinerProblem


def normalize(graph, terminals=None, variant=None, **options):
    return normalize_problem(graph, None, terminals, variant, options, Problem.variants)


def test_dispatch_uses_specification_fields_and_can_reuse_input():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 2), (1, 2, 3)])
    terminals = (node for node in [2, 2])
    spec = normalize(graph, terminals, root=0, preprocess=False)

    _, variant, first = build_problem(spec)
    _, _, second = build_problem(spec)
    _, _, rerooted = build_problem(replace(spec, root=1))

    assert variant == "directed"
    assert first.get_solution().objective == second.get_solution().objective == 5
    assert rerooted.get_solution().objective == 3
    assert spec.root == 0
    assert spec.terminal_groups == ((2,),)
    assert spec.options == {"preprocess": False}


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
    with pytest.raises(TypeError, match="cannot yet be combined"):
        build_problem(spec)


@pytest.mark.parametrize("name", ["node_weights", "node_prizes", "node_costs"])
def test_normalization_does_not_reinterpret_objective_data(name):
    costs = {0: -2, 1: 0, 2: 3}
    spec = normalize(nx.path_graph(3), [0, 2], **{name: costs}, penalty_cost=0)
    assert getattr(spec, name) is costs
    assert spec.penalty_cost == 0
    assert name not in spec.options
    # This stage stores data; support and mathematical validity belong to dispatch.
    assert costs == {0: -2, 1: 0, 2: 3}


def test_missing_required_data_is_not_replaced_by_spec_defaults():
    spec = normalize(nx.path_graph(3), [0, 2], variant="node_weighted")
    assert spec.node_weights is None
    with pytest.raises(TypeError, match="missing a required argument: 'node_weights'"):
        build_problem(spec)


def test_omitted_and_explicit_none_keep_different_dispatch_paths():
    graph = nx.path_graph(3).to_directed()
    plain = normalize(graph, [2], root=0)
    explicit_none = normalize(graph, [2], root=0, hop_limit=None)
    assert plain.hop_limit is explicit_none.hop_limit is None
    assert type(build_problem(plain)[2]) is DirectedSteinerProblem
    assert type(build_problem(explicit_none)[2]) is HopConstrainedSteinerProblem

    groups = [[0], [2]]
    legacy = normalize(nx.path_graph(3), groups=groups)
    composed = normalize(nx.path_graph(3), groups=groups, node_costs=None)
    assert build_problem(legacy)[1] == "group"
    assert isinstance(build_problem(composed)[2], GroupComposition)


def test_composition_snapshots_original_space_and_keeps_specification():
    graph = nx.path_graph(3)
    costs = {1: 4}
    source = normalize(graph, groups=[[0], [2]], node_costs=costs, weight="cost")
    compiled_spec, variant, strategy = build_problem(source)

    assert source.graph is graph
    assert source.node_costs is costs
    assert compiled_spec is strategy.spec
    assert compiled_spec.graph is not graph
    assert compiled_spec.node_costs is not costs
    assert compiled_spec.weight == "cost"
    assert variant is None
    graph.clear()
    costs.clear()
    assert strategy.get_solution().objective == 6


def test_normalization_keeps_unknown_options_for_dispatch_errors():
    spec = normalize(nx.path_graph(3), [0, 2], zebra=True, alpha=True)
    with pytest.raises(TypeError) as error:
        build_problem(spec)
    assert str(error.value) == "Unsupported option(s) for steiner: alpha, zebra"


@pytest.mark.parametrize(
    "graph, kwargs, error_type, message",
    [
        (None, {"variant": "bad"}, ValueError, "Unknown variant"),
        ("invalid", {"terminals": [], "terminal_groups": []}, TypeError, "not both"),
        (None, {"terminals": []}, TypeError, "Problem requires graph"),
        (
            None,
            {"variant": "directed", "terminals": []},
            TypeError,
            "directed requires graph",
        ),
        (
            nx.path_graph(3),
            {"terminals": [99], "root": 99},
            ValueError,
            "Required vertex",
        ),
    ],
)
def test_public_validation_order_is_preserved(graph, kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        Problem(graph, **kwargs)


def test_rectilinear_geometry_and_explicit_variant_remain_in_specification():
    points = [(0, 0), (1, 1)]
    spec = normalize(None, variant="rectilinear", points=points)
    assert spec.graph is None
    assert spec.points is points
    assert spec.requested_variant == "rectilinear"
    assert build_problem(spec)[1] == "rectilinear"
