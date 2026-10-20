"""Unsupported features must fail before constructors or transformations run."""

from functools import wraps
import math

import networkx as nx
import pytest

from steinerpy import Problem, objects
from steinerpy._composition import GroupComposition
from steinerpy._problem_dispatch import _IMPLEMENTATIONS
from steinerpy._problem_spec import normalize_problem
from steinerpy._problem_validation import validate_problem


@pytest.fixture
def forbid_construction(monkeypatch):
    def forbid(original):
        @wraps(original)
        def fail(*args, **kwargs):
            pytest.fail(
                "Validation must finish before any implementation is constructed"
            )

        return fail

    for cls in set(_IMPLEMENTATIONS.values()) | {GroupComposition}:
        monkeypatch.setattr(cls, "__init__", forbid(cls.__init__))


@pytest.mark.parametrize("name", ["hop_limit", "max_degree"])
@pytest.mark.parametrize("limit", [-1, 1.5, True, "2", math.inf, math.nan])
def test_invalid_directed_group_limits_fail_before_construction(
    name, limit, forbid_construction
):
    with pytest.raises(ValueError, match="non-negative integer"):
        Problem(
            nx.path_graph(3, create_using=nx.DiGraph),
            groups=[[2]],
            root=0,
            **{name: limit},
        )


@pytest.mark.parametrize("name", ["hop_limit", "max_degree"])
@pytest.mark.parametrize("limit", [-1, True, 1.5, "3", math.inf, math.nan])
def test_invalid_composed_hops_fail_before_construction(
    name, limit, forbid_construction
):
    with pytest.raises(ValueError, match="non-negative integer"):
        Problem(nx.path_graph(3), groups=[[0], [2]], node_costs={}, **{name: limit})


@pytest.mark.parametrize("budget", [-1, True, "3", math.inf, math.nan])
def test_invalid_directed_group_budget_fails_before_construction(
    budget, forbid_construction
):
    with pytest.raises(ValueError, match="budget.*finite and non-negative"):
        Problem(
            nx.path_graph(3, create_using=nx.DiGraph),
            groups=[[2]],
            root=0,
            budget=budget,
        )


@pytest.mark.parametrize(
    "directed, options, message",
    [
        (True, {"terminals": [2]}, "require root"),
        (True, {"terminals": [2], "root": 99}, "root.*not in the graph"),
        (False, {"variant": "directed", "terminals": [2], "root": 0}, "directed graph"),
        (True, {"variant": "group", "groups": [[2]]}, "undirected graph"),
        (
            False,
            {"groups": [[0], [2]], "node_costs": {}, "node_budget": 2},
            "composition",
        ),
        (
            False,
            {"groups": [[0], [2]], "node_costs": {}, "preprocess": True},
            "preprocess=False",
        ),
        (False, {"groups": [[0], [2]], "enumeration_safe": True}, "enumeration_safe"),
        (
            False,
            {
                "variant": "node_weighted",
                "terminals": [0, 2],
                "node_weights": {},
                "preprocess": True,
            },
            "preprocess=False",
        ),
        (
            False,
            {
                "variant": "node_weighted",
                "terminals": [0, 2],
                "node_weights": {},
                "max_degree": 2,
            },
            "cannot yet be combined",
        ),
        (
            False,
            {"variant": "node_weighted", "terminals": [0, 2], "node_weights": {1: -1}},
            "finite and non-negative",
        ),
        (
            False,
            {"variant": "node_weighted", "terminals": [0, 2], "node_weights": None},
            "must map",
        ),
        (
            False,
            {
                "variant": "prize_collecting",
                "terminals": [0, 2],
                "node_prizes": {2: math.nan},
            },
            "finite",
        ),
        (
            False,
            {
                "variant": "prize_collecting",
                "terminals": [0, 2],
                "node_prizes": {2: -1},
            },
            "non-negative",
        ),
        (
            False,
            {
                "variant": "prize_collecting",
                "terminals": [0, 2],
                "node_prizes": {99: 1},
            },
            "not in the graph",
        ),
        (
            False,
            {
                "variant": "prize_collecting",
                "terminals": [0, 2],
                "node_prizes": {},
                "pc_transform": True,
            },
            "classic prize-collecting",
        ),
        (
            False,
            {
                "variant": "prize_collecting",
                "terminals": [[0], [2]],
                "node_prizes": {},
                "penalty_cost": 0,
                "pc_transform": True,
            },
            "classic prize-collecting",
        ),
        (
            False,
            {
                "variant": "prize_collecting",
                "terminals": [0, 2],
                "node_prizes": {},
                "penalty_cost": 0,
                "pc_transform": True,
                "max_degree": 2,
            },
            "without degree limits or budgets",
        ),
        (
            True,
            {
                "variant": "directed_prize_collecting",
                "node_prizes": {},
                "pc_reduce": True,
            },
            "pc_reduce",
        ),
        (
            True,
            {
                "variant": "directed_prize_collecting",
                "node_prizes": {},
                "max_degree": 2,
            },
            "max_degree",
        ),
        (
            True,
            {
                "variant": "directed_prize_collecting",
                "node_prizes": {},
                "pc_transform": False,
            },
            "requires pc_transform=True",
        ),
        (
            False,
            {
                "variant": "partial_terminal",
                "terminals": [0, 2],
                "partial_terminals": [2],
                "budget": 4,
            },
            "original budget",
        ),
        (
            False,
            {"variant": "full_terminal", "terminals": [0, 2], "budget": 4},
            "original budget",
        ),
        (
            False,
            {
                "variant": "max_weight_connected",
                "node_weights": {0: 2},
                "pc_transform": True,
                "penalty_budget": 1,
            },
            "without degree limits or budgets",
        ),
        (
            False,
            {
                "variant": "budgeted_max_weight_connected",
                "node_weights": {},
                "node_costs": {},
                "node_budget": 2,
                "budget": 3,
            },
            "use node_budget",
        ),
        (
            False,
            {
                "variant": "budgeted_max_weight_connected",
                "node_weights": {},
                "node_costs": {},
                "node_budget": 2,
                "penalty_budget": 3,
            },
            "use node_budget",
        ),
        (
            False,
            {
                "variant": "budgeted_max_weight_connected",
                "node_weights": {},
                "node_costs": {},
                "node_budget": 2,
                "pc_transform": True,
            },
            "pc_transform",
        ),
        (
            False,
            {
                "variant": "budgeted_max_weight_connected",
                "node_weights": {},
                "node_costs": {},
                "node_budget": 2,
                "pc_reduce": True,
            },
            "pc_reduce",
        ),
        (
            False,
            {
                "variant": "budgeted_max_weight_connected",
                "node_weights": {},
                "node_costs": {1: -1},
                "node_budget": 2,
            },
            "non-negative",
        ),
        (False, {"terminals": [0, 2], "hop_limit": 2}, "Unsupported option.*hop_limit"),
        (
            False,
            {"terminals": [0, 2], "node_budget": 2},
            "Unsupported option.*node_budget",
        ),
    ],
)
def test_invalid_combinations_fail_before_construction(
    forbid_construction, directed, options, message
):
    graph = nx.path_graph(3)
    if directed:
        graph = graph.to_directed()
    before = graph.copy()
    with pytest.raises((TypeError, ValueError), match=message):
        Problem(graph, **options)
    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("cost", [-1, math.nan, math.inf, "invalid"])
def test_bad_original_edge_costs_fail_before_transformation(forbid_construction, cost):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, cost, "weight")
    with pytest.raises(ValueError, match="finite and non-negative"):
        Problem(graph, variant="node_weighted", terminals=[0, 2], node_weights={})


@pytest.mark.parametrize("cost", [0, -1])
def test_enumeration_safe_preprocessing_checks_original_costs(
    forbid_construction, cost
):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, cost, "weight")
    with pytest.raises(ValueError, match="strictly positive"):
        Problem(graph, terminals=[0, 2], enumeration_safe=True)


def test_explicit_steiner_directed_preprocessing_fails_early(forbid_construction):
    with pytest.raises(ValueError, match="preprocess=False"):
        Problem(nx.path_graph(3).to_directed(), terminals=[0, 2], variant="steiner")


def test_node_splitting_must_not_erase_original_arc_directions(forbid_construction):
    with pytest.raises(ValueError, match="undirected graph"):
        Problem(
            nx.path_graph(3).to_directed(),
            terminals=[0, 2],
            variant="node_weighted",
            node_weights={},
        )


def test_invalid_reduction_mode_fails_early(forbid_construction):
    with pytest.raises(ValueError, match="Unknown pc_reduce"):
        Problem(
            nx.path_graph(3),
            terminals=[0, 2],
            variant="prize_collecting",
            node_prizes={},
            pc_reduce="unknown",
        )


@pytest.mark.parametrize(
    "graph, options, message",
    [
        (
            nx.DiGraph(),
            {"variant": "directed_prize_collecting", "node_prizes": {}},
            "at least one node",
        ),
        (
            nx.path_graph(3).to_directed(),
            {
                "variant": "prize_collecting",
                "terminals": [0, 2],
                "node_prizes": {},
                "penalty_cost": 0,
                "pc_transform": True,
            },
            "requires an undirected graph",
        ),
    ],
)
def test_prize_graph_requirements_fail_early(
    forbid_construction, graph, options, message
):
    with pytest.raises(ValueError, match=message):
        Problem(graph, **options)


def test_validation_is_reusable_without_transforming_input():
    graph = nx.path_graph(3)
    options = {"budget": 3, "max_degree": 2, "preprocess": False}
    spec = normalize_problem(graph, None, [[0, 2]], None, options, Problem.variants)
    first = validate_problem(spec, "steiner", objects.SteinerProblem)
    second = validate_problem(spec, "steiner", objects.SteinerProblem)
    assert first == second
    first["terminal_groups"][0].clear()
    assert second["terminal_groups"] == [[0, 2]]
    assert spec.terminal_groups == ((0, 2),)
    assert options == {"budget": 3, "max_degree": 2, "preprocess": False}
    assert list(graph.edges) == [(0, 1), (1, 2)]


def test_signed_mwcs_weights_and_topology_only_edge_costs_remain_supported():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, -5, "weight")
    problem = Problem(
        graph,
        variant="budgeted_max_weight_connected",
        root=0,
        node_weights={0: 3, 1: -1, 2: 4},
        node_costs={0: 1, 1: 1, 2: 1},
        node_budget=3,
        max_degree=2,
    )
    solution = problem.get_solution()
    assert solution.objective == 6
    assert solution.gap == 0
    assert set(solution.selected_nodes) == {0, 1, 2}


def test_preprocessing_defaults_and_coercions_remain_supported():
    graph = nx.path_graph(3).to_directed()
    nx.set_edge_attributes(graph, 1, "weight")
    problem = Problem(graph, root=0, terminals=[2], hop_limit=2, preprocess=True)
    assert not problem.implementation.preprocess
    assert problem.get_solution().objective == 2

    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")
    with pytest.warns(UserWarning, match="Forcing preprocess=False"):
        problem = Problem(
            graph,
            variant="prize_collecting",
            terminals=[0, 2],
            node_prizes={0: 3, 2: 3},
            penalty_cost=0,
            pc_transform=True,
            preprocess=True,
        )
    assert not problem.implementation.preprocess
    assert problem.get_solution().objective == 2
