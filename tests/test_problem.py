"""End-to-end compatibility checks for the unified problem interface."""

import itertools

import networkx as nx
import pytest

from steinerpy import Problem, SteinerProblem
from steinerpy import objects

CASES = [
    ("steiner", objects.SteinerProblem, {"terminal_groups": [[0, 2]]}),
    ("directed", objects.DirectedSteinerProblem, {"root": 0, "terminals": [2]}),
    ("group", objects.GroupSteinerProblem, {"groups": [[0], [2]]}),
    (
        "directed_group",
        objects.DirectedGroupSteinerProblem,
        {"groups": [[2]], "root": 0},
    ),
    (
        "partial_terminal",
        objects.PartialTerminalSteinerProblem,
        {"terminal_groups": [[0, 2]], "partial_terminals": [2]},
    ),
    (
        "full_terminal",
        objects.FullTerminalSteinerProblem,
        {"terminal_groups": [[0, 2]]},
    ),
    (
        "hop_constrained",
        objects.HopConstrainedSteinerProblem,
        {"root": 0, "terminals": [2], "hop_limit": 2},
    ),
    (
        "prize_collecting",
        objects.PrizeCollectingProblem,
        {"terminal_groups": [[0, 2]], "node_prizes": {0: 3, 2: 5}},
    ),
    (
        "directed_prize_collecting",
        objects.DirectedPrizeCollectingProblem,
        {"root": 0, "node_prizes": {0: 3, 2: 5}},
    ),
    (
        "node_weighted",
        objects.NodeWeightedSteinerProblem,
        {"terminal_groups": [[0, 2]], "node_weights": {0: 1, 1: 2, 2: 1}},
    ),
    (
        "max_weight_connected",
        objects.MaxWeightConnectedSubgraph,
        {"node_weights": {0: 3, 1: 1, 2: 5}, "root": 0},
    ),
    (
        "budgeted_max_weight_connected",
        objects.BudgetedMaxWeightConnectedSubgraph,
        {
            "node_weights": {0: 3, 1: 1, 2: 5},
            "root": 0,
            "node_costs": {0: 0, 1: 1, 2: 2},
            "node_budget": 3,
        },
    ),
    (
        "rectilinear",
        objects.RectilinearSteinerProblem,
        {"points": [(0, 0), (1, 1), (2, 0)]},
    ),
]


@pytest.mark.parametrize("variant,legacy,options", CASES)
def test_variants_preserve_solutions(variant, legacy, options):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")
    if variant.startswith("directed") or variant == "hop_constrained":
        graph = graph.to_directed()
    inputs = dict(options)
    if variant != "rectilinear":
        inputs["graph"] = graph
    problem = Problem(variant=variant, **inputs)
    expected = legacy(**inputs).get_solution(solver="highs")
    actual = problem.get_solution(solver="highs")
    assert type(actual) is type(expected)
    assert actual.objective == pytest.approx(expected.objective)
    assert actual.gap == expected.gap == 0
    assert problem.variant == variant
    assert type(problem.implementation) is legacy
    for attr in ("selected_nodes", "total_prize", "connected_terminals", "segments"):
        if hasattr(expected, attr):
            assert getattr(actual, attr) == getattr(expected, attr)


def test_registry_coverage():
    assert set(Problem.variants) == {case[0] for case in CASES}


@pytest.mark.parametrize("options", [{}, {"max_degree": 2}, {"budget": 2}])
def test_plain_modifiers(options):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")
    actual = Problem(graph, [[0, 2]], preprocess=False, **options).get_solution()
    expected = SteinerProblem(
        graph, [[0, 2]], preprocess=False, **options
    ).get_solution()
    assert type(actual) is type(expected)
    assert actual.objective == expected.objective
    assert actual.gap == 0


def test_enumeration_matches_brute_force():
    graph = nx.cycle_graph(4)
    nx.set_edge_attributes(graph, 1, "weight")
    feasible = []
    edges = list(graph.edges())
    for mask in itertools.product((False, True), repeat=len(edges)):
        selected = [edge for edge, keep in zip(edges, mask) if keep]
        candidate = nx.Graph()
        candidate.add_nodes_from(graph)
        candidate.add_edges_from(selected)
        if nx.has_path(candidate, 0, 2):
            feasible.append(selected)
    optimum = min(map(len, feasible))
    canonical = lambda edges: frozenset(frozenset(edge) for edge in edges)
    expected = {canonical(edges) for edges in feasible if len(edges) == optimum}
    pool = Problem(graph, [[0, 2]], preprocess=False).get_optimal_solutions()
    assert pool.exhausted
    assert {canonical(solution.edges) for solution in pool.solutions} == expected


@pytest.mark.parametrize(
    "options,match",
    [
        ({"variant": "typo"}, "Unknown variant"),
        ({"variant": []}, "Unknown variant"),
        ({"variant": "group", "terminal_groups": [[0, 2]]}, "terminal_groups"),
        ({"variant": "steiner", "root": 0}, "root"),
        ({"variant": "steiner", "node_prizes": {}}, "node_prizes"),
        ({"variant": "steiner", "hop_limit": 2}, "hop_limit"),
        ({"variant": "steiner", "time_limit": 2}, "time_limit"),
        ({"variant": "directed", "terminals": [2]}, "root"),
        ({"variant": "rectilinear", "points": [(0, 0)]}, "not graph"),
    ],
)
def test_invalid_configuration(options, match):
    with pytest.raises((TypeError, ValueError), match=match):
        Problem(nx.path_graph(3), **options)


def test_missing_graph():
    with pytest.raises(TypeError, match="requires graph"):
        Problem(terminal_groups=[[0, 2]])


def test_node_weighted_preprocessing():
    options = dict(variant="node_weighted", node_weights={0: 1, 1: 1, 2: 1})
    Problem(nx.path_graph(3), [[0, 2]], preprocess=False, **options)
    with pytest.raises(ValueError, match="preprocess=False"):
        Problem(nx.path_graph(3), [[0, 2]], preprocess=True, **options)


def test_unsupported_enumeration():
    problem = Problem(variant="rectilinear", points=[(0, 0), (1, 1)])
    with pytest.raises(NotImplementedError):
        problem.get_optimal_solutions()
