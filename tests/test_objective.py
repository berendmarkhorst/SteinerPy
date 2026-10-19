"""Check objective conventions against independent tiny original-graph oracles."""

import itertools
import random

import networkx as nx
import pytest

from steinerpy import Problem
from steinerpy import mathematical_model as mm
from steinerpy._objective import ObjectiveSpec
from steinerpy._problem_spec import normalize_problem
from steinerpy.pc_transform import transform_mwcsp_to_pcstp


def candidates(graph, include_empty=False):
    if include_empty:
        yield [], []
    for count in range(1, len(graph) + 1):
        for chosen in itertools.combinations(graph, count):
            subgraph = graph.subgraph(chosen)
            if nx.is_connected(subgraph):
                tree = nx.minimum_spanning_tree(subgraph, weight="cost")
                yield list(tree.edges), list(chosen)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "mode",
    [
        "edge",
        "node",
        "composition",
        "pc_native",
        "pc_classic",
        "mwcs_native",
        "mwcs_classic",
        "mwcs_budget",
    ],
)
def test_original_space_objectives_match_oracles_and_solver_reports(seed, mode):
    rng = random.Random(seed)
    graph = nx.cycle_graph(4)
    nx.set_edge_attributes(
        graph, {edge: rng.randint(1, 5) for edge in graph.edges}, "cost"
    )
    costs = {node: rng.randint(0, 4) for node in graph}
    prizes = {node: rng.randint(1, 5) for node in graph}
    weights = {node: rng.randint(-4, 6) for node in graph}
    weights[0] = 3
    options = {"weight": "cost"}
    if mode == "edge":
        options.update(terminals=[0, 3], preprocess=False)
    elif mode == "node":
        options.update(variant="node_weighted", terminals=[0, 3], node_weights=costs)
    elif mode == "composition":
        options.update(groups=[[1, 2], [3]], terminals=[0], node_costs=costs)
    elif mode.startswith("pc_"):
        classic = mode == "pc_classic"
        options.update(
            variant="prize_collecting",
            # Make every rewarded vertex a potential terminal for the legacy ILP;
            # its nonterminal-leaf restrictions differ from classic PCSTP.
            terminals=[0, 3] if classic else list(graph),
            node_prizes=prizes,
            penalty_cost=0 if classic else 2,
            pc_transform=classic,
        )
    else:
        options.update(
            variant="max_weight_connected",
            node_weights=weights,
            root=0,
            pc_transform=mode == "mwcs_classic",
        )
        if mode == "mwcs_budget":
            options.update(
                variant="budgeted_max_weight_connected",
                node_costs=dict.fromkeys(graph, 1),
                node_budget=3,
            )

    problem = Problem(graph, **options)
    objective = problem._spec.objective(problem.variant)
    evaluated = []
    for edges, nodes in candidates(graph, include_empty=mode == "pc_classic"):
        chosen = set(nodes)
        if mode in {"edge", "node"} and not {0, 3} <= chosen:
            continue
        if mode == "composition" and not (
            0 in chosen and 3 in chosen and chosen.intersection({1, 2})
        ):
            continue
        if mode in {"pc_native", "mwcs_native", "mwcs_budget"} and 0 not in chosen:
            continue
        if mode == "mwcs_budget" and len(chosen) > 3:
            continue
        edge_cost = sum(graph.edges[edge]["cost"] for edge in edges)
        penalty_count = 0
        if mode == "edge":
            expected = edge_cost
        elif mode in {"node", "composition"}:
            expected = edge_cost + sum(costs[node] for node in chosen)
        elif mode == "pc_classic":
            expected = edge_cost + sum(
                prizes[node] for node in graph if node not in chosen
            )
        elif mode == "pc_native":
            penalty_count = len(graph) - len(chosen)
            expected = (
                edge_cost - sum(prizes[node] for node in chosen) + 2 * penalty_count
            )
        elif mode == "mwcs_native":
            penalty_count = sum(
                weights[node] > 0 for node in graph if node not in chosen
            )
            expected = edge_cost - sum(max(weights[node], 0) for node in chosen)
            expected += (
                max(max(weight, 0) for weight in weights.values()) + 1
            ) * penalty_count
        else:
            expected = sum(weights[node] for node in chosen)
        assert objective.value(
            edges, nodes, penalty_count=penalty_count
        ) == pytest.approx(expected)
        evaluated.append(expected)

    expected = (
        max(evaluated) if mode in {"mwcs_classic", "mwcs_budget"} else min(evaluated)
    )
    solution = problem.get_solution()
    assert solution.objective == pytest.approx(expected)
    assert solution.gap == 0
    nodes = getattr(
        solution,
        "selected_nodes",
        {0, 3} | {v for edge in solution.edges for v in edge},
    )
    penalty_count = len(getattr(solution, "penalties", {}))
    assert objective.value(
        solution.edges, nodes, penalty_count=penalty_count
    ) == pytest.approx(solution.objective)


def test_binary_selections_count_original_undirected_edges_and_nodes_once():
    objective = ObjectiveSpec(
        {frozenset(("a", "b")): 3},
        {"a": 2, "b": -4, "isolated": 7},
        penalty_cost=5,
        report_sign=-1,
        report_offset=20,
    )
    edges = [("a", "b"), ("b", "a")]
    nodes = ["a", "b", "isolated", "isolated"]
    assert objective.canonical_value(edges, nodes, penalty_count=2) == 18
    assert objective.value(edges, nodes, penalty_count=2) == 2


def test_arc_orientation_and_isolated_nodes_are_preserved():
    objective = ObjectiveSpec(
        {("a", "b"): 2, ("b", "a"): 9}, {"alone": 5}, directed=True
    )
    assert objective.value([("a", "b")], ["alone"]) == 7
    assert objective.value([("b", "a")]) == 9


@pytest.mark.parametrize("directed", [False, True])
def test_graph_coefficients_preserve_cost_defaults_and_snapshot_inputs(directed):
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edge("a", (1, 2), cost=3)
    graph.add_edge((1, 2), "c")
    if directed:
        graph.add_edge((1, 2), "a", cost=9)
    objective = ObjectiveSpec.from_graph(graph, "cost")
    legacy_node = ObjectiveSpec.from_graph(graph, "cost", default_edge_cost=0)
    graph.clear()
    assert objective.edge_cost(("a", (1, 2))) == 3
    assert objective.edge_cost(((1, 2), "c")) == 1
    assert legacy_node.edge_cost(((1, 2), "c")) == 0
    assert objective.edge_cost(((1, 2), "a")) == (9 if directed else 3)


@pytest.mark.parametrize("solver", ["highs", "gurobi"])
@pytest.mark.parametrize("mode", ["forest", "directed", "reduced", "node_split"])
def test_standard_solver_objective_matches_working_graph_oracle(solver, mode):
    if solver == "gurobi":
        gp = pytest.importorskip("gurobipy")
        try:
            with gp.Env(params={"OutputFlag": 0}) as env:
                with gp.Model(env=env):
                    pass
        except gp.GurobiError:
            pytest.skip("Gurobi license is not available")

    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 2), (1, 2, 3), (1, 3, 1), (0, 3, 8), (2, 3, 7)], weight="cost"
    )
    options = {"weight": "cost", "preprocess": False, "terminals": [0, 2]}
    if mode == "forest":
        options["terminals"] = [[0, 2], [0, 3]]
    elif mode == "directed":
        graph = graph.to_directed()
        graph[1][0]["cost"] = 20
        options["root"] = 0
    elif mode == "reduced":
        graph = nx.path_graph(4)
        nx.set_edge_attributes(graph, 2, "cost")
        options.update(
            terminals=[0, 3], preprocess=True, heavy=False, contract_terminals=False
        )
    else:
        graph = nx.path_graph(3)
        nx.set_edge_attributes(graph, 2, "cost")
        options.pop("preprocess")
        options.update(variant="node_weighted", node_weights={0: 4, 1: 3, 2: 5})
    problem = Problem(graph, **options)._implementation
    if mode in {"reduced", "node_split"}:
        assert set(problem.graph.edges) != set(graph.edges)

    # Enumerate working-graph edges independently of the coefficient factory.
    working = problem.graph
    best = float("inf")
    for flags in itertools.product((False, True), repeat=len(problem.edges)):
        edges = [edge for edge, keep in zip(problem.edges, flags) if keep]
        selected = type(working)()
        selected.add_nodes_from(working)
        selected.add_edges_from(edges)
        if all(
            nx.has_path(selected, group[0], terminal)
            for group in problem.terminal_groups
            for terminal in group[1:]
        ):
            best = min(best, sum(working.edges[e]["cost"] for e in edges))

    builder = mm.build_model if solver == "highs" else mm.build_model_gurobi
    runner = mm.run_model if solver == "highs" else mm.run_model_gurobi
    model, x, _, y2, z = builder(problem)
    try:
        gap, _, value, edges, status = runner(
            model, problem, x, y2, z, return_status=True
        )
        assert status == "optimal"
        assert gap == 0
        assert value == best
        assert value == sum(working.edges[e]["cost"] for e in edges)
        for edge, variable in x.items():
            coefficient = (
                model.getLp().col_cost_[variable.index]
                if solver == "highs"
                else variable.Obj
            )
            assert coefficient == working.edges[edge]["cost"]
    finally:
        if solver == "gurobi":
            model.dispose()


def test_prize_offset_cancellation_preserves_small_original_edge_cost():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    spec = normalize_problem(
        graph,
        None,
        ["a", "b"],
        "prize_collecting",
        {
            "node_prizes": {"a": 1e20, "b": 1e20},
            "penalty_cost": 0,
            "pc_transform": True,
        },
        Problem.variants,
    )
    assert spec.objective("prize_collecting").value([("a", "b")], ["a", "b"]) == 1


def test_missing_edge_cost_conventions_remain_distinct():
    graph = nx.path_graph(3)
    ordinary = Problem(graph, terminals=[0, 2], preprocess=False)
    weighted = Problem(
        graph, variant="node_weighted", terminals=[0, 2], node_weights={1: 3}
    )
    assert (
        ordinary._spec.objective(ordinary.variant).value(graph.edges, graph.nodes) == 2
    )
    objective = weighted._spec.objective(weighted.variant)
    assert objective.value(graph.edges, graph.nodes) == 3
    assert weighted.get_solution().objective == 3


def test_objective_coefficients_snapshot_inputs_without_changing_them():
    graph = nx.Graph()
    graph.add_edge((1, 2), "b", cost=3)
    costs = {(1, 2): 4}
    spec = normalize_problem(
        graph,
        None,
        None,
        None,
        {"groups": [[(1, 2)], ["b"]], "node_costs": costs, "weight": "cost"},
        Problem.variants,
    )
    objective = spec.objective(None)
    graph.clear()
    costs.clear()
    assert objective.value([((1, 2), "b")], [(1, 2), "b"]) == 7


def test_solve_time_transform_override_keeps_both_legacy_conventions():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 2, "weight")
    problem = Problem(
        graph, variant="max_weight_connected", root=0, node_weights={0: 4, 1: -20, 2: 5}
    )
    native = problem.get_solution()
    classic = problem.get_solution(pc_transform=True)
    assert native.objective == -5
    assert classic.objective == 5
    assert (
        problem._spec.objective(problem.variant).value(
            native.edges, native.selected_nodes, penalty_count=len(native.penalties)
        )
        == native.objective
    )
    assert (
        problem._spec.objective(problem.variant, pc_transform=True).value(
            classic.edges, classic.selected_nodes
        )
        == classic.objective
    )


@pytest.mark.parametrize("directed", [False, True])
def test_edge_budget_reports_connected_requirements_instead_of_edge_cost(directed):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")
    if directed:
        graph = graph.to_directed()
        problem = Problem(graph, root=0, terminals=[1, 2], budget=1)
    else:
        problem = Problem(graph, terminals=[0, 1, 2], budget=1, preprocess=False)
    objective = problem._spec.objective(problem.variant)
    solution = problem.get_solution()
    expected_count = 2
    assert solution.objective == expected_count
    assert objective.value(solution.edges, penalty_count=1) == expected_count


def test_terminal_count_offsets_count_requirements_not_unique_vertices():
    problem = Problem(
        nx.path_graph(3), terminals=[[0, 1], [0, 2]], budget=1, preprocess=False
    )
    objective = problem._spec.objective(problem.variant)
    assert objective.report_offset == 4
    assert objective.value(penalty_count=1) == 3


def test_geometry_requires_graph_conversion_before_describing_coefficients():
    spec = normalize_problem(
        None, None, None, "rectilinear", {"points": [(0, 0)]}, Problem.variants
    )
    with pytest.raises(NotImplementedError, match="Hanan grid"):
        spec.objective("rectilinear")


def test_unknown_objective_family_is_explicit():
    graph = nx.path_graph(2)
    spec = normalize_problem(graph, None, [0, 1], None, {}, Problem.variants)
    with pytest.raises(NotImplementedError, match="No original-space objective"):
        spec.objective("unknown")


@pytest.mark.parametrize("root", [None, 0])
def test_directed_classic_prizes_match_exhaustive_arc_and_vertex_selections(root):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 2), (1, 2, 1), (0, 2, 7)])
    prizes = {0: 0, 1: 2, 2: 6}
    problem = Problem(
        graph, variant="directed_prize_collecting", root=root, node_prizes=prizes
    )
    objective = problem._spec.objective(problem.variant)
    best = sum(prizes.values()) if root is None else float("inf")
    for flags in itertools.product((False, True), repeat=len(graph)):
        nodes = [node for node, keep in zip(graph, flags) if keep]
        if not nodes or (root is not None and root not in nodes):
            continue
        for arc_flags in itertools.product(
            (False, True), repeat=graph.number_of_edges()
        ):
            edges = [arc for arc, keep in zip(graph.edges, arc_flags) if keep]
            if any(u not in nodes or v not in nodes for u, v in edges):
                continue
            selected = nx.DiGraph()
            selected.add_nodes_from(nodes)
            selected.add_edges_from(edges)
            anchors = nodes if root is None else [root]
            if not any(
                len(nx.descendants(selected, anchor)) == len(nodes) - 1
                for anchor in anchors
            ):
                continue
            expected = sum(graph.edges[edge]["weight"] for edge in edges)
            expected += sum(prizes[node] for node in graph if node not in nodes)
            assert objective.value(edges, nodes) == expected
            best = min(best, expected)
    solution = problem.get_solution()
    assert solution.objective == best
    assert solution.gap == 0
    assert objective.value(solution.edges, solution.selected_nodes) == best


@pytest.mark.parametrize(
    "weights",
    [
        {0: 4, 1: -3, 2: 5},
        {0: 1, 1: 2, 2: 3},
        {0: -4, 1: -2, 2: -1},
    ],
)
def test_mwcs_transformation_constant_recovers_signed_original_weight(weights):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 7, "cost")
    spec = normalize_problem(
        graph,
        None,
        None,
        "max_weight_connected",
        {"node_weights": weights, "weight": "cost", "pc_transform": True},
        Problem.variants,
    )
    objective = spec.objective("max_weight_connected")
    transformed, prizes, constant = transform_mwcsp_to_pcstp(graph, weights, "cost")
    for edges, nodes in candidates(graph):
        pc_value = sum(transformed.edges[edge]["cost"] for edge in edges)
        pc_value += sum(prizes[node] for node in graph if node not in nodes)
        original_weight = sum(weights[node] for node in nodes)
        assert pc_value - constant == objective.canonical_value(edges, nodes)
        assert objective.reported_value(pc_value - constant) == original_weight


@pytest.mark.parametrize("variant", ["partial_terminal", "full_terminal"])
def test_terminal_leaf_reporting_removes_artificial_big_m_costs(variant):
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3)])
    options = {"partial_terminals": [1]} if variant == "partial_terminal" else {}
    problem = Problem(graph, variant=variant, terminals=[1, 2, 3], **options)
    solution = problem.get_solution()
    assert solution.objective == 6
    assert problem._spec.objective(problem.variant).value(solution.edges) == 6
