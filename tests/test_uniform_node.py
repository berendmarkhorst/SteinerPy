"""Uniform-cost node models retain the classical Steiner tree contract."""

import itertools
import math
import random

import networkx as nx
import pytest

from steinerpy import Problem, SteinerProblem
from steinerpy import _uniform_node as uniform
from steinerpy import mathematical_model as mm
from steinerpy import objects


@pytest.fixture(autouse=True)
def enable_node_model(monkeypatch):
    monkeypatch.setenv("STEINERPY_UNIFORM_NODE_MODEL", "1")
    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", "0")


@pytest.fixture(scope="module", params=["highs", "gurobi"])
def solver(request):
    if request.param == "gurobi":
        gp = pytest.importorskip("gurobipy")
        try:
            with gp.Env(params={"OutputFlag": 0}) as env:
                with gp.Model(env=env):
                    pass
        except gp.GurobiError:
            pytest.skip("Gurobi license is not available")
    return request.param


def exhaustive_cost(graph, terminals, weight="weight"):
    """Enumerate original edges independently of the node formulation."""
    edges = list(graph.edges)
    best = math.inf
    for flags in itertools.product((False, True), repeat=len(edges)):
        chosen = [edge for edge, keep in zip(edges, flags) if keep]
        cost = sum(graph.edges[edge].get(weight, 1) for edge in chosen)
        if cost >= best:
            continue
        selected = nx.Graph()
        selected.add_nodes_from(terminals)
        selected.add_edges_from(chosen)
        if nx.is_connected(selected):
            best = cost
    return best


def assert_tree(graph, terminals, edges, objective, weight="weight"):
    selected = nx.Graph()
    selected.add_nodes_from(terminals)
    selected.add_edges_from(edges)
    assert nx.is_tree(selected)
    assert all(graph.has_edge(*edge) for edge in edges)
    assert all(node in terminals or degree >= 2 for node, degree in selected.degree)
    assert objective == pytest.approx(
        sum(graph.edges[edge].get(weight, 1) for edge in edges), abs=0
    )


def disconnected_degree_feasible_graph():
    # The two triangles satisfy every local degree row, but connecting them
    # requires more nodes than selecting both triangles separately.
    graph = nx.Graph([(0, 1), (1, 2), (2, 0), (10, 11), (11, 12), (12, 10)])
    nx.add_path(graph, range(2, 11))
    return graph, [0, 10]


@pytest.mark.parametrize("seed", range(5))
def test_node_model_matches_independent_edge_oracle(solver, seed):
    graph = nx.cycle_graph(7)
    chords = list(nx.non_edges(graph))
    graph.add_edges_from(random.Random(seed).sample(chords, 2))
    terminals = [0, 2, 4]
    problem = SteinerProblem(graph, [terminals], preprocess=False)

    solution = problem.get_solution(solver=solver, time_limit=20)

    assert solution.gap == 0
    assert solution.objective == exhaustive_cost(graph, terminals)
    assert_tree(graph, terminals, solution.edges, solution.objective)
    assert problem.cut_stats["formulation"] == "uniform_node"
    assert problem.cut_stats["model_columns"] == graph.number_of_nodes()


@pytest.mark.parametrize("cost", [1e-12, 0.125, 1.0, 1e12])
def test_custom_weight_scales_and_mixed_node_labels(solver, cost):
    graph = nx.relabel_nodes(nx.cycle_graph(6), {0: "root", 2: ("t", 2), 4: "last"})
    nx.set_edge_attributes(graph, cost, "cost")
    terminals = ["root", ("t", 2), "last"]
    problem = SteinerProblem(graph, [terminals], weight="cost", preprocess=False)

    solution = problem.get_solution(solver=solver, time_limit=20)

    assert solution.gap == 0
    assert solution.objective == pytest.approx(4 * cost, abs=0)
    assert_tree(graph, terminals, solution.edges, solution.objective, "cost")


def test_overflowing_working_tree_cost_is_not_certified(solver):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1e308, "weight")
    problem = SteinerProblem(graph, [[0, 2]], preprocess=False)
    assert uniform.uniform_cost(problem) == 1e308

    with pytest.raises(RuntimeError, match="not finite"):
        problem.get_solution(solver=solver, time_limit=20)


def test_overflowing_original_tree_cost_is_not_certified(solver):
    graph = nx.complete_graph(4)
    graph.add_edge(4, 0)
    nx.set_edge_attributes(graph, 1e308, "weight")
    problem = SteinerProblem(graph, [[4, 1]], heavy=False, enumeration_safe=True)
    assert problem.reduction_tracker.fixed_cost == 1e308
    assert uniform.uniform_cost(problem) == 1e308

    # The reduced optimum has one finite-cost edge; restoring the fixed edge
    # overflows, so its otherwise valid node-count proof cannot be reported.
    with pytest.raises(RuntimeError, match="not finite after backmapping"):
        problem.get_solution(solver=solver, time_limit=20)


def test_integer_separation_connects_degree_feasible_components(solver):
    graph, terminals = disconnected_degree_feasible_graph()
    problem = SteinerProblem(graph, [terminals], preprocess=False)

    gap, _, objective, edges, status = uniform.solve_uniform_node(
        problem, 1, solver=solver, time_limit=20
    )

    assert status == "optimal" and gap == 0
    assert objective == nx.shortest_path_length(graph, *terminals)
    assert_tree(graph, terminals, edges, objective)
    assert problem.cut_stats["cuts_added"] > 0


def test_separators_are_violated_and_minimal():
    graph, terminals = disconnected_degree_feasible_graph()
    selected = {0, 1, 2, 10, 11, 12}

    separators = uniform.find_node_separators(graph, terminals, selected)

    assert separators
    for separator in separators:
        assert separator and not separator.intersection(selected)
        assert not nx.has_path(graph.subgraph(set(graph) - separator), *terminals)
        for node in separator:
            assert nx.has_path(
                graph.subgraph(set(graph) - (separator - {node})), *terminals
            )
    assert uniform.find_node_separators(graph, terminals, set(graph)) == []


def test_decode_removes_cycles_leaves_and_terminal_free_components():
    graph = nx.Graph(
        [("r", 1), (1, "t"), ("r", "t"), (1, "leaf"), (7, 8), (8, 9), (9, 7)]
    )
    edges = uniform.decode_tree(graph, ["r", "t"], set(graph))
    assert edges is not None
    assert_tree(graph, ["r", "t"], edges, len(edges))
    assert not set(itertools.chain.from_iterable(edges)).intersection({"leaf", 7, 8, 9})
    assert uniform.decode_tree(graph, ["r", "t"], {"r", 1}) is None
    assert uniform.decode_tree(graph, ["r", 7], set(graph)) is None


@pytest.mark.parametrize("cost", [0, math.inf, math.nan, 1 + 1e-10])
def test_nonpositive_nonfinite_and_near_uniform_costs_are_ineligible(cost):
    graph = nx.cycle_graph(5)
    graph[0][1]["weight"] = cost
    problem = SteinerProblem(graph, [[0, 2]], preprocess=False)
    assert uniform.uniform_cost(problem) is None


def test_uniform_zero_cost_graph_is_ineligible():
    graph = nx.cycle_graph(5)
    nx.set_edge_attributes(graph, 0, "weight")
    problem = SteinerProblem(graph, [[0, 2]], preprocess=False)
    assert uniform.uniform_cost(problem) is None


@pytest.mark.parametrize("modifier", ["budget", "max_degree", "hop_limit"])
def test_modifiers_are_ineligible(modifier):
    problem = SteinerProblem(
        nx.cycle_graph(5), [[0, 2]], preprocess=False, **{modifier: 3}
    )
    assert uniform.uniform_cost(problem) is None


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph])
def test_directed_and_parallel_edge_graphs_are_ineligible(graph_type):
    graph = graph_type(nx.cycle_graph(5))
    problem = SteinerProblem(graph, [[0, 2]], preprocess=False)
    assert uniform.uniform_cost(problem) is None


def test_self_loops_are_ineligible():
    graph = nx.cycle_graph(5)
    graph.add_edge(0, 0)
    problem = SteinerProblem(graph, [[0, 2]], preprocess=False)
    assert uniform.uniform_cost(problem) is None


def test_forest_subclass_and_trivial_instances_are_ineligible():
    class SpecializedSteinerProblem(SteinerProblem):
        pass

    graph = nx.cycle_graph(6)
    for problem in (
        SteinerProblem(graph, [[0, 2], [3, 5]], preprocess=False),
        SpecializedSteinerProblem(graph, [[0, 2]], preprocess=False),
        SteinerProblem(graph, [[0, 0]], preprocess=False),
    ):
        assert uniform.uniform_cost(problem) is None
    assert uniform.uniform_cost(SteinerProblem(graph, [[0, 2]], preprocess=False)) == 1


@pytest.mark.parametrize(
    "terminals, preprocess, dw_limit, expected",
    [([0, 2], False, 10, 2), ([0, 0], False, 0, 0), ([0, 2], True, 0, 2)],
    ids=["dynamic_program", "trivial", "preprocessing"],
)
def test_existing_exact_shortcuts_take_priority(
    monkeypatch, terminals, preprocess, dw_limit, expected
):
    def unexpected_model(*args, **kwargs):
        raise AssertionError("An existing exact shortcut should solve this instance")

    monkeypatch.setenv("STEINERPY_DW_MAX_TERMINALS", str(dw_limit))
    monkeypatch.setattr(uniform, "solve_uniform_node", unexpected_model)
    monkeypatch.setattr(objects, "build_model", unexpected_model)
    problem = SteinerProblem(nx.path_graph(3), [terminals], preprocess=preprocess)

    solution = problem.get_solution(time_limit=20)

    assert solution.gap == 0 and solution.objective == expected
    assert problem.cut_stats == {}


def test_dispatch_override_keeps_original_edge_backend(monkeypatch):
    calls = []
    real_build = objects.build_model

    def recording_build(*args, **kwargs):
        calls.append(True)
        return real_build(*args, **kwargs)

    monkeypatch.setenv("STEINERPY_UNIFORM_NODE_MODEL", "0")
    monkeypatch.setattr(objects, "build_model", recording_build)
    problem = SteinerProblem(nx.cycle_graph(6), [[0, 2, 4]], preprocess=False)
    assert uniform.uniform_cost(problem) is None

    solution = problem.get_solution(time_limit=20)

    assert calls and solution.objective == 4 and solution.gap == 0
    assert problem.cut_stats.get("formulation") != "uniform_node"


def test_repeated_solve_does_not_keep_node_model_diagnostics(monkeypatch, solver):
    problem = SteinerProblem(nx.cycle_graph(6), [[0, 2, 4]], preprocess=False)
    first = problem.get_solution(solver=solver, time_limit=20)
    assert problem.cut_stats["formulation"] == "uniform_node"
    monkeypatch.setenv("STEINERPY_UNIFORM_NODE_MODEL", "0")

    second = problem.get_solution(solver=solver, time_limit=20)

    assert first.gap == second.gap == 0
    assert first.objective == second.objective == 4
    assert problem.cut_stats.get("formulation") != "uniform_node"


@pytest.mark.parametrize(
    "flag", ["dual_ascent", "primal_local_search", "implied_profit"]
)
@pytest.mark.parametrize("at_construction", [False, True])
def test_explicit_accelerators_keep_existing_paths(monkeypatch, flag, at_construction):
    def unexpected_node_model(*args, **kwargs):
        raise AssertionError("Explicit edge-based accelerator reached node model")

    monkeypatch.setattr(uniform, "solve_uniform_node", unexpected_node_model)
    options = {flag: True}
    problem = SteinerProblem(
        nx.cycle_graph(6),
        [[0, 2, 4]],
        preprocess=False,
        **(options if at_construction else {}),
    )
    solution = problem.get_solution(
        time_limit=20, **({} if at_construction else options)
    )
    assert solution.gap == 0 and solution.objective == 4


def test_unified_problem_dispatch_uses_node_model():
    problem = Problem(nx.cycle_graph(6), terminals=[0, 2, 4], preprocess=False)
    solution = problem.get_solution(time_limit=20)
    assert solution.gap == 0 and solution.objective == 4
    assert problem.implementation.cut_stats["formulation"] == "uniform_node"


def test_preprocessing_fixed_edges_are_restored(solver):
    graph = nx.complete_bipartite_graph(3, 3)
    graph.add_edge(0, 6)
    terminals = [6, 1, 2]
    problem = SteinerProblem(graph, [terminals], heavy=False, enumeration_safe=True)
    assert problem.reduction_tracker.fixed_cost == 1
    assert uniform.uniform_cost(problem) == 1

    solution = problem.get_solution(solver=solver, time_limit=20)

    assert problem.cut_stats["formulation"] == "uniform_node"
    assert solution.gap == 0 and solution.objective == exhaustive_cost(graph, terminals)
    assert_tree(graph, terminals, solution.edges, solution.objective)
    assert frozenset((0, 6)) in {frozenset(edge) for edge in solution.edges}


def test_degree_two_contraction_can_disable_node_model():
    graph = nx.cycle_graph(7)
    problem = SteinerProblem(graph, [[0, 2, 4]], heavy=False, contract_terminals=False)
    assert {data.get("weight", 1) for _, _, data in problem.graph.edges(data=True)} == {
        2,
        3,
    }
    assert uniform.uniform_cost(problem) is None

    solution = problem.get_solution(time_limit=20)

    assert solution.gap == 0 and solution.objective == 4
    assert_tree(graph, [0, 2, 4], solution.edges, solution.objective)
    assert problem.cut_stats.get("formulation") != "uniform_node"


def test_infeasible_terminals_are_not_reported_as_an_optimum(solver):
    graph = nx.Graph([(0, 1), (2, 3)])
    problem = SteinerProblem(graph, [[0, 3]], preprocess=False)
    gap, _, objective, edges, status = uniform.solve_uniform_node(
        problem, 1, solver=solver, time_limit=20
    )
    assert status == "infeasible" and math.isinf(gap) and math.isinf(objective)
    assert edges == []


@pytest.mark.parametrize("limit", [0, 1e-12])
def test_tiny_timeout_has_no_false_proof(solver, limit):
    graph, terminals = disconnected_degree_feasible_graph()
    problem = SteinerProblem(graph, [terminals], preprocess=False)
    gap, _, objective, edges, status = uniform.solve_uniform_node(
        problem, 1, solver=solver, time_limit=limit
    )
    assert status == "incomplete" and math.isinf(gap)
    # Gurobi may find a valid heuristic incumbent even under a tiny limit.
    if edges:
        assert_tree(graph, terminals, edges, objective)
    else:
        assert math.isinf(objective)


@pytest.mark.parametrize("has_incumbent", [False, True])
def test_nonoptimal_status_never_certifies_zero_gap(monkeypatch, solver, has_incumbent):
    outcome = "_highs_outcome" if solver == "highs" else "_gurobi_outcome"
    monkeypatch.setattr(mm, outcome, lambda *args: ("incomplete", has_incumbent))
    graph = nx.complete_graph(3)
    problem = SteinerProblem(graph, [[0, 1, 2]], preprocess=False)

    gap, _, objective, edges, status = uniform.solve_uniform_node(
        problem, 1, solver=solver, time_limit=20
    )

    assert status == "incomplete" and math.isinf(gap)
    if has_incumbent:
        assert objective == 2
        assert_tree(graph, [0, 1, 2], edges, objective)
    else:
        assert math.isinf(objective) and edges == []


def test_final_decode_rejects_unseparated_disconnected_incumbent(monkeypatch, solver):
    monkeypatch.setattr(uniform, "find_node_separators", lambda *args: [])
    graph, terminals = disconnected_degree_feasible_graph()
    problem = SteinerProblem(graph, [terminals], preprocess=False)

    gap, _, objective, edges, status = uniform.solve_uniform_node(
        problem, 1, solver=solver, time_limit=20
    )

    assert status == "incomplete" and math.isinf(gap) and math.isinf(objective)
    assert edges == []


def test_deadline_after_cut_rejects_previous_disconnected_selection(monkeypatch):
    clock = [0.0]
    real_separate = uniform.find_node_separators

    def advance_after_separation(*args, **kwargs):
        cuts = real_separate(*args, **kwargs)
        assert cuts
        clock[0] = 2.0
        return cuts

    monkeypatch.setattr(uniform.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(uniform, "find_node_separators", advance_after_separation)
    graph, terminals = disconnected_degree_feasible_graph()
    problem = SteinerProblem(graph, [terminals], preprocess=False)

    gap, _, objective, edges, status = uniform.solve_uniform_node(
        problem, 1, time_limit=1
    )

    assert status == "incomplete" and math.isinf(gap) and math.isinf(objective)
    assert edges == []


def test_enumeration_retains_distinct_trees_on_the_same_nodes(monkeypatch):
    def unexpected_node_model(*args, **kwargs):
        raise AssertionError("Enumeration must retain its edge variables")

    monkeypatch.setattr(uniform, "solve_uniform_node", unexpected_node_model)
    graph = nx.complete_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")
    problem = SteinerProblem(graph, [[0, 1, 2]], preprocess=False)
    pool = problem.get_optimal_solutions(limit=10, time_limit=20)
    assert pool.exhausted
    assert len(pool.solutions) == 3
    assert all(
        solution.gap == 0 and solution.objective == 2 for solution in pool.solutions
    )
