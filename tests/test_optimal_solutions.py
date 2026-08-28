"""Tests for :meth:`BaseSteinerProblem.get_optimal_solutions`.

Covers the diamond-graph tie example from steinerpy#41, the `limit`/
`exhausted` contract, the scope boundary (classes whose `get_solution`
transforms the model in ways this method can't replicate), and an
independent brute-force oracle over small random graphs.
"""

import importlib
import itertools
import random

import networkx as nx
import pytest

from steinerpy import (
    SteinerProblem,
    OptimalSolutionPool,
    GroupSteinerProblem,
    PartialTerminalSteinerProblem,
    RectilinearSteinerProblem,
    PrizeCollectingProblem,
    NodeWeightedSteinerProblem,
)


def diamond_graph():
    """Two disjoint, equal-cost A->D paths: A-B-C-D and A-E-F-D, cost 3 each."""
    g = nx.Graph()
    for u, v in [("A", "B"), ("B", "C"), ("C", "D"), ("A", "E"), ("E", "F"), ("F", "D")]:
        g.add_edge(u, v, weight=1)
    return g


def edge_sets(pool):
    return {frozenset(frozenset(e) for e in sol.edges) for sol in pool}


def test_diamond_graph_finds_both_optimal_trees():
    g = diamond_graph()
    problem = SteinerProblem(g, [["A", "D"]], preprocess=False)
    pool = problem.get_optimal_solutions(limit=10)

    assert isinstance(pool, OptimalSolutionPool)
    assert pool.exhausted is True
    assert len(pool) == 2
    for sol in pool:
        assert sol.gap == 0.0
        assert sol.objective == 3.0

    assert edge_sets(pool) == {
        frozenset(frozenset(e) for e in [("A", "B"), ("B", "C"), ("C", "D")]),
        frozenset(frozenset(e) for e in [("A", "E"), ("E", "F"), ("F", "D")]),
    }


def test_preprocess_true_raises_value_error():
    g = diamond_graph()
    problem = SteinerProblem(g, [["A", "D"]])  # preprocess=True by default
    with pytest.raises(ValueError, match="preprocess"):
        problem.get_optimal_solutions()


def test_limit_truncates_and_reports_not_exhausted():
    g = nx.Graph()
    for m in ["M1", "M2", "M3", "M4"]:
        g.add_edge("A", m, weight=1)
        g.add_edge(m, "D", weight=1)

    problem = SteinerProblem(g, [["A", "D"]], preprocess=False)
    pool = problem.get_optimal_solutions(limit=2)

    assert len(pool) == 2
    assert pool.exhausted is False
    for sol in pool:
        assert sol.objective == 2.0


def brute_force_optimal_edge_sets(graph, terminals, weight="weight"):
    """Enumerate every edge subset of `graph` (2^|E|) whose induced subgraph
    connects all `terminals`, and return the minimum cost together with every
    edge set (as a frozenset of frozensets) achieving it."""
    edges = list(graph.edges())
    best_cost = None
    best_sets = []
    for r in range(len(edges) + 1):
        for combo in itertools.combinations(edges, r):
            sub = nx.Graph()
            sub.add_nodes_from(terminals)
            sub.add_edges_from(combo)
            if not all(nx.has_path(sub, terminals[0], t) for t in terminals[1:]):
                continue
            cost = sum(graph.edges[e][weight] for e in combo)
            if best_cost is None or cost < best_cost - 1e-9:
                best_cost = cost
                best_sets = [combo]
            elif abs(cost - best_cost) <= 1e-9:
                best_sets.append(combo)
    normalized = {frozenset(frozenset(e) for e in combo) for combo in best_sets}
    return best_cost, normalized


def random_graph(seed, n=7):
    rng = random.Random(seed)
    g = nx.Graph()
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n - 1):
        g.add_edge(perm[i], perm[i + 1], weight=rng.randint(1, 4))
    for _ in range(rng.randint(0, n)):
        u, v = rng.sample(range(n), 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v, weight=rng.randint(1, 4))
    return g


@pytest.mark.parametrize("seed", range(6))
def test_brute_force_oracle_small_random_graphs(seed):
    g = random_graph(seed)
    rng = random.Random(1000 + seed)
    terminals = rng.sample(list(g.nodes()), 3)

    oracle_cost, oracle_sets = brute_force_optimal_edge_sets(g, terminals)
    assert oracle_cost is not None, "random graph must connect the sampled terminals"

    problem = SteinerProblem(g, [terminals], preprocess=False)
    pool = problem.get_optimal_solutions(limit=len(oracle_sets) + 5)

    assert pool.exhausted is True
    assert all(sol.objective == oracle_cost for sol in pool)
    assert edge_sets(pool) == oracle_sets


def test_group_steiner_problem_not_implemented():
    g = nx.Graph()
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "C", weight=1)
    problem = GroupSteinerProblem(g, [["A"], ["C"]], preprocess=False)
    with pytest.raises(NotImplementedError):
        problem.get_optimal_solutions()


def test_partial_terminal_steiner_problem_not_implemented():
    g = nx.Graph()
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "C", weight=1)
    problem = PartialTerminalSteinerProblem(
        g, [["A", "B", "C"]], partial_terminals=["A"], preprocess=False
    )
    with pytest.raises(NotImplementedError):
        problem.get_optimal_solutions()


def test_rectilinear_steiner_problem_not_implemented():
    problem = RectilinearSteinerProblem([(0, 0), (1, 1), (0, 1)], preprocess=False)
    with pytest.raises(NotImplementedError):
        problem.get_optimal_solutions()


def test_prize_collecting_problem_not_implemented():
    g = nx.Graph()
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "C", weight=1)
    problem = PrizeCollectingProblem(g, [["A"]], node_prizes={"C": 5}, preprocess=False)
    with pytest.raises(NotImplementedError):
        problem.get_optimal_solutions()


def test_node_weighted_steiner_problem_not_implemented():
    g = nx.Graph()
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "C", weight=1)
    problem = NodeWeightedSteinerProblem(g, [["A", "C"]], node_weights={"B": 1})
    with pytest.raises(NotImplementedError):
        problem.get_optimal_solutions()


def test_budget_kwarg_raises_not_implemented():
    g = diamond_graph()
    problem = SteinerProblem(g, [["A", "D"]], preprocess=False, budget=10)
    with pytest.raises(NotImplementedError, match="budget"):
        problem.get_optimal_solutions()


def test_infeasible_instance_returns_empty_exhausted_pool():
    # Zero edges at all: run_model's fast infeasibility path (no ILP, no cut
    # generation), unlike a graph with disconnected *components* which the
    # lazy-cut loop can take a very long time to prove infeasible on.
    g = nx.Graph()
    g.add_node("A")
    g.add_node("B")

    problem = SteinerProblem(g, [["A", "B"]], preprocess=False)
    pool = problem.get_optimal_solutions()

    assert isinstance(pool, OptimalSolutionPool)
    assert len(pool) == 0
    assert pool.exhausted is True


def test_dual_ascent_flag_ignored():
    g = diamond_graph()
    problem = SteinerProblem(g, [["A", "D"]], preprocess=False, dual_ascent=True)
    pool = problem.get_optimal_solutions(limit=10)

    assert pool.exhausted is True
    assert len(pool) == 2
    assert edge_sets(pool) == {
        frozenset(frozenset(e) for e in [("A", "B"), ("B", "C"), ("C", "D")]),
        frozenset(frozenset(e) for e in [("A", "E"), ("E", "F"), ("F", "D")]),
    }


def test_gurobi_optimal_solutions_matches_highs():
    if importlib.util.find_spec("gurobipy") is None:
        pytest.skip("gurobipy is not installed.")
    try:
        import gurobipy as gp

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        gp.Model(env=env).dispose()
        env.dispose()
    except Exception:
        pytest.skip("Gurobi license not available.")

    g = diamond_graph()
    problem = SteinerProblem(g, [["A", "D"]], preprocess=False)

    pool_highs = problem.get_optimal_solutions(limit=10, solver="highs")
    pool_gurobi = problem.get_optimal_solutions(limit=10, solver="gurobi")

    assert pool_gurobi.exhausted == pool_highs.exhausted
    assert edge_sets(pool_gurobi) == edge_sets(pool_highs)
