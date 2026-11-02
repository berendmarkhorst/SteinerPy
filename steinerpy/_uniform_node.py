"""Uniform-cost Steiner trees using node variables and integer separator cuts.

Fischetti et al. (2017), *Thinning out Steiner trees: a node-based model for
uniform edge costs*, Mathematical Programming Computation 9, 203--229,
doi:10.1007/s12532-016-0111-0, Sections 2 and 2.1.
"""

import math
import os
import time
from collections import deque

import highspy as hp
import networkx as nx

from . import mathematical_model as mm


def uniform_cost(problem):
    """Return the common working cost when the plain node model is eligible."""
    from .objects import SteinerProblem

    graph = problem.graph
    if (
        os.environ.get("STEINERPY_UNIFORM_NODE_MODEL") == "0"
        or type(problem) is not SteinerProblem
        or graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
        or len(problem.terminal_groups) != 1
        or len(set(problem.terminal_groups[0])) < 2
        or any(
            getattr(problem, name, None) is not None
            for name in ("budget", "max_degree", "hop_limit")
        )
    ):
        return None
    cost = None
    for _, _, data in graph.edges(data=True):
        value = data.get(problem.weight, 1)
        if not math.isfinite(value) or value <= 0:
            return None
        if cost is None:
            cost = value
        elif value != cost:
            return None
    return cost


def prefer_node_model(problem):
    """Choose the node formulation within its mathematically eligible domain.

    Connected terminals attain the node-count lower bound without Steiner
    choices. Otherwise prefer terminal--Steiner incidence graphs with at least
    five edges per useful Steiner variable. This conservative empirical rule
    is evaluated in ``benchmarks/reports/uniform-selection`` on both backends.
    Nonterminal leaf pruning is only for selection; it does not alter the
    working graph or its reduction/backmapping state.
    """
    setting = os.environ.get("STEINERPY_UNIFORM_NODE_MODEL", "auto")
    if setting in {"0", "1"}:
        return setting == "1"
    graph = problem.graph
    terminals = set(problem.terminal_groups[0])
    if not terminals or not terminals <= set(graph):
        return False
    if nx.is_connected(graph.subgraph(terminals)):
        return True
    component = nx.node_connected_component(graph, problem.terminal_groups[0][0])
    if not terminals <= component:
        return False
    graph = graph.subgraph(component)
    if any(node not in terminals and degree == 1 for node, degree in graph.degree()):
        from .graph_reducer import degree_one_reduction

        graph = degree_one_reduction(graph, terminals)
    if any((left in terminals) == (right in terminals) for left, right in graph.edges):
        return False
    steiner_count = len(graph) - len(terminals)
    return steiner_count > 0 and graph.number_of_edges() >= 5 * steiner_count


def find_node_separators(graph, terminals, selected):
    """Find minimal root-to-terminal separators violated by an integer point.

    Starting from a selected component's boundary, keep only boundary vertices
    reachable from the root without crossing another boundary vertex (paper, §2.1).
    """
    terminals = list(dict.fromkeys(terminals))
    root = terminals[0]
    required = set(terminals)
    separators = []
    for component in nx.connected_components(graph.subgraph(selected)):
        if root in component or not required.intersection(component):
            continue
        boundary = {
            neighbor
            for node in component
            for neighbor in graph[node]
            if neighbor not in component
        }
        reached = {root}
        pending = [root]
        separator = set()
        while pending:
            for neighbor in graph[pending.pop()]:
                if neighbor in boundary:
                    separator.add(neighbor)
                elif neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        separators.append(frozenset(separator))
    return separators


def decode_tree(graph, terminals, selected):
    """Validate connectivity, discard irrelevant components and prune a tree."""
    terminals = list(dict.fromkeys(terminals))
    selected = set(selected)
    required = set(terminals)
    if not required <= selected or not selected <= set(graph):
        return None
    if not terminals:
        return []
    induced = graph.subgraph(selected)
    root = terminals[0]
    if not required <= nx.node_connected_component(induced, root):
        return None
    tree = nx.Graph()
    tree.add_node(root)
    tree.add_edges_from(nx.bfs_edges(induced, root))
    leaves = deque(
        node for node in tree if tree.degree(node) <= 1 and node not in required
    )
    while leaves:
        node = leaves.popleft()
        if node not in tree:
            continue
        neighbors = list(tree[node])
        tree.remove_node(node)
        for neighbor in neighbors:
            if tree.degree(neighbor) == 1 and neighbor not in required:
                leaves.append(neighbor)
    return [edge for edge in graph.edges if tree.has_edge(*edge)]


def _no_solution(start, status="incomplete"):
    return math.inf, time.perf_counter() - start, math.inf, [], status


def _checked_result(problem, selected, objective, outcome, start):
    edges = decode_tree(problem.graph, problem.terminal_groups[0], selected)
    if edges is None:
        return _no_solution(start)
    if outcome == "optimal" and (
        len(edges) != len(selected) - 1
        or not math.isclose(objective, len(edges), rel_tol=0.0, abs_tol=1e-6)
    ):
        raise RuntimeError("Uniform node objective does not match its decoded tree.")
    cost = sum(problem.graph.edges[edge].get(problem.weight, 1) for edge in edges)
    if not math.isfinite(cost):
        raise RuntimeError("Uniform node tree cost is not finite.")
    gap = 0.0 if outcome == "optimal" else math.inf
    return gap, time.perf_counter() - start, cost, edges, outcome


def _solve_highs(problem, cost, time_limit, logfile, threads):
    graph = problem.graph
    terminals = set(problem.terminal_groups[0])
    model = mm.make_model(time_limit, logfile, threads)
    model.setOptionValue("mip_rel_gap", 0.0)
    model.setOptionValue("mip_abs_gap", 0.0)
    y = {
        node: model.addVariable(
            lb=int(node in terminals), ub=1, type=hp.HighsVarType.kInteger
        )
        for node in graph
    }
    for node in graph:
        neighbors = sum((y[other] for other in graph[node]), 0 * y[node])
        model.addConstr(neighbors >= (1 if node in terminals else 2 * y[node]))
    # Unit edge count keeps arbitrarily small common costs out of MIP tolerances.
    objective = sum(y.values()) - 1
    pool = mm._HighsCutPool(model)
    start = time.perf_counter()
    try:
        while True:
            remaining = time_limit - (time.perf_counter() - start)
            if remaining <= 0:
                return _no_solution(start)
            model.setOptionValue("time_limit", remaining)
            solve_start = time.perf_counter()
            model.minimize(objective)
            pool.record_solve("mip", time.perf_counter() - solve_start)
            outcome, has_incumbent = mm._highs_outcome(model)
            if not has_incumbent:
                return _no_solution(start, outcome)
            selected = {node for node in graph if model.variableValue(y[node]) > 0.5}
            separators = find_node_separators(
                graph, problem.terminal_groups[0], selected
            )
            pool.record_separation()
            if not separators:
                return _checked_result(
                    problem, selected, model.getObjectiveValue(), outcome, start
                )
            if outcome != "optimal":
                return _no_solution(start)
            pool.age_and_purge()
            added = 0
            for separator in separators:
                if not separator:
                    return _no_solution(start, "infeasible")
                added += pool.add(
                    ("uniform_node", separator),
                    sum(y[node] for node in graph if node in separator),
                    1.0,
                )
            if not added:
                return _no_solution(start)
    finally:
        problem.cut_stats = dict(
            pool.stats, formulation="uniform_node", model_columns=model.getNumCol()
        )


def _solve_gurobi(problem, cost, time_limit, logfile, threads):
    import gurobipy as gp
    from gurobipy import GRB

    graph = problem.graph
    terminals = set(problem.terminal_groups[0])
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("TimeLimit", time_limit)
        count = mm._resolve_threads(threads)
        if count > 0:
            env.setParam("Threads", count)
        if logfile:
            env.setParam("LogFile", logfile)
        env.start()
        with gp.Model(env=env) as model:
            model.Params.MIPGap = 0.0
            model.Params.MIPGapAbs = 0.0
            model.Params.LazyConstraints = 1
            y = {
                node: model.addVar(lb=int(node in terminals), vtype=GRB.BINARY)
                for node in graph
            }
            for node in graph:
                neighbors = gp.quicksum(y[other] for other in graph[node])
                model.addConstr(neighbors >= (1 if node in terminals else 2 * y[node]))
            model.setObjective(gp.quicksum(y.values()) - 1, GRB.MINIMIZE)
            stats = dict(
                formulation="uniform_node",
                model_columns=len(y),
                cuts_added=0,
                separation_rounds=0,
                solve_rounds=1,
            )
            problem.cut_stats = stats
            callback_errors = []

            def separate(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    values = cb_model.cbGetSolution(list(y.values()))
                    selected = {node for node, value in zip(y, values) if value > 0.5}
                    separators = find_node_separators(
                        graph, problem.terminal_groups[0], selected
                    )
                    stats["separation_rounds"] += 1
                    # Lazy constraints may need to be re-emitted at later callbacks.
                    for separator in separators:
                        cb_model.cbLazy(
                            gp.quicksum(y[node] for node in graph if node in separator)
                            >= 1
                        )
                        stats["cuts_added"] += 1
                except Exception as error:
                    callback_errors.append(error)
                    cb_model.terminate()

            start = time.perf_counter()
            model.optimize(separate)
            if callback_errors:
                raise RuntimeError(
                    "Uniform node cut separation failed."
                ) from callback_errors[0]
            outcome, has_incumbent = mm._gurobi_outcome(model, GRB)
            if not has_incumbent:
                return _no_solution(start, outcome)
            selected = {node for node in graph if y[node].X > 0.5}
            return _checked_result(problem, selected, model.ObjVal, outcome, start)


def solve_uniform_node(
    problem, cost, time_limit=300, logfile="", solver="highs", threads=None
):
    """Solve an eligible node model, returning the existing five-item contract.

    Runtime excludes model construction. Incomplete solves carry an unknown gap;
    only independently checked, proven-optimal trees receive a zero gap.
    """
    start = time.perf_counter()
    problem.cut_stats = {"formulation": "uniform_node"}
    terminals = set(problem.terminal_groups[0])
    root = problem.terminal_groups[0][0]
    if root not in problem.graph or not terminals <= nx.node_connected_component(
        problem.graph, root
    ):
        return _no_solution(start, "infeasible")
    solve = _solve_gurobi if solver == "gurobi" else _solve_highs
    return solve(problem, cost, time_limit, logfile, threads)
