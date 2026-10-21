"""Enumerate minimal optima using original node/edge selections."""

import math
from time import perf_counter

from ._selection_model import NoIncumbent, SelectionModel, solve_selection
from .objects import OptimalSolutionPool


def enumerate_selections(strategy, limit, time_limit, logfile, solver, threads):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer.")
    if solver.lower() not in {"highs", "gurobi"}:
        raise ValueError("Unknown solver {!r}.".format(solver))
    solutions = []
    if limit == 0:
        return OptimalSolutionPool(solutions, exhausted=False)
    model = SelectionModel(strategy.spec, solver, time_limit, logfile, threads)
    try:
        coverage = strategy.configure(model)
        try:
            optimum, gap, _ = solve_selection(
                model,
                coverage if strategy.coverage_mode else [],
                strategy.decode,
                time_limit,
            )
        except NoIncumbent as error:
            return OptimalSolutionPool(
                solutions, exhausted=error.status == "infeasible"
            )
        if gap != 0:
            return OptimalSolutionPool(solutions, exhausted=False)
        model.add(model.cost == optimum[2])
        if strategy.coverage_mode:
            model.add(model.sum(coverage) == optimum[3])
        size = model.sum(model.nodes.values()) + model.sum(model.edges.values())
        while len(solutions) < limit:
            started = perf_counter()
            status, incumbent = model.solve(size, started + time_limit)
            if status == "infeasible":
                return OptimalSolutionPool(solutions, exhausted=True)
            if status != "optimal" or not incumbent:
                return OptimalSolutionPool(solutions, exhausted=False)
            result = strategy.decode(*model.selection())
            if not math.isclose(result[2], optimum[2], rel_tol=1e-8, abs_tol=1e-7) or (
                strategy.coverage_mode and result[3] != optimum[3]
            ):
                raise RuntimeError(
                    "Enumeration returned a selection outside the proven optimum."
                )
            solutions.append(strategy.solution(result, 0, perf_counter() - started))
            edges, nodes = result[:2]
            # Minimum cardinality proves minimality; exclude this selection and its supersets.
            model.add(
                model.sum(1 - model.edges[e] for e in edges)
                + model.sum(1 - model.nodes[n] for n in nodes)
                >= 1
            )
        return OptimalSolutionPool(solutions, exhausted=False)
    finally:
        model.close()
