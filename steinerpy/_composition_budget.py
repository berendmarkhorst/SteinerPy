"""Optional group coverage under original-edge budgets after node splitting."""

import math
from dataclasses import replace
from time import perf_counter

import networkx as nx

from . import mathematical_model as mm
from ._composition import compile_groups, make_view, decode_groups
from .objects import BudgetSolution


def decode_budget(spec, compiled, arcs, claimed):
    selected = nx.DiGraph()
    selected.add_node(compiled.root)
    selected.add_edges_from(arcs)
    reachable = {compiled.root} | nx.descendants(selected, compiled.root)
    used = {a for a in selected.edges if a[0] in reachable}
    if sum(a in used for a in compiled.source_arcs) != 1:
        raise RuntimeError(
            "Budget composition does not select exactly one original root."
        )
    paid = {n for a, n in compiled.node_arcs.items() if a in used}
    # Optional flows may leave surplus arcs ending at an unpaid input port.
    used = {
        a
        for a in used
        if a not in compiled.edge_arcs or set(compiled.edge_arcs[a]) <= paid
    }
    if paid:
        edges, nodes, cost = decode_groups(
            spec, compiled, used, None, require_all_groups=False
        )
    else:
        if spec.terminal_groups:
            raise RuntimeError("Budget composition omits mandatory terminals.")
        edges, nodes, cost = [], [], 0
    if sum(spec.graph.edges[e].get(spec.weight, 1) for e in edges) > spec.budget + 1e-7:
        raise RuntimeError("Budget composition exceeds the original-edge budget.")
    missed = [i for i, g in enumerate(spec.groups) if set(nodes).isdisjoint(g)]
    covered = len(spec.groups) - len(missed)
    if covered < claimed:
        raise RuntimeError("Budget composition overstates reached groups.")
    return edges, nodes, cost, missed, covered


def solve_budget(strategy, time_limit, logfile, solver, threads):
    started = perf_counter()
    spec = strategy.spec
    if not spec.groups:
        # Match plain edge-budget semantics: first terminal required, others optional.
        terminals = spec.terminal_groups[0]
        spec = replace(
            spec,
            groups=tuple((t,) for t in terminals),
            terminal_groups=((terminals[0],),),
        )
    compiled = compile_groups(spec)
    view = make_view(spec, compiled)
    view.budget = spec.budget
    view._budget_edge_costs = {
        a: spec.graph.edges[e].get(spec.weight, 1)
        for a, e in compiled.edge_arcs.items()
    }
    model, x, y1, _, _, _, penalties = mm.build_budget_model(
        view, time_limit=time_limit, logfile=logfile, threads=threads, solver=solver
    )
    try:
        model.addConstr(sum(x[a] for a in compiled.source_arcs) == 1)
        # Mandatory sinks are appended after the optional original groups.
        for terminal in compiled.terminals[len(spec.groups) :]:
            model.addConstr(penalties[0, terminal] == 0)
        gap, _, count, arcs, _, status = mm.run_budget_model(
            model, view, x, penalties, return_status=True, solver=solver
        )
        if status == "infeasible" or count == 0:
            raise RuntimeError(
                "Budget composition stopped without a feasible incumbent or is infeasible."
            )
        mandatory_count = len(compiled.terminals) - len(spec.groups)
        claimed = count - 1 - mandatory_count
        edges, nodes, cost, missed, covered = decode_budget(
            spec, compiled, arcs, claimed
        )
        gap = 0 if status == "optimal" and covered == claimed else math.inf
        remaining = max(0, time_limit - (perf_counter() - started))
        # First prove coverage, then minimize original edge plus node costs.
        if status == "optimal" and remaining > 0:
            model.addConstr(
                sum(penalties.values()) == len(compiled.terminals) - (count - 1)
            )
            cost_expr = sum(
                x[e] * compiled.graph.edges[e].get(spec.weight, 1) for e in view.edges
            )
            if solver == "highs":
                model.setOptionValue("time_limit", remaining)
                model.minimize(cost_expr)
                outcome, incumbent = mm._highs_outcome(model)
                value = model.variableValue
            else:
                from gurobipy import GRB

                model.Params.TimeLimit = remaining
                model.setObjective(cost_expr, GRB.MINIMIZE)
                model.optimize()
                outcome, incumbent = mm._gurobi_outcome(model, GRB)
                value = lambda v: v.X
            gap = math.inf
            if incumbent:
                candidate = [e for e in view.edges if value(x[e]) > 0.5]
                decoded = decode_budget(spec, compiled, candidate, claimed)
                if decoded[2] <= cost:
                    edges, nodes, cost, missed, covered = decoded
                working_cost = sum(
                    compiled.graph.edges[e].get(spec.weight, 1) for e in candidate
                )
                if (
                    outcome == "optimal"
                    and covered == claimed
                    and math.isclose(cost, working_cost, rel_tol=1e-7, abs_tol=1e-7)
                ):
                    gap = 0
        else:
            gap = math.inf
        result = BudgetSolution(
            gap=gap,
            runtime=perf_counter() - started,
            objective=covered,
            selected_edges=edges,
            original_selected_edges=edges,
            connected_terminals=covered,
            total_terminals=len(spec.groups),
            penalties={
                (
                    f"group_{i}"
                    if strategy.spec.groups
                    else f"group_0_{spec.groups[i][0]}"
                ): 1
                for i in missed
            },
        )
        result.selected_nodes = nodes
        result.total_cost = cost
        return result
    finally:
        if solver == "gurobi":
            model.dispose()
