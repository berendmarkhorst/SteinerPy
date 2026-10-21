"""Shared original-space selections and flow constraints for feature combinations."""

import math
from time import perf_counter

import highspy as hp

from . import mathematical_model as mm
from ._feature_preprocessing import feasible_edges


class NoIncumbent(RuntimeError):
    def __init__(self, status):
        super().__init__(
            "Feature composition is infeasible or stopped without a feasible incumbent."
        )
        self.status = status


class SelectionModel:
    """Build the same linear constraints with either supported solver backend."""

    def __init__(self, spec, solver, time_limit, logfile, threads):
        self.spec = spec
        self.solver = solver.lower()
        self.env = None
        if self.solver == "highs":
            self.model = mm.make_model(time_limit, logfile, threads)
        elif self.solver == "gurobi":
            mm._check_gurobipy()
            import gurobipy as gp

            self.env = gp.Env(empty=True)
            self.env.setParam("OutputFlag", 0)
            self.env.start()
            self.model = gp.Model(env=self.env)
            self.model.Params.TimeLimit = time_limit
            self.model.Params.Threads = mm._resolve_threads(threads)
            if logfile:
                self.model.Params.LogFile = logfile
        else:
            raise ValueError(
                "Unknown solver {!r}; choose 'highs' or 'gurobi'.".format(solver)
            )
        self.zero = self.variable(upper=0)
        self.nodes = {n: self.variable(binary=True) for n in spec.graph}
        edges = (
            feasible_edges(spec)
            if spec.options.get("preprocess", False)
            else spec.graph.edges
        )
        self.edges = {e: self.variable(binary=True) for e in edges}
        self.arcs = dict(self.edges)
        if not spec.graph.is_directed():
            self.arcs.update({(v, u): x for (u, v), x in self.edges.items()})
        self.incoming, self.outgoing = mm._arc_adjacency(self.arcs)
        for (u, v), edge in self.edges.items():
            self.add(edge <= self.nodes[u])
            self.add(edge <= self.nodes[v])
        incidence = mm._incident_edges(self.edges)
        if spec.max_degree is not None:
            for n in self.nodes:
                self.add(
                    self.sum(self.edges[e] for e in incidence.get(n, ()))
                    <= spec.max_degree
                )
        for n in spec.partial_terminals or ():
            self.add(self.sum(self.edges[e] for e in incidence.get(n, ())) <= 1)
        if spec.hop_limit is not None:
            self.add(self.sum(self.edges.values()) <= spec.hop_limit)
        self.edge_cost = self.sum(
            x * spec.graph.edges[e].get(spec.weight, 1) for e, x in self.edges.items()
        )
        node_cost = self.sum(
            x * (spec.node_costs or {}).get(n, 0) for n, x in self.nodes.items()
        )
        self.cost = self.edge_cost + node_cost
        if spec.node_budget is not None:
            self.add(node_cost <= spec.node_budget)
        if spec.budget is not None:
            self.add(self.edge_cost <= spec.budget)

    def variable(self, upper=1, binary=False):
        if self.solver == "highs":
            kind = hp.HighsVarType.kInteger if binary else hp.HighsVarType.kContinuous
            return self.model.addVariable(0, upper, type=kind)
        return self.model.addVar(lb=0, ub=upper, vtype="B" if binary else "C")

    def sum(self, values):
        # An anchored zero also handles edgeless graphs and constant objectives.
        return sum(values, 0 * self.zero)

    def add(self, constraint):
        self.model.addConstr(constraint)

    def flow(self, supplies, demands, capacity=1):
        """Require directed flow; undirected edge capacity is shared by its uses."""
        flows = {a: self.variable(upper=capacity) for a in self.arcs}
        for a, value in flows.items():
            self.add(value <= capacity * self.arcs[a])
        for n in self.nodes:
            balance = self.sum(flows[a] for a in self.outgoing.get(n, ()))
            balance -= self.sum(flows[a] for a in self.incoming.get(n, ()))
            self.add(balance == supplies.get(n, 0) - demands.get(n, 0))

    def attach_selected_nodes(self, roots):
        """Every selected node must be reachable from a required component root."""
        size = len(self.nodes)
        supplies = {r: self.variable(upper=size) for r in roots}
        for r, active in roots.items():
            self.add(supplies[r] <= size * active)
        self.flow(supplies, self.nodes, capacity=size)

    def choices(self, groups, roots, optional):
        coverage = []
        for group in groups:
            connected = self.variable(binary=True)
            coverage.append(connected)
            if not optional:
                self.add(connected == 1)
            supplies = {r: self.variable() for r in roots}
            for r, active in roots.items():
                self.add(supplies[r] <= active)
            demands = {n: self.variable() for n in group}
            for n, demand in demands.items():
                self.add(demand <= self.nodes[n])
            self.add(self.sum(demands.values()) == connected)
            self.flow(supplies, demands)
        return coverage

    def solve(self, objective, deadline):
        remaining = max(0, deadline - perf_counter())
        if self.solver == "highs":
            self.model.setOptionValue("time_limit", remaining)
            self.model.minimize(objective)
            return mm._highs_outcome(self.model)
        from gurobipy import GRB

        self.model.Params.TimeLimit = remaining
        self.model.setObjective(objective, GRB.MINIMIZE)
        self.model.optimize()
        return mm._gurobi_outcome(self.model, GRB)

    def value(self, variable):
        return (
            self.model.variableValue(variable) if self.solver == "highs" else variable.X
        )

    def selection(self):
        return (
            [e for e, x in self.edges.items() if self.value(x) > 0.5],
            [n for n, x in self.nodes.items() if self.value(x) > 0.5],
        )

    def objective_value(self):
        return (
            self.model.getObjectiveValue()
            if self.solver == "highs"
            else self.model.ObjVal
        )

    def close(self):
        if self.env is not None:
            self.model.dispose()
            self.env.dispose()


def solve_selection(model, coverage, decode, time_limit):
    """Share a deadline across coverage and cost phases, retaining valid incumbents."""
    started = perf_counter()
    deadline = started + time_limit
    status, incumbent = model.solve(
        -model.sum(coverage) if coverage else model.cost, deadline
    )
    if not incumbent:
        raise NoIncumbent(status)
    result = decode(*model.selection())
    if (
        not coverage
        and status == "optimal"
        and not math.isclose(
            result[2], model.objective_value(), rel_tol=1e-8, abs_tol=1e-7
        )
    ):
        raise RuntimeError(
            "Feature composition objective disagrees with original selections."
        )
    gap = 0 if status == "optimal" else math.inf
    if coverage:
        claimed = round(sum(model.value(x) for x in coverage))
        if result[3] < claimed:
            raise RuntimeError("Feature composition overstates connected requirements.")
        gap = math.inf
        if status == "optimal" and perf_counter() < deadline:
            model.add(model.sum(coverage) == claimed)
            cost_status, cost_incumbent = model.solve(model.cost, deadline)
            if cost_incumbent:
                candidate = decode(*model.selection())
                if cost_status == "optimal" and not math.isclose(
                    candidate[2], model.objective_value(), rel_tol=1e-8, abs_tol=1e-7
                ):
                    raise RuntimeError(
                        "Feature composition objective disagrees with original selections."
                    )
                if candidate[3] < claimed:
                    raise RuntimeError(
                        "Feature composition overstates connected requirements."
                    )
                if candidate[2] <= result[2]:
                    result = candidate
                if cost_status == "optimal" and result[3] == claimed:
                    gap = 0
    return result, gap, perf_counter() - started
