import highspy as hp
import logging
import networkx as nx
import time
from typing import List, Set, Tuple, Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO)

def make_model(time_limit: float, logfile: str = "") -> hp.HighsModel:
    """
    Creates a HiGHS model with the given time limit and logfile.
    :param time_limit: time limit in seconds for the HiGHS model.
    :param logfile: path to logfile.
    :return: HiGHS model.
    """
    # Create model
    model = hp.Highs()
    model.setOptionValue("time_limit", time_limit)
    model.setOptionValue("output_flag", False)  # Disable/enable console output

    # Clear the logfile and start logging
    if logfile != "":
        with open(logfile, "w") as _:
            pass
        model.setOptionValue("log_file", logfile)

    return model


def get_terminals(terminal_group: List[List]) -> List:
    """
    Turns a nested list of terminals into a list of terminals.
    :param terminal_group: nested list of terminals.
    :return: list of terminals.
    """
    return [t for group in terminal_group for t in group]

def terminal_groups_without_root(terminal_group: List[List], roots: List, group_index: int) -> Set:
    """
    Get terminal groups until index k without kth root.
    :param terminal_group: nested list of terminals.
    :param roots: list of roots.
    :param group_index: index of the terminal group.
    :return: subset of terminal groups from index k to K.
    """
    if len(terminal_group[0]) > 0:
        return set(get_terminals(terminal_group[group_index:])) - set([roots[group_index]])
    else:
        return set()

def get_terminal_groups_until_k(terminal_groups: List[List], group_index: int) -> Set:
    """
    Get terminal groups until index k.
    :param terminal_groups: nested list of terminals.
    :param group_index: index of the terminal group.
    :return: subset of terminal groups up till index k.
    """
    return set(get_terminals(terminal_groups[:group_index]))

def add_directed_constraints(model: hp.HighsModel, steiner_problem: 'SteinerProblem') -> Tuple[hp.HighsModel, Dict[str, hp.HighsVarType]]:
    """
    Adds DO-D constraints to the model (see Markhorst et al. 2025)
    :param model: HiGHS model.
    :param steiner_problem: AutomatedPipeRouting-object.
    :return: HiGHS model with DO-D constraints and decision variables.
    """
    # Sets
    group_indices = range(len(steiner_problem.terminal_groups))
    k_indices = [(k, l) for k in group_indices for l in group_indices if l >= k]

    # Decision variables
    x = {e: model.addVariable(0, 1, name=f"x[{e}]") for e in steiner_problem.edges}
    y1 = {a: model.addVariable(0, 1, name=f"y1[{a}]") for a in steiner_problem.arcs}
    y2 = {(group_id, a): model.addVariable(0, 1, name=f"y2[{group_id},{a}]") for group_id in group_indices
          for a in steiner_problem.arcs}
    z = {(k, l): model.addVariable(0, 1, name=f"z[{k},{l}]") for k, l in k_indices}

    for col in range(model.getNumCol()):
        model.changeColIntegrality(col, hp.HighsVarType.kInteger)

    # Constraint 1: connection between y2 and y1
    for group_id in group_indices:
        for a in steiner_problem.arcs:
            model.addConstr(y2[group_id, a] <= y1[a])

    # Constraint 2: indegree of each vertex cannot exceed 1
    for v in steiner_problem.nodes:
        incoming = [y1[a] for a in steiner_problem.arcs if a[1] == v]
        if incoming:
            model.addConstr(sum(incoming) <= 1)

    # Constraint 3: connection between y1 and x
    # For directed graphs only one arc direction exists per edge
    for u, v in steiner_problem.edges:
        if (v, u) in y1:
            model.addConstr(y1[(u, v)] + y1[(v, u)] <= x[(u, v)])
        else:
            model.addConstr(y1[(u, v)] <= x[(u, v)])

    # Constraint 4: enforce terminal group rooted at one root
    for group_id_k in group_indices:
        model.addConstr(sum(z[group_id_l, group_id_k] for group_id_l in group_indices
                            if group_id_l <= group_id_k) == 1)

    # Constraint 5: enforce one root per arborescence
    for group_id_k in group_indices:
        for group_id_l in group_indices:
            if group_id_l > group_id_k:
                model.addConstr(z[group_id_k, group_id_k] >= z[group_id_k, group_id_l])

    # Constraint 6: terminals in T^{1···k−1} cannot attach to root r k
    for group_id_k in group_indices:
        for t in get_terminal_groups_until_k(steiner_problem.terminal_groups, group_id_k):
            incoming = [y2[group_id_k, a] for a in steiner_problem.arcs if a[1] == t]
            if incoming:
                model.addConstr(sum(incoming) == 0)

    # Constraint 7: indegree at most outdegree for Steiner points
    for v in steiner_problem.steiner_points:
        in_arcs = [y1[a] for a in steiner_problem.arcs if a[1] == v]
        out_arcs = [y1[a] for a in steiner_problem.arcs if a[0] == v]
        if in_arcs:
            out_degree_sum = sum(out_arcs) if out_arcs else 0
            model.addConstr(sum(in_arcs) <= out_degree_sum)

    # Constraint 8: indegree at most outdegree per terminal group
    for group_id_k in group_indices:
        remaining_vertices = set(steiner_problem.nodes) - set(terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id_k))
        for v in remaining_vertices:
            in_arcs = [y2[group_id_k, a] for a in steiner_problem.arcs if a[1] == v]
            out_arcs = [y2[group_id_k, a] for a in steiner_problem.arcs if a[0] == v]
            if in_arcs:
                out_degree_sum = sum(out_arcs) if out_arcs else 0
                model.addConstr(sum(in_arcs) <= out_degree_sum)

    # Constraint 9: connect y2 and z
    for group_id_k in group_indices:
        for group_id_l in group_indices:
            if group_id_l > group_id_k:
                incoming = [y2[group_id_k, a] for a in steiner_problem.arcs if a[1] == steiner_problem.roots[group_id_l]]
                if incoming:
                    model.addConstr(sum(incoming) <= z[group_id_k, group_id_l])

    return model, x, y1, y2, z


def demand_and_supply_directed(steiner_problem: 'SteinerProblem', group_id_k: int, t: Tuple, v: Tuple, z: hp.HighsVarType) -> Union[hp.HighsVarType, int]:
    """
    Calculate the demand and supply for a directed model.
    :param cc_k: The current connected component.
    :param t: A terminal represented as a tuple of integers.
    :param v: A vertex represented as a tuple of integers.
    :param z: The decision variable z.
    :return: The value of z if the vertex is the root, -z if the vertex is a terminal, and 0 otherwise.
    """

    # We assume terminals are disjoint from each other
    group_id_l = [group_id for group_id, group in enumerate(steiner_problem.terminal_groups) if t in group][0]

    if v == steiner_problem.roots[group_id_k]:
        return z[(group_id_k, group_id_l)]
    elif v == t:
        return -z[(group_id_k, group_id_l)]
    else:
        return 0


def add_flow_constraints(model: hp.HighsModel, steiner_problem: 'SteinerProblem', z: hp.HighsVarType, y2: hp.HighsVarType) -> Tuple[hp.HighsModel, Dict[str, hp.HighsVarType]]:
    """
    We add the flow constraints to the HiGHS model.
    :param model: HiGHS model.
    :param steiner_problem: SteinerProblem-object.
    :param z: decision variable z.
    :param y2: decision variable y2.
    :return: HiGHS model and variable(s).
    """
    # Decision variables (binary flow variables)
    group_indices = range(len(steiner_problem.terminal_groups))
    f = {(group_id, t, a): model.addVariable(0, 1, hp.HighsVarType.kInteger, name=f"f[{group_id},{a}]") for group_id in group_indices
          for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id) for a in steiner_problem.arcs}

    # Constraint 1: flow conservation
    for v in steiner_problem.nodes:
        for group_id in group_indices:
            for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id):
                out_arcs = [a for a in steiner_problem.arcs if a[0] == v]
                in_arcs = [a for a in steiner_problem.arcs if a[1] == v]
                demand_and_supply = demand_and_supply_directed(steiner_problem, group_id, t, v, z)
                # demand_and_supply is either a HiGHS variable (root/terminal) or the integer 0.
                # When the node has no incident arcs and the demand is zero (isolated, non-source/sink),
                # the constraint is trivially satisfied and can be skipped.
                is_highs_expr = not isinstance(demand_and_supply, (int, float))
                has_arcs = bool(out_arcs or in_arcs)
                if not has_arcs and not is_highs_expr:
                    continue  # Isolated node with no demand: trivially satisfied
                first_term = sum(f[group_id, t, a] for a in out_arcs) if out_arcs else 0
                second_term = sum(f[group_id, t, a] for a in in_arcs) if in_arcs else 0
                left_hand_side = first_term - second_term
                model.addConstr(left_hand_side == demand_and_supply)

    # Constraint 2: connection between f and y2
    for group_id in group_indices:
        for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id):
            for a in steiner_problem.arcs:
                left_hand_side = f[group_id, t, a]
                right_hand_side = y2[group_id, a]
                model.addConstr(left_hand_side <= right_hand_side)

    # Constraint 3: prevent flow from leaving a terminal
    for group_id in group_indices:
        for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id):
            if sum(1 for u, v in steiner_problem.arcs if u == t) > 0:
                left_hand_side = sum(f[group_id, t, (u, v)] for u, v in steiner_problem.arcs if u == t)
                model.addConstr(left_hand_side == 0, name="flow_3")

    return model, f


def add_optional_flow_constraints(
    model: hp.HighsModel,
    steiner_problem: 'SteinerProblem',
    y2: Dict,
    connection_vars: Dict,
) -> Tuple[hp.HighsModel, Dict]:
    """
    Flow constraints where terminal connectivity is optional.

    For each non-root terminal t in group k, *connection_vars[(k, t)]* is a binary
    variable that is 1 when the terminal is connected and 0 when it is not.
    The flow demand is scaled by this variable so that no flow is required for
    disconnected (penalised) terminals.

    :param connection_vars: dict (group_id, terminal) -> HiGHS binary variable
    :return: model with added constraints and flow variable dict f.
    """
    group_indices = range(len(steiner_problem.terminal_groups))
    f = {
        (group_id, t, a): model.addVariable(0, 1, hp.HighsVarType.kInteger, name=f"f_opt[{group_id},{a}]")
        for group_id in group_indices
        for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id)
        for a in steiner_problem.arcs
    }

    # Constraint 1: optional flow conservation
    # demand = connection_var at root, -connection_var at terminal, 0 elsewhere
    for v in steiner_problem.nodes:
        for group_id in group_indices:
            for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id):
                out_arcs = [a for a in steiner_problem.arcs if a[0] == v]
                in_arcs = [a for a in steiner_problem.arcs if a[1] == v]
                c = connection_vars[(group_id, t)]

                if v == steiner_problem.roots[group_id]:
                    demand = c
                elif v == t:
                    demand = -c
                else:
                    # demand = 0 (flow conservation); skip when node has no arcs
                    if not out_arcs and not in_arcs:
                        continue
                    demand = 0

                first_term = sum(f[group_id, t, a] for a in out_arcs) if out_arcs else 0
                second_term = sum(f[group_id, t, a] for a in in_arcs) if in_arcs else 0
                model.addConstr(first_term - second_term == demand)

    # Constraint 2: flow can only use selected arcs
    for group_id in group_indices:
        for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id):
            for a in steiner_problem.arcs:
                model.addConstr(f[group_id, t, a] <= y2[group_id, a])

    # Constraint 3: no flow leaving a terminal
    for group_id in group_indices:
        for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id):
            out_arcs = [(u, v) for u, v in steiner_problem.arcs if u == t]
            if out_arcs:
                model.addConstr(sum(f[group_id, t, a] for a in out_arcs) == 0)

    return model, f


def find_violated_cuts(
    steiner_problem: 'SteinerProblem',
    y2: Dict,
    z: Dict,
    model: hp.HighsModel,
    eps: float = 1e-6,
) -> List[Tuple[int, int, List[Tuple]]]:
    """
    Find violated directed cut constraints for the current LP/MIP solution.

    For each pair (group_id_k, group_id_l) with k <= l and for each terminal t
    in terminal_groups[group_id_l], checks whether the directed cut from
    roots[group_id_k] to t is satisfied.  A cut is violated when the minimum
    cut value is strictly less than z[k, l].

    :param steiner_problem: SteinerProblem-object.
    :param y2: per-group arc variables {(group_id, arc): var}.
    :param z: connectivity variables {(k, l): var}.
    :param model: HiGHS model (used to read current variable values).
    :param eps: numerical tolerance / creep-flow added to each arc capacity.
    :return: list of (group_id_k, group_id_l, cut_arcs) for each violated cut.
    """
    # No terminals → nothing to check
    if len(steiner_problem.terminal_groups[0]) == 0:
        return []

    group_indices = range(len(steiner_problem.terminal_groups))
    violated_cuts = []

    for group_id_l in group_indices:
        for group_id_k in range(group_id_l + 1):  # k <= l
            root_k = steiner_problem.roots[group_id_k]
            z_val = model.variableValue(z[(group_id_k, group_id_l)])

            if z_val < eps:
                continue  # z = 0 → no connectivity required for this pair

            # Build a directed graph with y2 capacities for group k.
            # A small eps is added to each capacity for numerical stability in
            # the minimum-cut computation (prevents division-by-zero issues).
            digraph = nx.DiGraph()
            for (u, v) in steiner_problem.arcs:
                capacity = model.variableValue(y2[(group_id_k, (u, v))]) + eps
                digraph.add_edge(u, v, capacity=capacity)

            for t in steiner_problem.terminal_groups[group_id_l]:
                if t == root_k:
                    continue

                try:
                    cut_value, partition = nx.minimum_cut(
                        digraph, root_k, t, capacity="capacity"
                    )
                except nx.NetworkXError:
                    # No path exists — treat as a zero-capacity cut
                    cut_value = 0.0
                    partition = (
                        {root_k},
                        set(steiner_problem.nodes) - {root_k},
                    )

                if cut_value < z_val - eps:
                    cut_arcs = [
                        (u, v)
                        for (u, v) in steiner_problem.arcs
                        if u in partition[0] and v in partition[1]
                    ]
                    if not cut_arcs:
                        logging.warning(
                            f"Empty cut for group_k={group_id_k}, group_l={group_id_l}, "
                            f"terminal={t}: no arc exists from the root side to the terminal "
                            f"side. Forcing z[{group_id_k},{group_id_l}] = 0."
                        )
                    violated_cuts.append((group_id_k, group_id_l, cut_arcs))

    return violated_cuts


def build_model(steiner_problem: 'SteinerProblem', time_limit: float = 300, logfile: str = "") -> Tuple[hp.HighsModel, hp.HighsVarType, hp.HighsVarType, hp.HighsVarType, hp.HighsVarType]:
    """
    Returns the deterministic directed model without flow variables.
    Connectivity is enforced lazily via directed cut constraints added during
    solving (see :func:`run_model`).

    :param steiner_problem: SteinerProblem-object.
    :param time_limit: time limit in seconds for the HiGHS model. Default is 300 seconds.
    :param logfile: path to logfile.
    :return: (model, x, y1, y2, z) – HiGHS model and decision variables.
    """
    # Create the model
    logging.info("Building the model.")

    model = make_model(time_limit, logfile)

    # Start tracking compilation time
    start_time = time.time()

    # Add structural constraints (no flow variables)
    model, x, y1, y2, z = add_directed_constraints(model, steiner_problem)

    # Add degree constraints if max_degree is specified
    if getattr(steiner_problem, 'max_degree', None) is not None:
        add_degree_constraints(model, steiner_problem, x)

    # End tracking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    logging.info(f"Model built in {compilation_time:.2f} seconds.")

    return model, x, y1, y2, z


def run_model(model: hp.HighsModel, steiner_problem: 'SteinerProblem', x: hp.HighsVarType, y2: Dict, z: Dict) -> Tuple[float, float, float, List[Tuple]]:
    """
    Solves the model using an iterative cut-generation (lazy-cut) approach and
    returns the result.

    Instead of adding flow variables and constraints upfront, the solver is run
    repeatedly.  After each solve, violated directed cut constraints are
    identified via a minimum-cut computation (networkx) and appended to the
    model before the next solve.  The process terminates when no violated cut
    is found, guaranteeing that the returned solution is feasible.

    :param model: HiGHS model (built by :func:`build_model`).
    :param steiner_problem: SteinerProblem-object.
    :param x: edge selection decision variables.
    :param y2: per-group arc variables {(group_id, arc): var}.
    :param z: connectivity variables {(k, l): var}.
    :return: (gap, runtime, objective, selected_edges).
             Note: *runtime* covers only the iterative solve loop and does not
             include the model compilation time reported by :func:`build_model`.
    """
    logging.info("Started running the model with iterative cut generation.")

    objective_expr = sum(
        x[e] * steiner_problem.graph.edges[e][steiner_problem.weight]
        for e in steiner_problem.edges
    )

    start_time = time.time()

    while True:
        model.minimize(objective_expr)

        violated_cuts = find_violated_cuts(steiner_problem, y2, z, model)

        if not violated_cuts:
            break  # Solution is feasible with respect to all cut constraints

        # Add each violated cut as a new constraint: sum(y2[k,a] for a in cut) >= z[k,l]
        for group_id_k, group_id_l, cut_arcs in violated_cuts:
            lhs = sum(y2[(group_id_k, a)] for a in cut_arcs) if cut_arcs else 0
            model.addConstr(lhs >= z[(group_id_k, group_id_l)])

        logging.info(f"Added {len(violated_cuts)} violated cut(s), re-solving.")

    runtime = time.time() - start_time
    logging.info(f"Runtime: {runtime:.2f} seconds")

    selected_edges = [e for e in steiner_problem.edges if model.variableValue(x[e]) > 0.5]
    gap = model.getInfo().mip_gap
    objective = model.getObjectiveValue()

    return gap, runtime, objective, selected_edges


def add_degree_constraints(model: hp.HighsModel, steiner_problem: 'SteinerProblem', x: Dict) -> None:
    """
    Add maximum degree constraints to the model.
    For each node v: sum of x[e] for all edges e incident to v <= max_degree.
    :param model: HiGHS model.
    :param steiner_problem: SteinerProblem-object with max_degree attribute.
    :param x: edge selection decision variables.
    """
    max_degree = steiner_problem.max_degree
    for v in steiner_problem.nodes:
        incident = [x[e] for e in steiner_problem.edges if v in e]
        if incident:
            model.addConstr(sum(incident) <= max_degree)

def build_prize_collecting_model(steiner_problem: 'PrizeCollectingProblem', time_limit: float = 300, logfile: str = "") -> Tuple[hp.HighsModel, hp.HighsVarType, hp.HighsVarType, hp.HighsVarType, hp.HighsVarType, Dict[str, hp.HighsVarType], Dict[str, hp.HighsVarType], Dict[str, hp.HighsVarType]]:
    """
    Build prize collecting model by extending the base Steiner model.
    """
    # Start with the base model (reuse existing code)
    model, x, y1, y2, z = build_model(steiner_problem, time_limit, logfile)

    # Prize-collecting needs explicit flow constraints for connectivity enforcement
    model, f = add_flow_constraints(model, steiner_problem, z, y2)
    
    # Add prize collecting specific variables
    group_indices = range(len(steiner_problem.terminal_groups))
    
    # Node selection variables (whether we collect prize from a node)
    node_vars = {node: model.addVariable(0, 1, hp.HighsVarType.kInteger, name=f"node[{node}]") 
                 for node in steiner_problem.nodes}
    
    # Terminal penalty variables (penalty for not connecting a terminal)
    penalty_vars = {}
    for group_id in group_indices:
        for terminal in steiner_problem.terminal_groups[group_id]:
            penalty_vars[(group_id, terminal)] = model.addVariable(
                0, 1, hp.HighsVarType.kInteger, name=f"penalty[{group_id},{terminal}]"
            )
    
    # Add prize collecting constraints
    add_prize_collecting_constraints(model, steiner_problem, node_vars, penalty_vars, y1)
    
    return model, x, y1, y2, z, f, node_vars, penalty_vars


def add_prize_collecting_constraints(model: hp.HighsModel, steiner_problem: 'PrizeCollectingProblem', 
                                   node_vars: Dict, penalty_vars: Dict, y1: hp.HighsVarType):
    """
    Add prize collecting specific constraints to the base model.
    """
    group_indices = range(len(steiner_problem.terminal_groups))
    
    # Constraint: Node can only be selected if it's in the tree
    for node in steiner_problem.nodes:
        # If node is selected, it must have at least one incident edge (or be a terminal)
        incident_arcs = [y1[arc] for arc in steiner_problem.arcs 
                        if arc[0] == node or arc[1] == node]
        if incident_arcs:
            model.addConstr(node_vars[node] <= sum(incident_arcs) + 
                           sum(1 for group in steiner_problem.terminal_groups if node in group))
    
    # Constraint: Terminal connection or penalty
    for group_id in group_indices:
        # Treat the root of each group as connected by definition to avoid
        # forcing a penalty solely due to indegree 0 in the base model.
        group_root = getattr(steiner_problem, "roots", None)
        group_root = group_root[group_id] if group_root is not None else None
        for terminal in steiner_problem.terminal_groups[group_id]:
            # Terminal is either connected (in tree) or we pay penalty
            if group_root is not None and terminal == group_root:
                # Root is considered connected, even if it has indegree 0
                is_connected = 1
            else:
                is_connected = sum(
                    y1[arc] for arc in steiner_problem.arcs if arc[1] == terminal
                )
            model.addConstr(is_connected + penalty_vars[(group_id, terminal)] >= 1)
    
    # Optional: Budget constraint on total penalties
    if hasattr(steiner_problem, 'penalty_budget') and steiner_problem.penalty_budget is not None:
        total_penalties = sum(penalty_vars.values())
        model.addConstr(total_penalties <= steiner_problem.penalty_budget)


def run_prize_collecting_model(model: hp.HighsModel, steiner_problem: 'PrizeCollectingProblem', 
                              x: hp.HighsVarType, node_vars: Dict, penalty_vars: Dict) -> Tuple[float, float, float, List[Tuple], List[str], Dict]:
    """
    Solve prize collecting model and extract solution.
    """
    logging.info("Started running the prize collecting model...")
    
    # Build objective: edge costs - node prizes + penalties
    objective_expr = sum(x[e] * steiner_problem.graph.edges[e][steiner_problem.weight] 
                        for e in steiner_problem.edges)
    
    # Subtract node prizes
    for node in steiner_problem.nodes:
        prize = steiner_problem.node_prizes.get(node, 0)
        if prize > 0:
            objective_expr -= node_vars[node] * prize
    
    # Add penalty costs
    penalty_cost = getattr(steiner_problem, 'penalty_cost', 1000)
    for penalty_var in penalty_vars.values():
        objective_expr += penalty_var * penalty_cost
    
    # Minimize the objective
    model.minimize(objective_expr)
    
    logging.info(f"Runtime: {model.getRunTime():.2f} seconds")
    
    # Extract solution
    selected_edges = [e for e in steiner_problem.edges if model.variableValue(x[e]) > 0.5]
    selected_nodes = [node for node in steiner_problem.nodes if model.variableValue(node_vars[node]) > 0.5]
    
    penalties = {}
    for (group_id, terminal), var in penalty_vars.items():
        var_value = model.variableValue(var)
        if var_value > 0.5:
            penalties[f"group_{group_id}_{terminal}"] = penalty_cost * var_value
    
    gap = model.getInfo().mip_gap
    runtime = model.getRunTime()
    objective = model.getObjectiveValue()
    
    return gap, runtime, objective, selected_edges, selected_nodes, penalties


def build_budget_model(steiner_problem: 'BaseSteinerProblem', time_limit: float = 300, logfile: str = "") -> Tuple:
    """
    Build a budget-constrained Steiner model.
    Objective: maximize connected terminals (minimize unconnected terminals).
    Constraint: total edge cost <= budget.
    :param steiner_problem: SteinerProblem-object with a ``budget`` attribute.
    :param time_limit: time limit in seconds.
    :param logfile: path to logfile.
    :return: HiGHS model and decision variables.
    """
    model = make_model(time_limit, logfile)
    group_indices = range(len(steiner_problem.terminal_groups))

    model, x, y1, y2, z = add_directed_constraints(model, steiner_problem)

    if getattr(steiner_problem, 'max_degree', None) is not None:
        add_degree_constraints(model, steiner_problem, x)

    # Terminal penalty variables (1 if terminal is NOT connected)
    # and connection variables (1 if terminal IS connected)
    penalty_vars = {}
    connection_vars = {}
    for group_id in group_indices:
        group_root = steiner_problem.roots[group_id]
        for terminal in steiner_problem.terminal_groups[group_id]:
            if terminal == group_root:
                # Root is always connected; no penalty variable needed
                continue
            penalty_vars[(group_id, terminal)] = model.addVariable(
                0, 1, hp.HighsVarType.kInteger, name=f"penalty[{group_id},{terminal}]"
            )
            connection_vars[(group_id, terminal)] = model.addVariable(
                0, 1, hp.HighsVarType.kInteger, name=f"conn[{group_id},{terminal}]"
            )
            # Exactly one of connected or penalised
            model.addConstr(connection_vars[(group_id, terminal)] + penalty_vars[(group_id, terminal)] == 1)

    # Flow constraints that respect the optional connectivity
    model, f = add_optional_flow_constraints(model, steiner_problem, y2, connection_vars)

    # Budget constraint: total edge cost <= budget
    model.addConstr(
        sum(x[e] * steiner_problem.graph.edges[e][steiner_problem.weight]
            for e in steiner_problem.edges) <= steiner_problem.budget
    )

    return model, x, y1, y2, z, f, penalty_vars


def run_budget_model(model: hp.HighsModel, steiner_problem: 'BaseSteinerProblem',
                     x: Dict, penalty_vars: Dict) -> Tuple[float, float, int, List[Tuple], Dict]:
    """
    Solve budget-constrained model: minimize number of unconnected terminals.
    :param model: HiGHS model.
    :param steiner_problem: SteinerProblem-object with a ``budget`` attribute.
    :param x: edge selection variables.
    :param penalty_vars: terminal penalty variables.
    :return: gap, runtime, connected_count, selected_edges, penalties.
    """
    logging.info("Started running the budget-constrained model...")

    model.minimize(sum(penalty_vars.values()))

    logging.info(f"Runtime: {model.getRunTime():.2f} seconds")

    selected_edges = [e for e in steiner_problem.edges if model.variableValue(x[e]) > 0.5]

    penalties = {}
    for (group_id, terminal), var in penalty_vars.items():
        if model.variableValue(var) > 0.5:
            penalties[f"group_{group_id}_{terminal}"] = 1

    total_terminals = sum(len(g) for g in steiner_problem.terminal_groups)
    connected_count = total_terminals - len(penalties)

    gap = model.getInfo().mip_gap
    runtime = model.getRunTime()

    return gap, runtime, connected_count, selected_edges, penalties