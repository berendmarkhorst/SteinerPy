import highspy as hp
import logging
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
        lhs = sum(y1[(u, w)] for u, w in steiner_problem.arcs if v == w)
        model.addConstr(lhs <= 1)

    # Constraint 3: connection between y1 and x
    for u, v in steiner_problem.edges:
        model.addConstr(y1[(u, v)] + y1[(v, u)] <= x[(u, v)])

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
            lhs = sum(y2[group_id_k, a] for a in steiner_problem.arcs if a[1] == t)
            model.addConstr(lhs == 0)

    # Constraint 7: indegree at most outdegree for Steiner points
    for v in steiner_problem.steiner_points:
        model.addConstr(sum(y1[(a[0], a[1])] for a in steiner_problem.arcs if a[1] == v) <=
                        sum(y1[(a[0], a[1])] for a in steiner_problem.arcs if a[0] == v))

    # Constraint 8: indegree at most outdegree per terminal group
    for group_id_k in group_indices:
        remaining_vertices = set(steiner_problem.nodes) - set(terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id_k))
        for v in remaining_vertices:
            model.addConstr(sum(y2[group_id_k, (a[0], v)] for a in steiner_problem.arcs if a[1] == v) <=
                            sum(y2[group_id_k, (v, a[1])] for a in steiner_problem.arcs if a[0] == v))

    # Constraint 9: connect y2 and z
    for group_id_k in group_indices:
        for group_id_l in group_indices:
            if group_id_l > group_id_k:
                model.addConstr(sum(y2[group_id_k, a] for a in steiner_problem.arcs if a[1] == steiner_problem.roots[group_id_l]) <= z[group_id_k, group_id_l])

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
                first_term = sum(f[group_id, t, a] for a in steiner_problem.arcs if a[0] == v)
                second_term = sum(f[group_id, t, a] for a in steiner_problem.arcs if a[1] == v)
                left_hand_side = first_term - second_term
                demand_and_supply = demand_and_supply_directed(steiner_problem, group_id, t, v, z)
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


def build_model(steiner_problem: 'SteinerProblem', time_limit: float = 300, logfile: str = "") -> Tuple[hp.HighsModel, hp.HighsVarType, hp.HighsVarType, hp.HighsVarType, hp.HighsVarType, Dict[str, hp.HighsVarType]]:
    """
    Returns the deterministic directed model.
    :param steiner_problem: SteinerProblem-object.
    :param time_limit: time limit in seconds for the HiGHS model. Default is 300 seconds.
    :param logfile: path to logfile.
    :return: HiGHS model.
    """
    # Create the model
    logging.info("Building the model.")

    model = make_model(time_limit, logfile)

    # Start tracking compilation time
    start_time = time.time()

    # Add constraints
    model, x, y1, y2, z = add_directed_constraints(model, steiner_problem)
    model, f = add_flow_constraints(model, steiner_problem, z, y2)

    # End tracking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    logging.info(f"Model built in {compilation_time:.2f} seconds.")

    return model, x, y1, y2, z, f


def run_model(model: hp.HighsModel, steiner_problem: 'SteinerProblem', x: hp.HighsVarType) -> Tuple[float, float, float, List[Tuple]]:
    """
    Solves the model and returns the result.
    :param model: highspy model.
    :param steiner_problem: SteinerProblem-object.
    :param x: highspy variable.
    :return: Solution-object.
    """
    logging.info(f"Started with running the model...")

    # Optimize model
    model.minimize(sum(x[e] * steiner_problem.graph.edges[e][steiner_problem.weight] for e in steiner_problem.edges))

    logging.info(f"Runtime: {model.getRunTime():.2f} seconds")

    selected_edges = [e for e in steiner_problem.edges if model.variableValue(x[e]) > 0.5]
    gap = model.getInfo().mip_gap
    runtime = model.getRunTime()
    objective = model.getObjectiveValue()

    return gap, runtime, objective, selected_edges