import highspy as hp
import logging
import math
import networkx as nx
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Set, Tuple, Dict, Union

from ._fastgraph import HAS_SCIPY, get_arc_csr, min_cut_scipy, cpu_count

# Configure logging
logging.basicConfig(level=logging.INFO)


def _resolve_threads(threads) -> int:
    """Resolve a HiGHS/Gurobi thread count.

    ``None`` -> read ``STEINERPY_THREADS`` env var, else 0 (solver default: all
    cores).  ``0`` keeps the solver default.  A positive int is used verbatim.
    """
    if threads is None:
        env = os.environ.get("STEINERPY_THREADS")
        if env is not None:
            try:
                return max(0, int(env))
            except ValueError:
                return 0
        return 0
    try:
        return max(0, int(threads))
    except (TypeError, ValueError):
        return 0


# Separation parallelism: number of worker threads for the per-terminal min-cut
# computations.  scipy's maximum_flow releases the GIL, so threads overlap.
def _sep_thread_count() -> int:
    env = os.environ.get("STEINERPY_SEP_THREADS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(8, cpu_count()))


# Below this many independent min-cut tasks per group it is not worth paying the
# thread-pool dispatch overhead; run them serially instead.
_SEP_PARALLEL_MIN_TASKS = 4


def _nested_cut_rounds() -> int:
    """Extra *nested cuts* per violated terminal (Koch & Martin 1998, Sect. 4).

    After a violated minimum cut is found, its arcs' capacities are raised to 1
    (as if fully selected) and the max-flow is re-run; while the new minimum cut
    is still violated it is added as well.  Each round yields a structurally
    different cut for the price of one extra max-flow, which typically collapses
    the number of LP/MIP re-solve rounds in the cut loop.  Configure with
    ``STEINERPY_NESTED_CUTS`` (0 disables); only extra max-flows are spent on
    terminals whose first cut was violated.

    The default is 1: on the HiGHS path every added row is carried through a
    full MIP re-solve each round, and benchmarking showed one nested cut per
    violated terminal is a consistent win (~1.25x on tree and forest) while
    three bloat the model enough to cost more than the rounds they save.
    """
    env = os.environ.get("STEINERPY_NESTED_CUTS")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return 1


def _lp_cut_rounds() -> int:
    """Maximum rounds of cut separation on the LP *relaxation* (HiGHS path).

    Before the integer cut loop starts, connectivity cuts are separated on the
    LP relaxation: each round is a cheap LP re-solve instead of a full
    branch-and-bound run, and the cuts found strengthen the root bound of every
    subsequent MIP solve — the standard root-separation scheme of
    branch-and-cut Steiner codes (Koch & Martin 1998).  Configure with
    ``STEINERPY_LP_CUT_ROUNDS`` (0 disables the LP phase).
    """
    env = os.environ.get("STEINERPY_LP_CUT_ROUNDS")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return 50


def make_model(time_limit: float, logfile: str = "", threads=None) -> hp.HighsModel:
    """
    Creates a HiGHS model with the given time limit and logfile.

    :param time_limit: time limit in seconds for the HiGHS model.
    :param logfile: path to logfile.
    :param threads: HiGHS thread count.  ``None`` -> ``STEINERPY_THREADS`` env or
        the solver default (all cores); ``0`` -> solver default; positive int ->
        that many threads.  Multithreading parallelises the MIP branch-and-bound.
    :return: HiGHS model.
    """
    # Create model
    model = hp.Highs()
    model.setOptionValue("time_limit", time_limit)
    model.setOptionValue("output_flag", False)  # Disable/enable console output

    nthreads = _resolve_threads(threads)
    try:
        model.setOptionValue("threads", nthreads)
        # Enable parallel MIP/simplex unless the caller pinned a single thread.
        if nthreads != 1:
            model.setOptionValue("parallel", "on")
    except Exception:  # pragma: no cover - older highspy without the option
        pass

    # Clear the logfile and start logging
    if logfile != "":
        with open(logfile, "w") as _:
            pass
        model.setOptionValue("log_file", logfile)

    return model


def _arc_adjacency(arcs) -> Tuple[Dict, Dict]:
    """Per-node incoming/outgoing arc lists, built in one pass over ``arcs``.

    The model builders need "arcs entering v" / "arcs leaving v" for many nodes
    (and, for the forest model, for every terminal group); scanning the full arc
    list per node makes model construction O(|V|·|A|).  Building the adjacency
    once keeps it O(|A|).

    :return: ``(in_arcs, out_arcs)`` dicts mapping node -> list of arcs.
    """
    in_arcs: Dict = {}
    out_arcs: Dict = {}
    for a in arcs:
        out_arcs.setdefault(a[0], []).append(a)
        in_arcs.setdefault(a[1], []).append(a)
    return in_arcs, out_arcs


def _incident_edges(edges) -> Dict:
    """Per-node incident edge lists, built in one pass over ``edges``."""
    incident: Dict = {}
    for e in edges:
        incident.setdefault(e[0], []).append(e)
        incident.setdefault(e[1], []).append(e)
    return incident


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
    in_arcs, out_arcs = _arc_adjacency(steiner_problem.arcs)

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
        incoming = [y1[a] for a in in_arcs.get(v, ())]
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
            incoming = [y2[group_id_k, a] for a in in_arcs.get(t, ())]
            if incoming:
                model.addConstr(sum(incoming) == 0)

    # Constraint 7: indegree at most outdegree for Steiner points
    for v in steiner_problem.steiner_points:
        entering = [y1[a] for a in in_arcs.get(v, ())]
        leaving = [y1[a] for a in out_arcs.get(v, ())]
        if entering:
            out_degree_sum = sum(leaving) if leaving else 0
            model.addConstr(sum(entering) <= out_degree_sum)

    # Constraint 8: indegree at most outdegree per terminal group
    for group_id_k in group_indices:
        remaining_vertices = set(steiner_problem.nodes) - set(terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id_k))
        for v in remaining_vertices:
            entering = [y2[group_id_k, a] for a in in_arcs.get(v, ())]
            leaving = [y2[group_id_k, a] for a in out_arcs.get(v, ())]
            if entering:
                out_degree_sum = sum(leaving) if leaving else 0
                model.addConstr(sum(entering) <= out_degree_sum)

    # Constraint 9: connect y2 and z
    for group_id_k in group_indices:
        for group_id_l in group_indices:
            if group_id_l > group_id_k:
                incoming = [y2[group_id_k, a] for a in in_arcs.get(steiner_problem.roots[group_id_l], ())]
                if incoming:
                    model.addConstr(sum(incoming) <= z[group_id_k, group_id_l])

    return model, x, y1, y2, z


def demand_and_supply_directed(steiner_problem: 'SteinerProblem', group_id_k: int, t: Tuple, v: Tuple, z: hp.HighsVarType, t_group: int = None) -> Union[hp.HighsVarType, int]:
    """
    Calculate the demand and supply for a directed model.

    :param cc_k: The current connected component.
    :param t: A terminal represented as a tuple of integers.
    :param v: A vertex represented as a tuple of integers.
    :param z: The decision variable z.
    :param t_group: index of the terminal group containing ``t``; looked up when
        omitted (callers in a loop should precompute it once per terminal).
    :return: The value of z if the vertex is the root, -z if the vertex is a terminal, and 0 otherwise.
    """

    # We assume terminals are disjoint from each other
    if t_group is None:
        t_group = [group_id for group_id, group in enumerate(steiner_problem.terminal_groups) if t in group][0]

    if v == steiner_problem.roots[group_id_k]:
        return z[(group_id_k, t_group)]
    elif v == t:
        return -z[(group_id_k, t_group)]
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
    # Decision variables: continuous flow in [0, 1].  Integrality of f is not
    # needed: for any fixed integer values of y2 and z each (group, t) block is
    # a unit s-t flow problem with integral capacities, and f never enters the
    # objective — so relaxing f keeps the model exact while removing
    # O(|T|·|A|) integer columns from branch-and-bound.
    group_indices = range(len(steiner_problem.terminal_groups))
    f = {(group_id, t, a): model.addVariable(0, 1, name=f"f[{group_id},{a}]") for group_id in group_indices
          for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id) for a in steiner_problem.arcs}

    in_arcs, out_arcs = _arc_adjacency(steiner_problem.arcs)
    t_to_group = {t: gid for gid, group in enumerate(steiner_problem.terminal_groups)
                  for t in group}

    # Constraint 1: flow conservation
    for v in steiner_problem.nodes:
        arcs_out = out_arcs.get(v, ())
        arcs_in = in_arcs.get(v, ())
        for group_id in group_indices:
            for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id):
                demand_and_supply = demand_and_supply_directed(
                    steiner_problem, group_id, t, v, z, t_group=t_to_group[t])
                # demand_and_supply is either a HiGHS variable (root/terminal) or the integer 0.
                # When the node has no incident arcs and the demand is zero (isolated, non-source/sink),
                # the constraint is trivially satisfied and can be skipped.
                is_highs_expr = not isinstance(demand_and_supply, (int, float))
                has_arcs = bool(arcs_out or arcs_in)
                if not has_arcs and not is_highs_expr:
                    continue  # Isolated node with no demand: trivially satisfied
                first_term = sum(f[group_id, t, a] for a in arcs_out) if arcs_out else 0
                second_term = sum(f[group_id, t, a] for a in arcs_in) if arcs_in else 0
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
            terminal_out = out_arcs.get(t, ())
            if terminal_out:
                left_hand_side = sum(f[group_id, t, a] for a in terminal_out)
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
    # Continuous flow in [0, 1] — see add_flow_constraints: with integral y2 and
    # connection variables, a feasible continuous unit flow exists iff an
    # integral one does, and f never enters the objective.
    f = {
        (group_id, t, a): model.addVariable(0, 1, name=f"f_opt[{group_id},{a}]")
        for group_id in group_indices
        for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id)
        for a in steiner_problem.arcs
    }

    in_arcs, out_arcs = _arc_adjacency(steiner_problem.arcs)

    # Constraint 1: optional flow conservation
    # demand = connection_var at root, -connection_var at terminal, 0 elsewhere
    for v in steiner_problem.nodes:
        arcs_out = out_arcs.get(v, ())
        arcs_in = in_arcs.get(v, ())
        for group_id in group_indices:
            for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id):
                c = connection_vars[(group_id, t)]

                if v == steiner_problem.roots[group_id]:
                    demand = c
                elif v == t:
                    demand = -c
                else:
                    # demand = 0 (flow conservation); skip when node has no arcs
                    if not arcs_out and not arcs_in:
                        continue
                    demand = 0

                first_term = sum(f[group_id, t, a] for a in arcs_out) if arcs_out else 0
                second_term = sum(f[group_id, t, a] for a in arcs_in) if arcs_in else 0
                model.addConstr(first_term - second_term == demand)

    # Constraint 2: flow can only use selected arcs
    for group_id in group_indices:
        for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id):
            for a in steiner_problem.arcs:
                model.addConstr(f[group_id, t, a] <= y2[group_id, a])

    # Constraint 3: no flow leaving a terminal
    for group_id in group_indices:
        for t in terminal_groups_without_root(steiner_problem.terminal_groups, steiner_problem.roots, group_id):
            terminal_out = out_arcs.get(t, ())
            if terminal_out:
                model.addConstr(sum(f[group_id, t, a] for a in terminal_out) == 0)

    return model, f


def _residual_source_set(residual, source, eps: float) -> Set:
    """Nodes reachable from *source* in residual network *residual* over arcs
    with strictly positive residual capacity (capacity - flow > eps).

    This is the source side S of a minimum cut; the corresponding cut is the set
    of outgoing arcs delta+(S).
    """
    seen = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for v in residual.successors(u):
            attr = residual[u][v]
            if v not in seen and attr["capacity"] - attr["flow"] > eps:
                seen.add(v)
                stack.append(v)
    return seen


def _residual_sink_set(residual, sink, eps: float) -> Set:
    """Nodes that can reach *sink* in residual network *residual* over arcs with
    strictly positive residual capacity (reverse traversal).

    The back cut is the complement S_bar = V \\ (this set): a second minimum cut
    that hugs the terminal side instead of the root side.
    """
    seen = {sink}
    stack = [sink]
    while stack:
        v = stack.pop()
        for u in residual.predecessors(v):
            attr = residual[u][v]
            if u not in seen and attr["capacity"] - attr["flow"] > eps:
                seen.add(u)
                stack.append(u)
    return seen


def _emit_cuts_for_terminal(group_id_k, group_id_l, t, cut_value, z_val,
                            source_sides, csr, eps, node_cuts):
    """Build the (k, l, cut_arcs) tuples for one terminal's min cut.

    ``source_sides`` is a list of root-side node sets (index sets when ``csr`` is
    a fast :class:`ArcCSR`, node-name sets for the networkx fallback).
    ``node_cuts`` selects which interpretation to use.  Returns a list of tuples
    (possibly empty when the cut is already satisfied).
    """
    if cut_value >= z_val - eps:
        return []  # cut satisfied
    out = []
    seen_arc_sets: Set = set()
    for side in source_sides:
        if node_cuts:
            cut_arcs = [(u, v) for (u, v) in node_cuts if u in side and v not in side]
        else:
            cut_arcs = csr.cut_arcs(side)
        key = frozenset(cut_arcs)
        if key in seen_arc_sets:
            continue  # duplicate of an already-emitted cut
        seen_arc_sets.add(key)
        if not cut_arcs:
            logging.warning(
                f"Empty cut for group_k={group_id_k}, group_l={group_id_l}, "
                f"terminal={t}: no arc exists from the root side to the terminal "
                f"side. Forcing z[{group_id_k},{group_id_l}] = 0."
            )
        out.append((group_id_k, group_id_l, cut_arcs))
    return out


def _group_cuts_scipy(steiner_problem, csr, group_id_k, tasks, y2_vals,
                      eps, back_cuts, threads):
    """Min-cut separation for one source group ``k`` using scipy max-flow.

    ``tasks`` is a list of ``(group_id_l, terminal, z_val)`` to check from
    ``roots[group_id_k]``.  The capacity CSR is built **once** for the whole
    group (capacities depend only on ``k``); the per-terminal min cuts are
    independent and run concurrently in a thread pool (scipy releases the GIL).
    """
    import numpy as np
    root_k = steiner_problem.roots[group_id_k]
    ni = csr.node_index
    src_idx = ni.get(root_k)
    if src_idx is None:
        return []

    cap = np.fromiter(
        (y2_vals[(group_id_k, a)] + eps for a in csr.arcs),
        dtype=np.float64, count=len(csr.arcs),
    )
    int_csr = csr.build_int_csr(cap)
    max_nested = _nested_cut_rounds()

    def _cut_arc_indices(side):
        heads = csr.heads
        out = csr.out_by_tail
        return [ai for u in side for ai in out[u] if int(heads[ai]) not in side]

    def _solve(idx_task):
        i, (group_id_l, t, z_val) = idx_task
        sink_idx = ni.get(t)
        if sink_idx is None or sink_idx == src_idx:
            return i, None
        flow_value, src_side, back_side = min_cut_scipy(int_csr, src_idx, sink_idx)
        sides = [src_side]
        if back_cuts and back_side != src_side:
            sides.append(back_side)
        # Nested cuts (Koch & Martin 1998): saturate the arcs of the violated
        # cut just found (capacity -> 1, "as if selected") and re-run max-flow.
        # Capacities are only ever *raised*, so a nested min cut whose value is
        # still below z is guaranteed violated w.r.t. the original y2 values.
        if flow_value < z_val - eps and max_nested > 0:
            cap_mod = cap.copy()
            cur_side = src_side
            for _ in range(max_nested):
                cut_idx = _cut_arc_indices(cur_side)
                if not cut_idx:
                    break
                cap_mod[cut_idx] = np.maximum(cap_mod[cut_idx], 1.0)
                fv, s_side, _b = min_cut_scipy(
                    csr.build_int_csr(cap_mod), src_idx, sink_idx)
                if fv >= z_val - eps:
                    break
                sides.append(s_side)
                cur_side = s_side
        return i, (group_id_l, t, flow_value, z_val, sides)

    indexed = list(enumerate(tasks))
    if threads > 1 and len(indexed) >= _SEP_PARALLEL_MIN_TASKS:
        with ThreadPoolExecutor(max_workers=min(threads, len(indexed))) as ex:
            results = list(ex.map(_solve, indexed))
    else:
        results = [_solve(it) for it in indexed]

    # Re-assemble in task order for a deterministic constraint sequence.
    results.sort(key=lambda r: r[0])
    violated: List[Tuple[int, int, List[Tuple]]] = []
    for _i, payload in results:
        if payload is None:
            continue
        group_id_l, t, flow_value, z_val, sides = payload
        violated.extend(_emit_cuts_for_terminal(
            group_id_k, group_id_l, t, flow_value, z_val, sides, csr, eps,
            node_cuts=None,
        ))
    return violated


def _group_cuts_nx(steiner_problem, group_id_k, tasks, y2_vals, eps, back_cuts):
    """networkx fallback min-cut separation for one source group ``k``.

    The capacity digraph is built once per group (was previously rebuilt per
    ``l``) and reused across all terminals.
    """
    root_k = steiner_problem.roots[group_id_k]
    digraph = nx.DiGraph()
    for (u, v) in steiner_problem.arcs:
        digraph.add_edge(u, v, capacity=y2_vals[(group_id_k, (u, v))] + eps)
    node_cuts = steiner_problem.arcs

    violated: List[Tuple[int, int, List[Tuple]]] = []
    for (group_id_l, t, z_val) in tasks:
        if t == root_k:
            continue
        try:
            residual = nx.algorithms.flow.preflow_push(
                digraph, root_k, t, capacity="capacity", value_only=True
            )
            cut_value = residual.graph["flow_value"]
            sides = [_residual_source_set(residual, root_k, eps)]
            if back_cuts:
                sink_set = _residual_sink_set(residual, t, eps)
                sides.append(set(digraph.nodes) - sink_set)
        except nx.NetworkXError:
            cut_value = 0.0
            sides = [{root_k}]
        violated.extend(_emit_cuts_for_terminal(
            group_id_k, group_id_l, t, cut_value, z_val, sides, None, eps,
            node_cuts=node_cuts,
        ))
    return violated


def find_violated_cuts_from_values(
    steiner_problem: 'SteinerProblem',
    y2_vals: Dict,
    z_vals: Dict,
    eps: float = 1e-6,
    back_cuts: bool = True,
    threads: int = None,
) -> List[Tuple[int, int, List[Tuple]]]:
    """
    Find violated directed cut constraints given pre-extracted variable values.

    This is the core cut-separation routine shared by both the HiGHS and Gurobi
    backends.  It operates on plain value dicts rather than solver-specific
    variable objects, making it solver-agnostic.

    For each pair (group_id_k, group_id_l) with k <= l and for each terminal t
    in terminal_groups[group_id_l], checks whether the directed cut from
    roots[group_id_k] to t is satisfied.  A cut is violated when the minimum
    cut value is strictly less than z[k, l].

    The capacity graph for a fixed source group ``k`` is built **once** and
    reused across every ``l >= k`` and every terminal (it depends only on ``k``).
    Minimum cuts are computed with ``scipy.sparse.csgraph.maximum_flow`` (C, GIL
    releasing) when scipy is available and run concurrently across terminals;
    otherwise a networkx ``preflow_push`` fallback is used.

    Two acceleration techniques from Schmidt, Zey & Margot (2021), Sect. 4.1 are
    applied: *creep flows* (the ``eps`` added to each arc capacity, which biases
    the minimum cut towards cutting few arcs) and, when ``back_cuts`` is set, the
    *back cut* — the second minimum cut on the terminal side, added alongside the
    usual root-side cut.  The scipy path additionally emits *nested cuts*
    (Koch & Martin 1998): for each violated terminal the cut arcs are saturated
    and the max-flow re-run, yielding up to ``STEINERPY_NESTED_CUTS`` further
    violated cuts per separation round (see :func:`_nested_cut_rounds`).

    :param steiner_problem: SteinerProblem-object.
    :param y2_vals: per-group arc values {(group_id, arc): float}.
    :param z_vals: connectivity variable values {(k, l): float}.
    :param eps: numerical tolerance / creep-flow added to each arc capacity.
    :param back_cuts: also emit the terminal-side (back) minimum cut.
    :param threads: separation worker threads (``None`` -> auto).
    :return: list of (group_id_k, group_id_l, cut_arcs) for each violated cut.
    """
    # No terminals → nothing to check
    if len(steiner_problem.terminal_groups[0]) == 0:
        return []

    group_indices = range(len(steiner_problem.terminal_groups))
    nthreads = _sep_thread_count() if threads is None else max(1, int(threads))
    use_scipy = HAS_SCIPY
    csr = get_arc_csr(steiner_problem) if use_scipy else None

    violated_cuts: List[Tuple[int, int, List[Tuple]]] = []
    for group_id_k in group_indices:
        root_k = steiner_problem.roots[group_id_k]
        # Collect the (l, terminal, z) tasks for this source group.
        tasks: List[Tuple[int, object, float]] = []
        for group_id_l in range(group_id_k, len(steiner_problem.terminal_groups)):
            z_val = z_vals[(group_id_k, group_id_l)]
            if z_val < eps:
                continue  # z = 0 -> no connectivity required for this pair
            for t in steiner_problem.terminal_groups[group_id_l]:
                if t == root_k:
                    continue
                tasks.append((group_id_l, t, z_val))
        if not tasks:
            continue

        if use_scipy:
            violated_cuts.extend(_group_cuts_scipy(
                steiner_problem, csr, group_id_k, tasks, y2_vals,
                eps, back_cuts, nthreads,
            ))
        else:
            violated_cuts.extend(_group_cuts_nx(
                steiner_problem, group_id_k, tasks, y2_vals, eps, back_cuts,
            ))

    return violated_cuts


def find_violated_cuts(
    steiner_problem: 'SteinerProblem',
    y2: Dict,
    z: Dict,
    model: hp.HighsModel,
    eps: float = 1e-6,
    back_cuts: bool = True,
) -> List[Tuple[int, int, List[Tuple]]]:
    """
    Find violated directed cut constraints for the current HiGHS LP/MIP solution.

    Reads variable values from the HiGHS model and delegates to
    :func:`find_violated_cuts_from_values`.

    :param steiner_problem: SteinerProblem-object.
    :param y2: per-group arc variables {(group_id, arc): var}.
    :param z: connectivity variables {(k, l): var}.
    :param model: HiGHS model (used to read current variable values).
    :param eps: numerical tolerance / creep-flow added to each arc capacity.
    :param back_cuts: also emit the terminal-side (back) minimum cut.
    :return: list of (group_id_k, group_id_l, cut_arcs) for each violated cut.
    """
    group_indices = range(len(steiner_problem.terminal_groups))
    # One bulk solution read instead of O(|groups| * |arcs|) variableValue calls.
    try:
        col_value = model.getSolution().col_value
        y2_vals = {(group_id, a): col_value[y2[(group_id, a)].index]
                   for group_id in group_indices for a in steiner_problem.arcs}
        z_vals = {key: col_value[var.index] for key, var in z.items()}
    except Exception:  # pragma: no cover - defensive fallback
        y2_vals = {(group_id, a): model.variableValue(y2[(group_id, a)])
                   for group_id in group_indices for a in steiner_problem.arcs}
        z_vals = {key: model.variableValue(var) for key, var in z.items()}
    return find_violated_cuts_from_values(steiner_problem, y2_vals, z_vals, eps, back_cuts)


def build_model(steiner_problem: 'SteinerProblem', time_limit: float = 300, logfile: str = "", threads=None) -> Tuple[hp.HighsModel, hp.HighsVarType, hp.HighsVarType, hp.HighsVarType, hp.HighsVarType]:
    """
    Returns the deterministic directed model without flow variables.
    Connectivity is enforced lazily via directed cut constraints added during
    solving (see :func:`run_model`).

    :param steiner_problem: SteinerProblem-object.
    :param time_limit: time limit in seconds for the HiGHS model. Default is 300 seconds.
    :param logfile: path to logfile.
    :param threads: HiGHS thread count (see :func:`make_model`).
    :return: (model, x, y1, y2, z) – HiGHS model and decision variables.
    """
    # Create the model
    logging.info("Building the model.")

    model = make_model(time_limit, logfile, threads=threads)

    # Start tracking compilation time
    start_time = time.time()

    # Add structural constraints (no flow variables)
    model, x, y1, y2, z = add_directed_constraints(model, steiner_problem)

    # Add degree constraints if max_degree is specified
    if getattr(steiner_problem, 'max_degree', None) is not None:
        add_degree_constraints(model, steiner_problem, x)

    # Add hop constraint if hop_limit is specified (directed problems)
    if getattr(steiner_problem, 'hop_limit', None) is not None:
        add_hop_constraint(model, steiner_problem, y1)

    # End tracking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    logging.info(f"Model built in {compilation_time:.2f} seconds.")

    return model, x, y1, y2, z


def run_model(model: hp.HighsModel, steiner_problem: 'SteinerProblem', x: hp.HighsVarType, y2: Dict, z: Dict, reapply_start=None) -> Tuple[float, float, float, List[Tuple]]:
    """
    Solves the model using an iterative cut-generation (lazy-cut) approach and
    returns the result.

    Instead of adding flow variables and constraints upfront, the solver is run
    repeatedly.  Cuts are first separated on the **LP relaxation** (cheap LP
    re-solves that build up the root cuts, see :func:`_lp_cut_rounds`); then
    integrality is restored and the integer cut loop runs.  After each solve,
    violated directed cut constraints are identified via a minimum-cut
    computation and appended to the model before the next solve.  The process
    terminates when no violated cut is found, guaranteeing that the returned
    solution is feasible.

    :param model: HiGHS model (built by :func:`build_model`).
    :param steiner_problem: SteinerProblem-object.
    :param x: edge selection decision variables.
    :param y2: per-group arc variables {(group_id, arc): var}.
    :param z: connectivity variables {(k, l): var}.
    :param reapply_start: optional zero-argument callable that re-applies a MIP
        warm start; called after the LP phase (which would otherwise clear a
        start set before this function).
    :return: (gap, runtime, objective, selected_edges).
             Note: *runtime* covers only the iterative solve loop and does not
             include the model compilation time reported by :func:`build_model`.
    """
    logging.info("Started running the model with iterative cut generation.")

    objective_expr = sum(
        x[e] * steiner_problem.graph.edges[e][steiner_problem.weight]
        for e in steiner_problem.edges
    )

    # Bound the *total* cut-generation time. Each model.minimize() is already
    # capped at the model's time_limit, but the loop can add many rounds of cuts,
    # so without a global deadline the wall-clock is unbounded (the same flaw fixed
    # in solve_sap_highs). Read the configured limit, then give each re-solve only
    # the time remaining and stop once it is exhausted.
    _tl = model.getOptionValue("time_limit")
    total_tl = float(_tl[1] if isinstance(_tl, tuple) else _tl)

    start_time = time.time()

    # Phase 1 — root cuts on the LP relaxation.  All columns of the base model
    # are integer, so relax them all and restore afterwards.
    lp_rounds = _lp_cut_rounds()
    if lp_rounds > 0:
        num_col = model.getNumCol()
        for col in range(num_col):
            model.changeColIntegrality(col, hp.HighsVarType.kContinuous)
        try:
            for _ in range(lp_rounds):
                remaining = total_tl - (time.time() - start_time)
                if remaining <= 0:
                    break
                model.setOptionValue("time_limit", remaining)
                model.minimize(objective_expr)
                if (model.getModelStatus() != hp.HighsModelStatus.kOptimal
                        or not model.getSolution().value_valid):
                    break
                violated_cuts = find_violated_cuts(steiner_problem, y2, z, model)
                if not violated_cuts:
                    break
                for group_id_k, group_id_l, cut_arcs in violated_cuts:
                    lhs = sum(y2[(group_id_k, a)] for a in cut_arcs) if cut_arcs else 0
                    model.addConstr(lhs >= z[(group_id_k, group_id_l)])
                logging.info(f"LP cut phase: added {len(violated_cuts)} cut(s).")
        finally:
            for col in range(num_col):
                model.changeColIntegrality(col, hp.HighsVarType.kInteger)
        if reapply_start is not None:
            reapply_start()

    # Phase 2 — the integer cut loop.
    converged = False
    while True:
        remaining = total_tl - (time.time() - start_time)
        if remaining <= 0:
            break
        model.setOptionValue("time_limit", remaining)
        model.minimize(objective_expr)

        # If the model has no feasible/valid primal (e.g. an objective cutoff or
        # variable fixing has rendered it infeasible), there is nothing to
        # separate cuts from — reading the empty solution would yield spurious
        # "violations" and loop forever.  Stop the cut loop instead of hanging.
        status = model.getModelStatus()
        if status in (
            hp.HighsModelStatus.kInfeasible,
            hp.HighsModelStatus.kObjectiveBound,
            hp.HighsModelStatus.kUnbounded,
            hp.HighsModelStatus.kUnboundedOrInfeasible,
        ) or not model.getSolution().value_valid:
            runtime = time.time() - start_time
            logging.warning(
                "Cut loop stopped early: model status %s with no valid primal "
                "solution; returning no solution.", status,
            )
            return float("inf"), runtime, float("inf"), []

        violated_cuts = find_violated_cuts(steiner_problem, y2, z, model)

        if not violated_cuts:
            # Optimal only if this re-solve proved optimality. A time-limit-
            # interrupted solve can return a connected but suboptimal incumbent
            # with no violated cuts; treating it as converged would falsely report
            # gap 0 (the false-optimal bug fixed in solve_sap_highs).
            converged = status == hp.HighsModelStatus.kOptimal
            break  # feasible w.r.t. all cut constraints

        # Add each violated cut as a new constraint: sum(y2[k,a] for a in cut) >= z[k,l]
        for group_id_k, group_id_l, cut_arcs in violated_cuts:
            lhs = sum(y2[(group_id_k, a)] for a in cut_arcs) if cut_arcs else 0
            model.addConstr(lhs >= z[(group_id_k, group_id_l)])

        logging.info(f"Added {len(violated_cuts)} violated cut(s), re-solving.")

    runtime = time.time() - start_time
    logging.info(f"Runtime: {runtime:.2f} seconds")

    selected_edges = [e for e in steiner_problem.edges if model.variableValue(x[e]) > 0.5]
    objective = model.getObjectiveValue()
    if converged:
        gap = model.getInfo().mip_gap
    else:
        # Global time limit hit before all connectivity cuts were separated:
        # selected_edges may be disconnected and the relaxation objective is only a
        # lower bound, so do not report a (spurious) ~0 MIP gap.
        gap = float("inf")

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
    incident_edges = _incident_edges(steiner_problem.edges)
    for v in steiner_problem.nodes:
        incident = [x[e] for e in incident_edges.get(v, ())]
        if incident:
            model.addConstr(sum(incident) <= max_degree)


def add_hop_constraint(model: hp.HighsModel, steiner_problem: 'SteinerProblem', y1: Dict) -> None:
    """
    Add a hop-limit constraint to the directed model (Rehfeldt 2021, Ch. 5.8).

    The number of arcs in the arborescence equals ``sum(y1[a])``; bounding it by
    ``hop_limit`` enforces ``|A(S)| <= H``.  Used by the hop-constrained directed
    Steiner tree problem (:class:`steinerpy.objects.HopConstrainedSteinerProblem`).

    :param model: HiGHS model.
    :param steiner_problem: SteinerProblem-object with a ``hop_limit`` attribute.
    :param y1: global arc-selection decision variables.
    """
    arcs = [y1[a] for a in steiner_problem.arcs if a in y1]
    if arcs:
        model.addConstr(sum(arcs) <= steiner_problem.hop_limit)


def build_prize_collecting_model(steiner_problem: 'PrizeCollectingProblem', time_limit: float = 300, logfile: str = "", threads=None) -> Tuple[hp.HighsModel, hp.HighsVarType, hp.HighsVarType, hp.HighsVarType, hp.HighsVarType, Dict[str, hp.HighsVarType], Dict[str, hp.HighsVarType], Dict[str, hp.HighsVarType]]:
    """
    Build prize collecting model by extending the base Steiner model.
    """
    # Start with the base model (reuse existing code)
    model, x, y1, y2, z = build_model(steiner_problem, time_limit, logfile, threads=threads)

    # Prize-collecting needs explicit flow constraints for connectivity enforcement
    model, f = add_flow_constraints(model, steiner_problem, z, y2)
    
    # Add prize collecting specific variables
    group_indices = range(len(steiner_problem.terminal_groups))
    
    # Node selection variables (whether we collect prize from a node)
    node_vars = {node: model.addVariable(0, 1, type=hp.HighsVarType.kInteger, name=f"node[{node}]") 
                 for node in steiner_problem.nodes}
    
    # Terminal penalty variables (penalty for not connecting a terminal)
    penalty_vars = {}
    for group_id in group_indices:
        for terminal in steiner_problem.terminal_groups[group_id]:
            penalty_vars[(group_id, terminal)] = model.addVariable(
                0, 1, type=hp.HighsVarType.kInteger, name=f"penalty[{group_id},{terminal}]"
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
    in_arcs, out_arcs = _arc_adjacency(steiner_problem.arcs)

    # Constraint: Node can only be selected if it's in the tree
    for node in steiner_problem.nodes:
        # If node is selected, it must have at least one incident edge (or be a terminal)
        incident_arcs = [y1[arc] for arc in in_arcs.get(node, ())]
        incident_arcs += [y1[arc] for arc in out_arcs.get(node, ())]
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
                is_connected = sum(y1[arc] for arc in in_arcs.get(terminal, ()))
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


def build_budget_model(steiner_problem: 'BaseSteinerProblem', time_limit: float = 300, logfile: str = "", threads=None) -> Tuple:
    """
    Build a budget-constrained Steiner model.
    Objective: maximize connected terminals (minimize unconnected terminals).
    Constraint: total edge cost <= budget.

    :param steiner_problem: SteinerProblem-object with a ``budget`` attribute.
    :param time_limit: time limit in seconds.
    :param logfile: path to logfile.
    :param threads: HiGHS thread count (see :func:`make_model`).
    :return: HiGHS model and decision variables.
    """
    model = make_model(time_limit, logfile, threads=threads)
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
                0, 1, type=hp.HighsVarType.kInteger, name=f"penalty[{group_id},{terminal}]"
            )
            connection_vars[(group_id, terminal)] = model.addVariable(
                0, 1, type=hp.HighsVarType.kInteger, name=f"conn[{group_id},{terminal}]"
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


# ---------------------------------------------------------------------------
# Gurobi backend (optional – requires gurobipy and a valid Gurobi license)
# ---------------------------------------------------------------------------

def _check_gurobipy() -> None:
    """Raise a clear ImportError when gurobipy is not installed."""
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        raise ImportError(
            "gurobipy is not installed or no valid Gurobi license was found. "
            "Install gurobipy and obtain a license to use solver='gurobi'."
        )


def build_model_gurobi(
    steiner_problem: 'SteinerProblem',
    time_limit: float = 300,
    logfile: str = "",
    threads=None,
) -> Tuple:
    """
    Build the cut-based Steiner model using Gurobi.

    Mirrors :func:`build_model` but creates a ``gurobipy.Model`` instead of a
    HiGHS model.  Connectivity is enforced lazily inside
    :func:`run_model_gurobi` via a branch-and-cut callback, so no flow
    variables are added here.

    :param steiner_problem: SteinerProblem-object.
    :param time_limit: time limit in seconds.
    :param logfile: path to a Gurobi log file (empty string = no file).
    :return: (model, x, y1, y2, z) – Gurobi model and decision variables.
    """
    _check_gurobipy()
    import gurobipy as gp
    from gurobipy import GRB

    logging.info("Building the Gurobi model.")
    start_time = time.time()

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("TimeLimit", time_limit)
    _gthreads = _resolve_threads(threads)
    if _gthreads > 0:
        env.setParam("Threads", _gthreads)
    if logfile:
        env.setParam("LogFile", logfile)
    env.start()
    model = gp.Model(env=env)

    # Sets
    group_indices = range(len(steiner_problem.terminal_groups))
    k_indices = [(k, l) for k in group_indices for l in group_indices if l >= k]
    in_arcs, out_arcs = _arc_adjacency(steiner_problem.arcs)

    # Decision variables
    x = {e: model.addVar(vtype=GRB.BINARY, name=f"x[{e}]")
         for e in steiner_problem.edges}
    y1 = {a: model.addVar(vtype=GRB.BINARY, name=f"y1[{a}]")
          for a in steiner_problem.arcs}
    y2 = {(group_id, a): model.addVar(vtype=GRB.BINARY, name=f"y2[{group_id},{a}]")
          for group_id in group_indices for a in steiner_problem.arcs}
    z = {(k, l): model.addVar(vtype=GRB.BINARY, name=f"z[{k},{l}]")
         for k, l in k_indices}

    model.update()

    # Constraint 1: connection between y2 and y1
    for group_id in group_indices:
        for a in steiner_problem.arcs:
            model.addConstr(y2[group_id, a] <= y1[a])

    # Constraint 2: indegree of each vertex cannot exceed 1
    for v in steiner_problem.nodes:
        incoming = [y1[a] for a in in_arcs.get(v, ())]
        if incoming:
            model.addConstr(gp.quicksum(incoming) <= 1)

    # Constraint 3: connection between y1 and x
    for u, v in steiner_problem.edges:
        if (v, u) in y1:
            model.addConstr(y1[(u, v)] + y1[(v, u)] <= x[(u, v)])
        else:
            model.addConstr(y1[(u, v)] <= x[(u, v)])

    # Constraint 4: enforce terminal group rooted at one root
    for group_id_k in group_indices:
        model.addConstr(
            gp.quicksum(z[group_id_l, group_id_k]
                        for group_id_l in group_indices if group_id_l <= group_id_k) == 1
        )

    # Constraint 5: enforce one root per arborescence
    for group_id_k in group_indices:
        for group_id_l in group_indices:
            if group_id_l > group_id_k:
                model.addConstr(z[group_id_k, group_id_k] >= z[group_id_k, group_id_l])

    # Constraint 6: terminals in T^{1...k-1} cannot attach to root r_k
    for group_id_k in group_indices:
        for t in get_terminal_groups_until_k(steiner_problem.terminal_groups, group_id_k):
            incoming = [y2[group_id_k, a] for a in in_arcs.get(t, ())]
            if incoming:
                model.addConstr(gp.quicksum(incoming) == 0)

    # Constraint 7: indegree at most outdegree for Steiner points
    for v in steiner_problem.steiner_points:
        entering = [y1[a] for a in in_arcs.get(v, ())]
        leaving = [y1[a] for a in out_arcs.get(v, ())]
        if entering:
            out_sum = gp.quicksum(leaving) if leaving else 0
            model.addConstr(gp.quicksum(entering) <= out_sum)

    # Constraint 8: indegree at most outdegree per terminal group
    for group_id_k in group_indices:
        remaining_vertices = (
            set(steiner_problem.nodes)
            - set(terminal_groups_without_root(
                steiner_problem.terminal_groups, steiner_problem.roots, group_id_k
            ))
        )
        for v in remaining_vertices:
            entering = [y2[group_id_k, a] for a in in_arcs.get(v, ())]
            leaving = [y2[group_id_k, a] for a in out_arcs.get(v, ())]
            if entering:
                out_sum = gp.quicksum(leaving) if leaving else 0
                model.addConstr(gp.quicksum(entering) <= out_sum)

    # Constraint 9: connect y2 and z
    for group_id_k in group_indices:
        for group_id_l in group_indices:
            if group_id_l > group_id_k:
                incoming = [y2[group_id_k, a]
                            for a in in_arcs.get(steiner_problem.roots[group_id_l], ())]
                if incoming:
                    model.addConstr(gp.quicksum(incoming) <= z[group_id_k, group_id_l])

    # Optional degree constraints
    if getattr(steiner_problem, 'max_degree', None) is not None:
        max_degree = steiner_problem.max_degree
        incident_edges = _incident_edges(steiner_problem.edges)
        for v in steiner_problem.nodes:
            incident = [x[e] for e in incident_edges.get(v, ())]
            if incident:
                model.addConstr(gp.quicksum(incident) <= max_degree)

    # Optional hop constraint (directed problems, thesis Ch. 5.8)
    if getattr(steiner_problem, 'hop_limit', None) is not None:
        hop_arcs = [y1[a] for a in steiner_problem.arcs if a in y1]
        if hop_arcs:
            model.addConstr(gp.quicksum(hop_arcs) <= steiner_problem.hop_limit)

    model.update()

    compilation_time = time.time() - start_time
    logging.info(f"Gurobi model built in {compilation_time:.2f} seconds.")

    return model, x, y1, y2, z


def run_model_gurobi(
    model,
    steiner_problem: 'SteinerProblem',
    x: Dict,
    y2: Dict,
    z: Dict,
) -> Tuple[float, float, float, List[Tuple]]:
    """
    Solve the Steiner model using Gurobi with a lazy-cut callback.

    The cut-separation logic (directed minimum cuts) is identical to the HiGHS
    iterative approach in :func:`run_model`, but here the cuts are injected as
    *lazy constraints* inside a branch-and-cut callback, which lets Gurobi
    exploit its full branch-and-bound tree rather than re-solving from scratch.

    :param model: Gurobi model (built by :func:`build_model_gurobi`).
    :param steiner_problem: SteinerProblem-object.
    :param x: edge selection decision variables.
    :param y2: per-group arc variables {(group_id, arc): var}.
    :param z: connectivity variables {(k, l): var}.
    :return: (gap, runtime, objective, selected_edges).
    """
    _check_gurobipy()
    import gurobipy as gp
    from gurobipy import GRB

    logging.info("Started running the Gurobi model with lazy cut callback.")

    group_indices = range(len(steiner_problem.terminal_groups))

    # Objective: minimise total edge cost
    obj = gp.quicksum(
        x[e] * steiner_problem.graph.edges[e][steiner_problem.weight]
        for e in steiner_problem.edges
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Enable lazy constraints
    model.Params.LazyConstraints = 1

    # Attach data needed inside the callback
    model._steiner_problem = steiner_problem
    model._y2 = y2
    model._z = z
    model._group_indices = group_indices

    def _cut_callback(cb_model, where):
        # Lazy cuts at integer solutions; user cuts at fractional LP nodes.
        if where == GRB.Callback.MIPSOL:
            get_val = cb_model.cbGetSolution
            add_cut = cb_model.cbLazy
            kind = "lazy"
        elif (where == GRB.Callback.MIPNODE
              and cb_model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
            get_val = cb_model.cbGetNodeRel
            add_cut = cb_model.cbCut
            kind = "user"
        else:
            return

        # Extract the current solution (integer at MIPSOL, fractional at MIPNODE)
        y2_vals = {
            (group_id, a): get_val(cb_model._y2[(group_id, a)])
            for group_id in cb_model._group_indices
            for a in cb_model._steiner_problem.arcs
        }
        z_vals = {key: get_val(var) for key, var in cb_model._z.items()}

        violated = find_violated_cuts_from_values(
            cb_model._steiner_problem, y2_vals, z_vals
        )

        for group_id_k, group_id_l, cut_arcs in violated:
            lhs = (
                gp.quicksum(cb_model._y2[(group_id_k, a)] for a in cut_arcs)
                if cut_arcs else 0
            )
            add_cut(lhs >= cb_model._z[(group_id_k, group_id_l)])

        if violated:
            logging.info(f"Gurobi callback: added {len(violated)} {kind} cut(s).")

    start_time = time.time()
    model.optimize(_cut_callback)
    runtime = time.time() - start_time

    logging.info(f"Gurobi runtime: {runtime:.2f} seconds")

    if model.SolCount == 0:
        # No feasible solution found within the time limit
        return float("inf"), runtime, float("inf"), []

    selected_edges = [e for e in steiner_problem.edges if x[e].X > 0.5]
    gap = model.MIPGap
    objective = model.ObjVal

    return gap, runtime, objective, selected_edges


# ---------------------------------------------------------------------------
# Single-group Steiner-arborescence (SAP) directed-cut solver
# ---------------------------------------------------------------------------
#
# Used by the prize-collecting / MWCSP -> SAP transform path
# (steinerpy.pc_transform). Unlike build_model/run_model, which models an
# *undirected* problem by bidirecting each edge and would double-count a
# bidirected arc's cost (two x-columns per edge), this is a pure arc-based
# directed-cut formulation: one binary x(a) per arc, objective sum c(a) x(a),
# indegree <= 1 per non-root vertex, and connectivity enforced lazily by the same
# minimum-cut separation as the undirected model (find_violated_cuts_from_values
# applied to a single group). It directly implements Formulation 1 (DCut) of both
# Rehfeldt & Koch papers and is exactly the directed setting dual ascent works on,
# so the reduced costs / Steiner cuts / cutoff from dual ascent apply unchanged.


def _sap_indegree_into(view) -> Dict:
    into: Dict = {}
    for a in view.arcs:
        into.setdefault(a[1], []).append(a)
    return into


def solve_sap_highs(view, time_limit: float = 300, logfile: str = "",
                    fixing=None, da_cuts=None, da_ub=None, primal=None,
                    threads=None
                    ) -> Tuple[float, float, float, List[Tuple]]:
    """Solve a single-group SAP by HiGHS + iterative directed-cut generation.

    ``view`` is any object exposing ``graph`` (DiGraph), ``arcs``, ``nodes``,
    ``roots`` (``[r']``), ``terminal_groups`` (``[[r', t1', ...]]``) and
    ``weight`` — e.g. a :class:`steinerpy.objects.DirectedSteinerProblem`.

    Optional dual-ascent acceleration (all best-effort): ``fixing`` (a
    :class:`steinerpy.dual_ascent.FixingResult`; its ``fix_y1_arcs`` are fixed to
    0), ``da_cuts`` (triples from :func:`steinerpy.dual_ascent.steiner_cuts`,
    seeded as initial cuts), ``da_ub`` (objective cutoff), and ``primal`` (a list
    of arcs used as a MIP warm start).

    :returns: ``(gap, runtime, objective, selected_arcs)``.
    """
    model = make_model(time_limit, logfile, threads=threads)
    arcs = list(view.arcs)
    root = view.roots[0]

    xa = {a: model.addVariable(0, 1, name=f"xa[{a}]") for a in arcs}
    for col in range(model.getNumCol()):
        model.changeColIntegrality(col, hp.HighsVarType.kInteger)

    # Indegree <= 1 for every non-root vertex (yields an arborescence).
    for v, in_arcs in _sap_indegree_into(view).items():
        if v == root:
            continue
        model.addConstr(sum(xa[a] for a in in_arcs) <= 1)

    # Dual-ascent reduced-cost fixing: arcs in no optimal solution -> 0.
    if fixing is not None:
        for a in getattr(fixing, "fix_y1_arcs", ()):
            if a in xa:
                model.changeColBounds(xa[a].index, 0, 0)

    # Seed the dual-ascent Steiner cuts (sum of entering arcs >= 1).
    if da_cuts:
        for (_k, _l, cut_arcs) in da_cuts:
            terms = [xa[a] for a in cut_arcs if a in xa]
            if terms:
                model.addConstr(sum(terms) >= 1)

    # NOTE: deliberately do NOT pass da_ub to HiGHS as `objective_bound`. Unlike
    # Gurobi's `Params.Cutoff` (a true pruning cutoff), HiGHS treats objective_bound
    # as a termination target inside the cut loop: a loose dual-ascent UB makes a
    # re-solve stop kOptimal at a feasible-but-suboptimal incumbent, which the loop
    # then reports as "proven optimal" (false-optimal bug observed on PCSPG P400).
    # da_ub is still used below only to report an honest gap on a non-proven solve.

    obj = sum(xa[a] * view.graph.edges[a][view.weight] for a in arcs)

    start = time.time()

    # Phase 1 — root cuts on the LP relaxation (see _lp_cut_rounds).
    lp_rounds = _lp_cut_rounds()
    if lp_rounds > 0:
        num_col = model.getNumCol()
        for col in range(num_col):
            model.changeColIntegrality(col, hp.HighsVarType.kContinuous)
        try:
            for _ in range(lp_rounds):
                remaining = time_limit - (time.time() - start)
                if remaining <= 0:
                    break
                model.setOptionValue("time_limit", remaining)
                model.minimize(obj)
                if (model.getModelStatus() != hp.HighsModelStatus.kOptimal
                        or not model.getSolution().value_valid):
                    break
                col_value = model.getSolution().col_value
                xa_vals = {(0, a): col_value[xa[a].index] for a in arcs}
                violated = find_violated_cuts_from_values(view, xa_vals, {(0, 0): 1.0})
                if not violated:
                    break
                for (_k, _l, cut_arcs) in violated:
                    if cut_arcs:
                        model.addConstr(sum(xa[a] for a in cut_arcs) >= 1)
        finally:
            for col in range(num_col):
                model.changeColIntegrality(col, hp.HighsVarType.kInteger)

    # MIP warm start from the dual-ascent primal (applied after the LP phase,
    # which would otherwise clear it).
    if primal:
        try:
            import numpy as np
            sel = set(primal)
            idx = [xa[a].index for a in arcs]
            val = [1.0 if a in sel else 0.0 for a in arcs]
            model.setSolution(len(idx), np.array(idx, dtype=np.int32),
                              np.array(val, dtype=np.float64))
        except Exception:
            pass

    # Phase 2 — the integer cut loop.
    converged = False
    _STOP = (
        hp.HighsModelStatus.kInfeasible,
        hp.HighsModelStatus.kObjectiveBound,
        hp.HighsModelStatus.kUnbounded,
        hp.HighsModelStatus.kUnboundedOrInfeasible,
    )
    while True:
        # Bound the *total* cut-generation time, not just each individual re-solve.
        # The loop can add many rounds of Steiner cuts, and each model.minimize()
        # otherwise gets the full time_limit, so the wall-clock was unbounded — it
        # hung on hard instances (e.g. PCSPG P400, where dual ascent fixes nothing
        # and many cut rounds are needed). Give each solve only the time remaining
        # and stop once it is exhausted.
        remaining = time_limit - (time.time() - start)
        if remaining <= 0:
            break
        model.setOptionValue("time_limit", remaining)
        model.minimize(obj)
        status = model.getModelStatus()
        if status in _STOP or not model.getSolution().value_valid:
            break
        col_value = model.getSolution().col_value
        xa_vals = {(0, a): col_value[xa[a].index] for a in arcs}
        violated = find_violated_cuts_from_values(view, xa_vals, {(0, 0): 1.0})
        if not violated:
            # Optimal only if this re-solve actually *proved* optimality. A solve
            # interrupted by the (shrinking) time limit can return a connected but
            # suboptimal incumbent with no violated cuts; treating that as converged
            # would stamp a non-optimal tree "proven optimal" (observed on P400).
            converged = status == hp.HighsModelStatus.kOptimal
            break
        for (_k, _l, cut_arcs) in violated:
            if cut_arcs:
                model.addConstr(sum(xa[a] for a in cut_arcs) >= 1)
    runtime = time.time() - start

    selected = [a for a in arcs if model.variableValue(xa[a]) > 0.5]
    objective = model.getObjectiveValue()
    if converged:
        gap = model.getInfo().mip_gap
    else:
        # Time limit hit before all connectivity cuts were separated: the model is
        # a relaxation whose optimum is only a lower bound, so its MIP gap would be
        # a spurious ~0. Report an honest gap against the dual-ascent feasible upper
        # bound instead (the caller maps the best valid component of `selected`).
        if da_ub is not None and math.isfinite(da_ub) and da_ub > 0:
            gap = max(0.0, (da_ub - objective) / max(1.0, abs(da_ub)))
        else:
            gap = float("inf")
    return gap, runtime, objective, selected


def solve_sap_gurobi(view, time_limit: float = 300, logfile: str = "",
                     fixing=None, da_cuts=None, da_ub=None, primal=None,
                     threads=None
                     ) -> Tuple[float, float, float, List[Tuple]]:
    """Gurobi branch-and-cut counterpart of :func:`solve_sap_highs`.

    Connectivity is separated lazily inside a callback, mirroring
    :func:`run_model_gurobi`.
    """
    _check_gurobipy()
    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("TimeLimit", time_limit)
    _gthreads = _resolve_threads(threads)
    if _gthreads > 0:
        env.setParam("Threads", _gthreads)
    if logfile:
        env.setParam("LogFile", logfile)
    env.start()
    model = gp.Model(env=env)

    arcs = list(view.arcs)
    root = view.roots[0]
    xa = {a: model.addVar(vtype=GRB.BINARY, name=f"xa[{a}]") for a in arcs}
    model.update()

    for v, in_arcs in _sap_indegree_into(view).items():
        if v == root:
            continue
        model.addConstr(gp.quicksum(xa[a] for a in in_arcs) <= 1)

    if fixing is not None:
        for a in getattr(fixing, "fix_y1_arcs", ()):
            if a in xa:
                xa[a].UB = 0
    if da_cuts:
        for (_k, _l, cut_arcs) in da_cuts:
            terms = [xa[a] for a in cut_arcs if a in xa]
            if terms:
                model.addConstr(gp.quicksum(terms) >= 1)
    if primal:
        sel = set(primal)
        for a in arcs:
            xa[a].Start = 1.0 if a in sel else 0.0
    if da_ub is not None:
        model.Params.Cutoff = float(da_ub)
    model.update()

    model.setObjective(
        gp.quicksum(xa[a] * view.graph.edges[a][view.weight] for a in arcs),
        GRB.MINIMIZE,
    )
    model.Params.LazyConstraints = 1
    model._view = view
    model._xa = xa

    def _cb(cb_model, where):
        # Lazy cuts at integer solutions; user cuts at fractional LP nodes.
        if where == GRB.Callback.MIPSOL:
            get_val = cb_model.cbGetSolution
            add_cut = cb_model.cbLazy
        elif (where == GRB.Callback.MIPNODE
              and cb_model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
            get_val = cb_model.cbGetNodeRel
            add_cut = cb_model.cbCut
        else:
            return
        xa_vals = {(0, a): get_val(cb_model._xa[a])
                   for a in cb_model._view.arcs}
        violated = find_violated_cuts_from_values(cb_model._view, xa_vals, {(0, 0): 1.0})
        for (_k, _l, cut_arcs) in violated:
            if cut_arcs:
                add_cut(gp.quicksum(cb_model._xa[a] for a in cut_arcs) >= 1)

    start = time.time()
    model.optimize(_cb)
    runtime = time.time() - start

    if model.SolCount == 0:
        return float("inf"), runtime, float("inf"), []
    selected = [a for a in arcs if xa[a].X > 0.5]
    return model.MIPGap, runtime, model.ObjVal, selected


# ---------------------------------------------------------------------------
# Max-weight connected subgraph with a vertex-cost budget (MWCSPB) — Ch. 5.6
# ---------------------------------------------------------------------------
#
# Rooted directed model: an arborescence from the (always-selected) root, with one
# binary node[v] per vertex tied to its single entering arc, optional per-positive-node
# connectivity via the existing optional-flow constraints, a vertex-cost budget, and a
# node-weight-maximising objective.  Connectivity is modelled with flow (added upfront)
# rather than lazy cuts, so the identical formulation solves directly on both HiGHS and
# Gurobi.  Mirrors the structure of build_budget_model.


def _undirected_from_arcs(used_arcs: List[Tuple]) -> List[Tuple]:
    """Collapse a list of directed arcs to deduplicated undirected edges."""
    seen: Set = set()
    edges: List[Tuple] = []
    for (u, v) in used_arcs:
        key = frozenset((u, v))
        if key not in seen:
            seen.add(key)
            edges.append((u, v))
    return edges


def build_mwcsb_model(steiner_problem, time_limit: float = 300, logfile: str = "",
                      threads=None) -> Tuple:
    """Build the HiGHS MWCSPB model (see module note above).

    :param steiner_problem: a :class:`BudgetedMaxWeightConnectedSubgraph` exposing
        ``node_costs``, ``node_budget`` and ``_mwcs_node_weights``.
    :return: ``(model, x, y1, y2, z, node_vars)``.
    """
    model, x, y1, y2, z = build_model(steiner_problem, time_limit, logfile, threads=threads)
    root = steiner_problem.roots[0]

    node_vars = {
        v: model.addVariable(0, 1, type=hp.HighsVarType.kInteger, name=f"node[{v}]")
        for v in steiner_problem.nodes
    }
    # The root is always part of the subgraph; every other selected node has exactly
    # one entering arc, so node[v] equals its indegree in the arborescence.
    in_arcs, _out_arcs = _arc_adjacency(steiner_problem.arcs)
    model.addConstr(node_vars[root] == 1)
    for v in steiner_problem.nodes:
        if v == root:
            continue
        entering = [y1[a] for a in in_arcs.get(v, ())]
        model.addConstr(node_vars[v] == (sum(entering) if entering else 0))

    # Optional reachability for each positive node, scaled by its node variable.
    connection_vars = {
        (0, t): node_vars[t]
        for t in steiner_problem.terminal_groups[0] if t != root
    }
    model, _f = add_optional_flow_constraints(model, steiner_problem, y2, connection_vars)

    # Vertex-cost budget over the whole chosen subgraph.
    node_costs = steiner_problem.node_costs
    model.addConstr(
        sum(node_costs.get(v, 0) * node_vars[v] for v in steiner_problem.nodes)
        <= steiner_problem.node_budget
    )

    return model, x, y1, y2, z, node_vars


def run_mwcsb_model(model, steiner_problem, y1: Dict, node_vars: Dict) -> Tuple:
    """Solve the HiGHS MWCSPB model and extract the connected subgraph.

    :return: ``(gap, runtime, mwcs_weight, selected_edges, selected_nodes)`` where
        ``mwcs_weight`` is the sum of node weights over the chosen subgraph.
    """
    nw = steiner_problem._mwcs_node_weights
    # Maximise total node weight == minimise its negation.
    objective_expr = sum(-nw.get(v, 0.0) * node_vars[v] for v in steiner_problem.nodes)
    model.minimize(objective_expr)

    status = model.getModelStatus()
    if status in (
        hp.HighsModelStatus.kInfeasible,
        hp.HighsModelStatus.kUnbounded,
        hp.HighsModelStatus.kUnboundedOrInfeasible,
    ) or not model.getSolution().value_valid:
        return float("inf"), model.getRunTime(), float("-inf"), [], []

    selected_nodes = [v for v in steiner_problem.nodes if model.variableValue(node_vars[v]) > 0.5]
    used_arcs = [a for a in steiner_problem.arcs if model.variableValue(y1[a]) > 0.5]
    selected_edges = _undirected_from_arcs(used_arcs)
    mwcs_weight = sum(nw.get(v, 0.0) for v in selected_nodes)

    return model.getInfo().mip_gap, model.getRunTime(), mwcs_weight, selected_edges, selected_nodes


def build_mwcsb_model_gurobi(steiner_problem, time_limit: float = 300, logfile: str = "",
                             threads=None) -> Tuple:
    """Gurobi counterpart of :func:`build_mwcsb_model`."""
    _check_gurobipy()
    import gurobipy as gp

    model, x, y1, y2, z = build_model_gurobi(steiner_problem, time_limit, logfile, threads=threads)
    from gurobipy import GRB
    root = steiner_problem.roots[0]
    non_root = [t for t in steiner_problem.terminal_groups[0] if t != root]

    node_vars = {v: model.addVar(vtype=GRB.BINARY, name=f"node[{v}]")
                 for v in steiner_problem.nodes}
    model.update()

    in_arcs, out_arcs = _arc_adjacency(steiner_problem.arcs)

    model.addConstr(node_vars[root] == 1)
    for v in steiner_problem.nodes:
        if v == root:
            continue
        entering = [y1[a] for a in in_arcs.get(v, ())]
        model.addConstr(node_vars[v] == (gp.quicksum(entering) if entering else 0))

    # Optional flow for each positive node, scaled by its node variable.
    # Continuous in [0, 1]: with integral y2/node_vars each t-block is a unit
    # flow with integral capacities and f never enters the objective.
    f = {(t, a): model.addVar(lb=0.0, ub=1.0, name=f"f[{t},{a}]")
         for t in non_root for a in steiner_problem.arcs}
    model.update()
    for v in steiner_problem.nodes:
        arcs_out = out_arcs.get(v, ())
        arcs_in = in_arcs.get(v, ())
        for t in non_root:
            if v == root:
                demand = node_vars[t]
            elif v == t:
                demand = -node_vars[t]
            else:
                if not arcs_out and not arcs_in:
                    continue
                demand = 0
            model.addConstr(
                gp.quicksum(f[t, a] for a in arcs_out)
                - gp.quicksum(f[t, a] for a in arcs_in) == demand
            )
    for t in non_root:
        for a in steiner_problem.arcs:
            model.addConstr(f[t, a] <= y2[(0, a)])
        terminal_out = out_arcs.get(t, ())
        if terminal_out:
            model.addConstr(gp.quicksum(f[t, a] for a in terminal_out) == 0)

    node_costs = steiner_problem.node_costs
    model.addConstr(
        gp.quicksum(node_costs.get(v, 0) * node_vars[v] for v in steiner_problem.nodes)
        <= steiner_problem.node_budget
    )
    model.update()

    return model, x, y1, y2, z, node_vars


def run_mwcsb_model_gurobi(model, steiner_problem, y1: Dict, node_vars: Dict) -> Tuple:
    """Gurobi counterpart of :func:`run_mwcsb_model`."""
    _check_gurobipy()
    import gurobipy as gp
    from gurobipy import GRB

    nw = steiner_problem._mwcs_node_weights
    model.setObjective(
        gp.quicksum(-nw.get(v, 0.0) * node_vars[v] for v in steiner_problem.nodes),
        GRB.MINIMIZE,
    )
    start = time.time()
    model.optimize()
    runtime = time.time() - start

    if model.SolCount == 0:
        return float("inf"), runtime, float("-inf"), [], []

    selected_nodes = [v for v in steiner_problem.nodes if node_vars[v].X > 0.5]
    used_arcs = [a for a in steiner_problem.arcs if y1[a].X > 0.5]
    selected_edges = _undirected_from_arcs(used_arcs)
    mwcs_weight = sum(nw.get(v, 0.0) for v in selected_nodes)

    return model.MIPGap, runtime, mwcs_weight, selected_edges, selected_nodes