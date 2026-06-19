"""Tests for the dual-ascent accelerator (steinerpy.dual_ascent).

Covers the bound invariants (LB <= opt <= UB), the forest and directed cases,
the reduced-cost fixing safety invariant (never fix an optimal arc/edge), and
the end-to-end guarantee that the opt-in accelerator returns the SAME optimum
as the baseline solver.
"""

import math

import networkx as nx
import pytest

from steinerpy import SteinerProblem, DirectedSteinerProblem
from steinerpy.dual_ascent import dual_ascent, reduced_cost_fixing


def _chain(n, weight=1):
    g = nx.Graph()
    for i in range(n - 1):
        g.add_edge(i, i + 1, weight=weight)
    return g


def _notebook_tree():
    g = nx.Graph()
    for (a, b), w in zip([("A", "C"), ("A", "D"), ("B", "C"), ("C", "D")], [1, 10, 1, 1]):
        g.add_edge(a, b, weight=w)
    return g


def _norm(edges):
    return {tuple(sorted(e)) for e in edges}


# ---------------------------------------------------------------------------
# Bound invariants
# ---------------------------------------------------------------------------

def test_da_path_lb_equals_ub_equals_opt():
    g = _chain(4)  # 0-1-2-3, opt = 3
    p = SteinerProblem(g, [[0, 3]], preprocess=False)
    da = dual_ascent(p)
    assert da.feasible
    assert da.lower_bound == pytest.approx(3.0)
    assert da.upper_bound == pytest.approx(3.0)
    assert _norm(da.primal_edges) == {(0, 1), (1, 2), (2, 3)}


def test_da_tree_brackets_optimum():
    g = _notebook_tree()  # opt = 3 (AC, BC, CD)
    p = SteinerProblem(g, [["A", "B", "D"]], preprocess=False)
    da = dual_ascent(p)
    assert da.lower_bound <= 3.0 + 1e-9
    assert da.upper_bound >= 3.0 - 1e-9
    assert da.feasible


def test_da_brackets_optimum_on_random_graphs():
    import random
    rng = random.Random(12345)
    for trial in range(8):
        n = rng.randint(5, 9)
        g = nx.gnp_random_graph(n, 0.5, seed=rng.randint(0, 10 ** 6))
        if not nx.is_connected(g):
            g = nx.minimum_spanning_tree(nx.complete_graph(n))
        for a, b in g.edges:
            g.edges[a, b]["weight"] = rng.randint(1, 9)
        terms = rng.sample(list(g.nodes), 3)
        p = SteinerProblem(g, [terms], preprocess=False)
        opt = p.get_solution(time_limit=30).objective
        da = dual_ascent(p)
        assert da.lower_bound <= opt + 1e-6, f"LB {da.lower_bound} > opt {opt}"
        assert da.upper_bound >= opt - 1e-6, f"UB {da.upper_bound} < opt {opt}"


# ---------------------------------------------------------------------------
# Forest
# ---------------------------------------------------------------------------

def test_da_forest_valid_bounds():
    g = _chain(5, weight=1)  # A path 0-1-2-3-4
    p = SteinerProblem(g, [[0, 2], [2, 4]], preprocess=False)
    opt = p.get_solution(time_limit=30).objective
    da = dual_ascent(p)
    assert math.isfinite(da.lower_bound)
    assert da.lower_bound <= opt + 1e-6   # KEY forest validity invariant
    assert da.upper_bound >= opt - 1e-6
    assert da.feasible


# ---------------------------------------------------------------------------
# Directed
# ---------------------------------------------------------------------------

def test_da_directed_no_reverse_arcs():
    dg = nx.DiGraph()
    dg.add_edge("A", "B", weight=1)
    dg.add_edge("B", "C", weight=1)
    dg.add_edge("A", "C", weight=10)  # direct but expensive, opt = 2
    p = DirectedSteinerProblem(dg, root="A", terminals=["B", "C"])
    da = dual_ascent(p)
    assert da.is_directed
    assert da.lower_bound <= 2.0 + 1e-9
    assert da.upper_bound >= 2.0 - 1e-9
    # primal uses only real directed arcs
    for (u, v) in da.primal_edges:
        assert dg.has_edge(u, v)


# ---------------------------------------------------------------------------
# Reduced-cost fixing safety
# ---------------------------------------------------------------------------

def test_fixing_never_removes_optimal_edge():
    g = _notebook_tree()
    p = SteinerProblem(g, [["A", "B", "D"]], preprocess=False)
    da = dual_ascent(p)
    fixing = reduced_cost_fixing(p, da)
    optimum = {frozenset(e) for e in [("A", "C"), ("B", "C"), ("C", "D")]}
    for e in fixing.fix_x_edges:
        assert frozenset(e) not in optimum
    # the expensive A-D edge is the one that should be eliminated
    assert ("A", "D") in fixing.fix_x_edges or ("D", "A") in fixing.fix_x_edges


def test_fixing_directed_keeps_optimal_arcs():
    dg = nx.DiGraph()
    dg.add_edge("A", "B", weight=1)
    dg.add_edge("B", "C", weight=1)
    dg.add_edge("A", "C", weight=10)
    p = DirectedSteinerProblem(dg, root="A", terminals=["B", "C"])
    da = dual_ascent(p)
    fixing = reduced_cost_fixing(p, da)
    assert not (fixing.fix_y1_arcs & {("A", "B"), ("B", "C")})
    assert ("A", "C") in fixing.fix_y1_arcs


def test_undirected_edge_fixed_only_when_both_directions_fixable():
    # In the notebook tree, the A-D edge is fixed (both directions unusable);
    # confirm fix_x only contains edges whose BOTH arcs are in fix_y1.
    g = _notebook_tree()
    p = SteinerProblem(g, [["A", "B", "D"]], preprocess=False)
    da = dual_ascent(p)
    fixing = reduced_cost_fixing(p, da)
    for (u, v) in fixing.fix_x_edges:
        assert (u, v) in fixing.fix_y1_arcs and (v, u) in fixing.fix_y1_arcs


# ---------------------------------------------------------------------------
# End-to-end: accelerator returns the SAME optimum as the baseline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", [
    lambda da: SteinerProblem(_chain(4), [[0, 3]], preprocess=True, dual_ascent=da),
    lambda da: SteinerProblem(_notebook_tree(), [["A", "B", "D"]], preprocess=False, dual_ascent=da),
    lambda da: SteinerProblem(_chain(5), [[0, 2], [2, 4]], preprocess=False, dual_ascent=da),
])
def test_da_matches_baseline_objective(build):
    base = build(False).get_solution(time_limit=30)
    accel = build(True).get_solution(time_limit=30, dual_ascent=True)
    assert accel.objective == pytest.approx(base.objective, abs=1e-6)


def test_da_directed_matches_baseline_objective():
    def build(da):
        dg = nx.DiGraph()
        dg.add_edge("A", "B", weight=1)
        dg.add_edge("B", "C", weight=1)
        dg.add_edge("A", "C", weight=10)
        return DirectedSteinerProblem(dg, root="A", terminals=["B", "C"], dual_ascent=da)
    base = build(False).get_solution(time_limit=30)
    accel = build(True).get_solution(time_limit=30, dual_ascent=True)
    assert accel.objective == pytest.approx(base.objective, abs=1e-6)


def test_da_early_exit_skips_ilp(monkeypatch):
    """When LB == UB the ILP must not be built."""
    import steinerpy.objects as objmod
    called = {"build": False}
    real_build = objmod.build_model

    def spy_build(*args, **kwargs):
        called["build"] = True
        return real_build(*args, **kwargs)

    monkeypatch.setattr(objmod, "build_model", spy_build)
    p = SteinerProblem(_chain(4), [[0, 3]], preprocess=False, dual_ascent=True)
    sol = p.get_solution(time_limit=30)
    assert sol.objective == pytest.approx(3.0)
    assert called["build"] is False  # solved by dual ascent alone


def test_da_off_by_default_does_not_import_module():
    """With the flag off, behaviour is unchanged and dual_ascent is not used."""
    import steinerpy.objects as objmod
    sentinel = {"called": False}
    import steinerpy.dual_ascent as damod
    real = damod.dual_ascent

    def spy(*a, **k):
        sentinel["called"] = True
        return real(*a, **k)

    objmod_dual = getattr(damod, "dual_ascent")
    damod.dual_ascent = spy
    try:
        SteinerProblem(_chain(4), [[0, 3]], preprocess=False).get_solution(time_limit=30)
        assert sentinel["called"] is False
    finally:
        damod.dual_ascent = objmod_dual


def test_da_disconnected_terminal_declines_gracefully():
    """A terminal unreachable from the root -> DA reports infinite/infeasible and
    produces no fixing, so the caller safely falls through to the normal solver
    path (we don't exercise the ILP itself, which is infeasible here)."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1)
    g.add_node(2)  # isolated, will be a terminal
    p = SteinerProblem(g, [[0, 2]], preprocess=False, dual_ascent=True)
    da = dual_ascent(p)
    assert math.isinf(da.lower_bound) or not da.feasible
    fixing = reduced_cost_fixing(p, da)
    assert fixing.total() == 0  # never fix anything on an infeasible DA result
