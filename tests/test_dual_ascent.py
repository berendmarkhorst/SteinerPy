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
from steinerpy.dual_ascent import dual_ascent, reduced_cost_fixing, steiner_cuts


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


# ---------------------------------------------------------------------------
# Cut initialization: retained Steiner cuts + seeding
# ---------------------------------------------------------------------------

def _arc_separates(arcs_removed, root, terminal, all_arcs):
    """True if removing arcs_removed disconnects root from terminal in the digraph."""
    h = nx.DiGraph()
    h.add_nodes_from({n for a in all_arcs for n in a} | {root, terminal})
    removed = set(arcs_removed)
    h.add_edges_from(a for a in all_arcs if a not in removed)
    return not nx.has_path(h, root, terminal)


@pytest.mark.parametrize("build", [
    lambda: SteinerProblem(_notebook_tree(), [["A", "B", "D"]], preprocess=False),
    lambda: SteinerProblem(_chain(5), [[0, 2], [2, 4]], preprocess=False),
])
def test_steiner_cuts_are_valid(build):
    p = build()
    da = dual_ascent(p)

    # Each retained W is a genuine Steiner cut: root outside, a terminal inside.
    for ga in da.groups:
        for W in ga.cuts:
            assert ga.root not in W
            assert set(ga.terminals) & W

    triples = steiner_cuts(da)
    assert triples, "expected at least one seeded cut on a non-trivial instance"
    arc_set = set(da.arcs)
    for (k, l, cut_arcs) in triples:
        assert k == l
        assert cut_arcs
        assert all(a in arc_set for a in cut_arcs)
        # The cut actually separates the group root from some group terminal.
        root = da.groups[k].root
        assert any(
            _arc_separates(cut_arcs, root, t, da.arcs)
            for t in da.groups[k].terminals if t != root
        )


def test_steiner_cuts_directed():
    dg = nx.DiGraph()
    dg.add_edge("A", "B", weight=1)
    dg.add_edge("B", "C", weight=1)
    dg.add_edge("A", "C", weight=10)
    p = DirectedSteinerProblem(dg, root="A", terminals=["B", "C"])
    da = dual_ascent(p)
    triples = steiner_cuts(da)
    assert triples
    for (k, l, cut_arcs) in triples:
        assert k == 0 and l == 0
        for (u, v) in cut_arcs:
            assert dg.has_edge(u, v)  # directed: only real arcs, no reverse


def test_seeding_preserves_optimum_and_fires():
    """Seeding (and the bundled cutoff) must not change the optimum, and the
    cuts must actually be produced on a non-trivial instance."""
    p_cuts = SteinerProblem(_notebook_tree(), [["A", "B", "D"]], preprocess=False)
    assert len(steiner_cuts(dual_ascent(p_cuts))) > 0

    base = SteinerProblem(_notebook_tree(), [["A", "B", "D"]], preprocess=False)
    accel = SteinerProblem(_notebook_tree(), [["A", "B", "D"]], preprocess=False,
                           dual_ascent=True)
    assert accel.get_solution(time_limit=30, dual_ascent=True).objective == \
        pytest.approx(base.get_solution(time_limit=30).objective, abs=1e-6)


def test_seeding_preserves_optimum_on_random_graphs():
    """Exercise the in-solve seed + cutoff path on instances that do not
    early-exit, and confirm the objective is unchanged vs the baseline."""
    import random
    rng = random.Random(98765)
    for _ in range(6):
        n = rng.randint(6, 10)
        g = nx.gnp_random_graph(n, 0.5, seed=rng.randint(0, 10 ** 6))
        if not nx.is_connected(g):
            g = nx.minimum_spanning_tree(nx.complete_graph(n))
        for a, b in g.edges:
            g.edges[a, b]["weight"] = rng.randint(1, 9)
        terms = rng.sample(list(g.nodes), 3)
        base = SteinerProblem(g, [terms], preprocess=False).get_solution(time_limit=30)
        accel = SteinerProblem(g, [terms], preprocess=False, dual_ascent=True) \
            .get_solution(time_limit=30, dual_ascent=True)
        assert accel.objective == pytest.approx(base.objective, abs=1e-6)


def test_seeding_reduces_cut_loop_rounds(monkeypatch):
    """Seeding the dual-ascent cuts must not increase the number of cut-loop
    rounds (one round == one find_violated_cuts call), and there must be a
    non-trivial number of rounds to reduce."""
    import steinerpy.mathematical_model as mm

    # A graph whose baseline cut loop needs several separation rounds.
    rng = __import__("random").Random(2024)
    g = nx.gnp_random_graph(11, 0.45, seed=7)
    if not nx.is_connected(g):
        g = nx.minimum_spanning_tree(nx.complete_graph(11))
    for a, b in g.edges:
        g.edges[a, b]["weight"] = rng.randint(1, 9)
    terms = [0, 5, 9, 3]

    real = mm.find_violated_cuts
    counter = {"n": 0}

    def spy(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(mm, "find_violated_cuts", spy)

    counter["n"] = 0
    SteinerProblem(g, [terms], preprocess=False).get_solution(time_limit=60)
    rounds_base = counter["n"]

    counter["n"] = 0
    SteinerProblem(g, [terms], preprocess=False, dual_ascent=True) \
        .get_solution(time_limit=60, dual_ascent=True)
    rounds_accel = counter["n"]

    assert rounds_base >= 1
    assert rounds_accel <= rounds_base


# ---------------------------------------------------------------------------
# Multi-start primal (tighter upper bound -> more early-exits / fixing)
# ---------------------------------------------------------------------------

def test_multistart_primal_never_worse_than_single_root():
    """The multi-start upper bound must be <= the single-(model-)root primal and
    must still bracket the optimum (LB <= opt <= UB)."""
    from steinerpy.dual_ascent import (
        _arc_costs, _primal_for_group, _edges_cost, _node_universe, _ascent, EPS,
    )
    import random
    rng = random.Random(31415)
    for _ in range(12):
        n = rng.randint(6, 12)
        g = nx.gnp_random_graph(n, 0.45, seed=rng.randint(0, 10 ** 6))
        if not nx.is_connected(g):
            g = nx.minimum_spanning_tree(nx.complete_graph(n))
        for a, b in g.edges:
            g.edges[a, b]["weight"] = rng.randint(1, 12)
        terms = rng.sample(list(g.nodes), 3)
        p = SteinerProblem(g, [terms], preprocess=False)
        opt = p.get_solution(time_limit=30).objective

        # Single-root primal (model root only).
        arcs = list(p.arcs)
        root = p.roots[0]
        costs = _arc_costs(g, arcs, p.weight)
        ga = _ascent(list(arcs), root, terms, costs, _node_universe(arcs, terms, root))
        sat = {a for a in arcs if ga.reduced_costs[a] <= EPS}
        pe, ok = _primal_for_group(g, arcs, root, terms, sat, p.weight, False)
        single_ub = _edges_cost(g, pe, p.weight) if ok else math.inf

        da = dual_ascent(p)
        assert da.upper_bound <= single_ub + 1e-9       # multi-start never worse
        assert da.lower_bound <= opt + 1e-6 <= da.upper_bound + 1e-6


# ---------------------------------------------------------------------------
# Step 5: dual-ascent bound reduction (da_reduce)
# ---------------------------------------------------------------------------

def test_da_reduce_removes_provably_bad_edge_and_shrinks():
    # Notebook tree: A-D (weight 10) is in no optimal solution and survives the
    # degree reductions (all degree-1/2 nodes are terminals), so da_reduce removes it.
    plain = SteinerProblem(_notebook_tree(), [["A", "B", "D"]], preprocess=True)
    reduced = SteinerProblem(_notebook_tree(), [["A", "B", "D"]], preprocess=True, da_reduce=True)
    assert reduced.graph.number_of_edges() < plain.graph.number_of_edges()
    assert not reduced.graph.has_edge("A", "D")


@pytest.mark.parametrize("build", [
    lambda red: SteinerProblem(_notebook_tree(), [["A", "B", "D"]], preprocess=True, da_reduce=red),
    lambda red: SteinerProblem(_chain(6), [[0, 5]], preprocess=True, da_reduce=red),
])
def test_da_reduce_preserves_optimum(build):
    assert build(True).get_solution(time_limit=30).objective == \
        pytest.approx(build(False).get_solution(time_limit=30).objective, abs=1e-6)


def test_da_reduce_preserves_optimum_random():
    import random
    rng = random.Random(246)
    for _ in range(6):
        n = rng.randint(7, 12)
        g = nx.gnp_random_graph(n, 0.4, seed=rng.randint(0, 10 ** 6))
        if not nx.is_connected(g):
            g = nx.minimum_spanning_tree(nx.complete_graph(n))
        for a, b in g.edges:
            g.edges[a, b]["weight"] = rng.randint(1, 15)
        terms = rng.sample(list(g.nodes), 3)
        base = SteinerProblem(g, [terms], preprocess=True).get_solution(time_limit=30)
        red = SteinerProblem(g, [terms], preprocess=True, da_reduce=True).get_solution(time_limit=30)
        assert red.objective == pytest.approx(base.objective, abs=1e-6)
        # Back-map: every reported original edge exists in the ORIGINAL graph.
        for (u, v) in red.original_selected_edges:
            assert g.has_edge(u, v)


def test_da_reduce_ignored_for_directed():
    dg = nx.DiGraph()
    dg.add_edge("A", "B", weight=1)
    dg.add_edge("B", "C", weight=1)
    dg.add_edge("A", "C", weight=10)
    # Directed problems force preprocess=False, so da_reduce is a harmless no-op.
    p = DirectedSteinerProblem(dg, root="A", terminals=["B", "C"], da_reduce=True)
    assert p.get_solution(time_limit=30).objective == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# A2: faster _ascent must be exactly equivalent to the naive implementation
# ---------------------------------------------------------------------------

def _reference_ascent(arcs, root, terminals, costs, nodes):
    """Naive Wong ascent (rebuilds a networkx DiGraph + ancestors/descendants
    each iteration, scans all arcs for the in-cut). Reference for equivalence;
    mirrors the pre-optimization implementation exactly."""
    from steinerpy.dual_ascent import EPS
    c = dict(costs)
    sat = {a for a in arcs if c[a] <= EPS}
    demand = set(terminals) - {root}
    lb = 0.0
    cuts = []
    seen = set()
    while True:
        S = nx.DiGraph()
        S.add_nodes_from(nodes)
        S.add_edges_from(sat)
        reach = (nx.descendants(S, root) | {root}) if root in S else {root}
        unconnected = demand - reach
        if not unconnected:
            return lb, c, cuts
        t = min(unconnected, key=lambda n: str(n))
        W = (nx.ancestors(S, t) | {t}) if t in S else {t}
        fw = frozenset(W)
        if fw not in seen:
            seen.add(fw)
            cuts.append(fw)
        delta_in = [a for a in arcs if a[1] in W and a[0] not in W]
        if not delta_in:
            return math.inf, c, cuts
        delta = min(c[a] for a in delta_in)
        lb += delta
        for a in delta_in:
            c[a] -= delta
            if c[a] <= EPS:
                sat.add(a)


def test_ascent_equivalence_random():
    """The optimized _ascent yields the SAME lower bound, reduced costs, and
    Steiner-cut set as the naive reference across random instances and roots."""
    import random
    from steinerpy.dual_ascent import _ascent, _node_universe, _arc_costs
    rng = random.Random(20260620)
    for _ in range(25):
        n = rng.randint(5, 12)
        g = nx.gnp_random_graph(n, rng.uniform(0.3, 0.7), seed=rng.randint(0, 10 ** 6))
        if g.number_of_edges() == 0:
            continue
        for a, b in g.edges:
            g.edges[a, b]["weight"] = rng.randint(1, 15)
        terms = rng.sample(list(g.nodes), min(4, n))
        arcs = list(g.edges()) + [(v, u) for (u, v) in g.edges()]
        costs = _arc_costs(g, arcs, "weight")
        for root in terms:
            nodes = _node_universe(arcs, terms, root)
            ga = _ascent(list(arcs), root, terms, costs, nodes)
            ref_lb, ref_c, ref_cuts = _reference_ascent(list(arcs), root, terms, costs, nodes)
            assert ga.lower_bound == pytest.approx(ref_lb) or (
                math.isinf(ga.lower_bound) and math.isinf(ref_lb))
            assert ga.reduced_costs == pytest.approx(ref_c)
            assert set(ga.cuts) == set(ref_cuts)


# ---------------------------------------------------------------------------
# A1: adaptive multi-root early-stop
# ---------------------------------------------------------------------------

def test_multi_root_early_stops_when_optimal(monkeypatch):
    """When the first (model) root already closes the gap, _multi_root_group
    stops without trying the remaining candidate roots."""
    import steinerpy.dual_ascent as da_mod
    from steinerpy.dual_ascent import _candidate_roots, _arc_costs

    g = _chain(4)  # 0-1-2-3, opt = 3, root 0 proves it
    terms = [0, 3]
    arcs = list(g.edges()) + [(v, u) for (u, v) in g.edges()]
    costs = _arc_costs(g, arcs, "weight")
    cand = _candidate_roots(terms, 0)
    assert len(cand) > 1  # there really are extra roots to skip

    calls = {"n": 0}
    orig = da_mod._ascent

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(da_mod, "_ascent", counting)
    ga, primal, ub = da_mod._multi_root_group(g, arcs, terms, costs, cand, "weight", False)

    assert calls["n"] == 1  # stopped after the first root
    assert ga.lower_bound == pytest.approx(3.0)
    assert ub == pytest.approx(3.0)


def test_adaptive_root_preserves_optimum():
    """The adaptive stop never changes the optimum the accelerator returns."""
    # tree
    g = _notebook_tree()
    assert SteinerProblem(g, [["A", "B", "D"]], preprocess=False).get_solution(
        time_limit=30, dual_ascent=True).objective == pytest.approx(
        SteinerProblem(g, [["A", "B", "D"]], preprocess=False).get_solution(time_limit=30).objective)
    # chain
    c = _chain(6)
    assert SteinerProblem(c, [[0, 5]], preprocess=False).get_solution(
        time_limit=30, dual_ascent=True).objective == pytest.approx(5.0)
    # forest (two groups)
    f = nx.Graph()
    for (a, b, w) in [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (0, 4, 10)]:
        f.add_edge(a, b, weight=w)
    fb = SteinerProblem(f, [[0, 2], [3, 4]], preprocess=False).get_solution(time_limit=30)
    fd = SteinerProblem(f, [[0, 2], [3, 4]], preprocess=False).get_solution(time_limit=30, dual_ascent=True)
    assert fd.objective == pytest.approx(fb.objective)
    # directed
    dg = nx.DiGraph()
    dg.add_edge("A", "B", weight=1); dg.add_edge("B", "C", weight=1); dg.add_edge("A", "C", weight=10)
    assert DirectedSteinerProblem(dg, root="A", terminals=["B", "C"]).get_solution(
        time_limit=30, dual_ascent=True).objective == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# B: heuristic-only mode (exact=False)
# ---------------------------------------------------------------------------

def _connected_per_group(edges, groups):
    h = nx.Graph()
    h.add_edges_from((u, v) for (u, v) in edges)
    for grp in groups:
        present = [t for t in grp if t in h]
        if len(present) < len(grp):
            return False
        comp = nx.node_connected_component(h, present[0])
        if any(t not in comp for t in present):
            return False
    return True


def test_heuristic_mode_no_ilp(monkeypatch):
    """exact=False returns a feasible primal without ever building/solving an ILP."""
    import steinerpy.objects as objmod

    def boom(*a, **k):
        raise AssertionError("ILP must not be built in heuristic mode")

    monkeypatch.setattr(objmod, "build_model", boom)
    monkeypatch.setattr(objmod, "run_model", boom)
    monkeypatch.setattr(objmod, "build_model_gurobi", boom)
    monkeypatch.setattr(objmod, "run_model_gurobi", boom)

    g = _notebook_tree()
    sol = SteinerProblem(g, [["A", "B", "D"]], preprocess=False).get_solution(exact=False)
    assert _connected_per_group(sol.edges, [["A", "B", "D"]])
    assert sol.objective == pytest.approx(3.0)


def test_heuristic_mode_reports_valid_gap():
    import random
    rng = random.Random(77)
    for _ in range(8):
        n = rng.randint(6, 11)
        g = nx.gnp_random_graph(n, 0.5, seed=rng.randint(0, 10 ** 6))
        if not nx.is_connected(g):
            g = nx.minimum_spanning_tree(nx.complete_graph(n))
        for a, b in g.edges:
            g.edges[a, b]["weight"] = rng.randint(1, 12)
        terms = rng.sample(list(g.nodes), 3)
        opt = SteinerProblem(g, [terms], preprocess=False).get_solution(time_limit=30).objective
        da = dual_ascent(SteinerProblem(g, [terms], preprocess=False))
        sol = SteinerProblem(g, [terms], preprocess=False).get_solution(exact=False)
        assert sol.gap >= -1e-12
        assert sol.objective + 1e-9 >= opt           # UB >= opt
        assert da.lower_bound <= opt + 1e-6          # LB <= opt <= UB
        assert sol.objective == pytest.approx(da.upper_bound)
        exp_gap = (da.upper_bound - da.lower_bound) / max(1.0, abs(da.upper_bound))
        assert sol.gap == pytest.approx(exp_gap)
        assert _connected_per_group(sol.edges, [terms])


def test_heuristic_mode_matches_exact_when_provably_optimal():
    g = _chain(6)  # dual ascent proves optimum on a path
    sol = SteinerProblem(g, [[0, 5]], preprocess=False).get_solution(exact=False)
    assert sol.gap == pytest.approx(0.0)
    exact = SteinerProblem(g, [[0, 5]], preprocess=False).get_solution(time_limit=30).objective
    assert sol.objective == pytest.approx(exact)


def test_heuristic_mode_directed():
    dg = nx.DiGraph()
    dg.add_edge("A", "B", weight=1)
    dg.add_edge("B", "C", weight=1)
    dg.add_edge("A", "C", weight=10)
    sol = DirectedSteinerProblem(dg, root="A", terminals=["B", "C"]).get_solution(exact=False)
    assert sol.objective == pytest.approx(2.0)
    assert sol.gap == pytest.approx(0.0)


def test_heuristic_mode_unsupported_raises():
    g = _chain(5)
    with pytest.raises(NotImplementedError):
        SteinerProblem(g, [[0, 4]], preprocess=False, budget=3.0).get_solution(exact=False)
    with pytest.raises(NotImplementedError):
        SteinerProblem(g, [[0, 4]], preprocess=False, max_degree=2).get_solution(exact=False)
