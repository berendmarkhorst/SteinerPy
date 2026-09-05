"""Validate accelerated separation against independent NetworkX min cuts."""
from types import SimpleNamespace
import random

import networkx as nx
import pytest

from steinerpy import _fastgraph as fg
from steinerpy import mathematical_model as mm

EPS = 1e-6


def _problem(graph, groups):
    return SimpleNamespace(nodes=list(graph), arcs=list(graph.edges()),
                           roots=[g[0] for g in groups], terminal_groups=groups)


@pytest.mark.parametrize('threads', [1, 3])
@pytest.mark.parametrize('back_cuts', [False, True])
@pytest.mark.parametrize('scipy', [False, True])
def test_separator_against_networkx(threads, back_cuts, scipy, monkeypatch):
    """Every emitted cut is violated and valid; every violated demand is found.

    Exercise directed, disconnected, fractional, integer, and multi-group
    solutions, including tiny capacities on either side of the tolerance.
    """
    monkeypatch.setattr(mm, 'HAS_SCIPY', scipy)
    eps = EPS
    for seed in range(20):
        rng = random.Random(seed)
        labels = ['root', 1, ('terminal', 2), 3, 4, 5, 6, 7]
        graph = nx.DiGraph()
        graph.add_nodes_from(labels)
        graph.add_edges_from((u, v) for u in labels for v in labels
                             if u != v and rng.random() < 0.3)
        problem = _problem(graph, [[labels[0], labels[2], labels[4]],
                                   [labels[1], labels[5], labels[7]]])
        choices = [0.0, 1.0] if seed % 2 else [0, 1e-6, 2e-6, .2, .5, .8, 1.0]
        values = {(k, a): rng.choice(choices) for k in range(2) for a in problem.arcs}
        demands = {(0, 0): 1.0, (0, 1): .5 if seed % 3 else 0.0, (1, 1): 1.0}
        cuts = mm.find_violated_cuts_from_values(
            problem, values, demands, eps=eps, back_cuts=back_cuts, threads=threads)
        for k in range(2):
            capacity_graph = graph.copy()
            for a in problem.arcs:
                cap = values[k, a] + eps
                if scipy:
                    cap = round(cap * fg.FLOW_SCALE) / fg.FLOW_SCALE
                capacity_graph.edges[a]['capacity'] = cap
            for l in range(k, 2):
                demand = demands[k, l]
                relevant = [arcs for kk, ll, arcs in cuts if (kk, ll) == (k, l)]
                if demand <= eps:
                    assert not relevant
                    continue
                root = problem.roots[k]
                terminals = [t for t in problem.terminal_groups[l] if t != root]
                disconnected_by_cut = set()
                for arcs in relevant:
                    assert sum(capacity_graph.edges[a]['capacity'] for a in arcs) < demand - eps
                    remainder = graph.copy()
                    remainder.remove_edges_from(arcs)
                    missing = {t for t in terminals if not nx.has_path(remainder, root, t)}
                    assert missing, 'Returned arcs must separate a demanded terminal'
                    disconnected_by_cut.update(missing)
                for terminal in terminals:
                    minimum, _ = nx.minimum_cut(capacity_graph, root, terminal)
                    assert (minimum < demand - eps) == (terminal in disconnected_by_cut)


def test_satisfied_paths_skip_max_flow(monkeypatch):
    graph = nx.DiGraph([('r', 'a'), ('a', 'b'), ('b', 'c')])
    problem = _problem(graph, [['r', 'a', 'b', 'c']])
    def unexpected(*args, **kwargs):
        pytest.fail('A sufficient-capacity path must not need maximum flow')
    monkeypatch.setattr(mm, 'min_cut_scipy', unexpected)
    # A fractional demand is covered by the same sufficient path test.
    values = {(0, a): .5 for a in problem.arcs}
    assert mm.find_violated_cuts_from_values(problem, values, {(0, 0): .5}) == []


def test_split_flow_is_not_mistaken_for_disconnection(monkeypatch):
    """No individually sufficient path, but two half-capacity paths suffice."""
    graph = nx.DiGraph([('r', 'a'), ('a', 't'), ('r', 'b'), ('b', 't')])
    problem = _problem(graph, [['r', 't']])
    values = {(0, a): .5 for a in problem.arcs}
    calls = []
    original = mm.min_cut_scipy
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(mm, 'min_cut_scipy', record)
    assert mm.find_violated_cuts_from_values(problem, values, {(0, 0): 1.0}) == []
    assert calls


def test_networkx_source_cut_requires_completed_flow(monkeypatch):
    """A preflow can leave excess at a and incorrectly return the r->a cut."""
    graph = nx.DiGraph([('r', 'a'), ('a', 't')])
    problem = _problem(graph, [['r', 't']])
    monkeypatch.setattr(mm, 'HAS_SCIPY', False)
    values = {(0, ('r', 'a')): 1.0, (0, ('a', 't')): .5}
    cuts = mm.find_violated_cuts_from_values(
        problem, values, {(0, 0): 1.0}, back_cuts=False)
    assert cuts == [(0, 0, [('a', 't')])]


def test_satisfied_flow_skips_residual_traversal(monkeypatch):
    csr = fg.ArcCSR(['r', 'a', 'b', 't'],
                    [('r', 'a'), ('a', 't'), ('r', 'b'), ('b', 't')])
    capacities = csr.build_int_csr([.5, .5, .5, .5])
    def unexpected(*args, **kwargs):
        pytest.fail('Satisfied flows must not traverse residual graphs')
    monkeypatch.setattr(fg, '_sp_bfo', unexpected)
    assert fg.min_cut_scipy(capacities, 0, 3, required=1.0) == (1.0, set(), set())


@pytest.mark.parametrize('side', [set(), {0}, {1, 2}, {0, 1, 2, 3}])
@pytest.mark.parametrize('scipy', [False, True])
def test_cut_extraction_with_arbitrary_labels(side, scipy, monkeypatch):
    csr = fg.ArcCSR(['r', 7, ('t', 1), 'isolated'],
                    [('r', 7), (7, 'r'), (7, ('t', 1)), (('t', 1), 'r')])
    monkeypatch.setattr(fg, 'HAS_SCIPY', scipy)
    expected = {a for a in csr.arcs if csr.node_index[a[0]] in side
                and csr.node_index[a[1]] not in side}
    assert set(csr.cut_arcs(side)) == expected


@pytest.mark.parametrize('required', [1 - EPS, 1, 1 + EPS])
def test_capacity_screen_agrees_with_scaled_flow_at_tolerance(required):
    csr = fg.ArcCSR(range(3), [(0, 1), (1, 2)])
    for capacity in [1 - 2*EPS, 1 - EPS, 1, 1 + EPS]:
        matrix = csr.build_int_csr([capacity, capacity])
        reached = fg.capacity_reachable(matrix, 0, required)
        value, _, _ = fg.min_cut_scipy(matrix, 0, 2)
        assert bool(reached[2]) == (value >= required)


def test_separation_threads_remain_configurable(monkeypatch):
    monkeypatch.delenv('STEINERPY_SEP_THREADS', raising=False)
    assert mm._sep_thread_count() == 1
    monkeypatch.setenv('STEINERPY_SEP_THREADS', '4')
    assert mm._sep_thread_count() == 4


@pytest.mark.parametrize('back_cuts', [False, True])
@pytest.mark.parametrize('noise', [-1e-9, 0.0, 1e-9])
def test_disconnected_integral_candidates_need_no_max_flow(monkeypatch, back_cuts, noise):
    graph = nx.DiGraph([('r', 'a'), ('a', 't'), ('a', 'u'), ('t', 'u')])
    problem = _problem(graph, [['r', 't', 'u']])
    values = {(0, a): 1.0 - noise if a == ('r', 'a') else noise for a in problem.arcs}
    def unexpected(*args, **kwargs):
        pytest.fail('Integral disconnection should have a reachability certificate')
    monkeypatch.setattr(mm, 'min_cut_scipy', unexpected)
    cuts = mm.find_violated_cuts_from_values(problem, values, {(0, 0): 1.0},
                                            back_cuts=back_cuts)
    assert cuts
    assert len(cuts) == len({(k, l, frozenset(arcs)) for k, l, arcs in cuts})
    for _, _, arcs in cuts:
        assert sum(values[0, a] for a in arcs) < 1.0 - EPS
        remainder = graph.copy()
        remainder.remove_edges_from(arcs)
        assert any(not nx.has_path(remainder, 'r', t) for t in ['t', 'u'])


def test_integral_certificate_checks_capacity_and_falls_back(monkeypatch):
    # Large creep capacities can collectively satisfy a demand even with all
    # root arcs at zero. The quick partition is not a violated cut here.
    graph = nx.DiGraph()
    graph.add_edges_from(('r', i) for i in range(5))
    graph.add_edges_from((i, 't') for i in range(5))
    problem = _problem(graph, [['r', 't']])
    values = {(0, a): 0.0 if a[0] == 'r' else 1.0 for a in problem.arcs}
    calls = []
    original = mm.min_cut_scipy
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(mm, 'min_cut_scipy', record)
    assert mm.find_violated_cuts_from_values(problem, values, {(0, 0): 1.0}, eps=.2) == []
    assert calls


def test_integral_cut_for_graph_without_arcs():
    graph = nx.DiGraph()
    graph.add_nodes_from(['r', 't'])
    problem = _problem(graph, [['r', 't']])
    assert mm.find_violated_cuts_from_values(problem, {}, {(0, 0): 1.0}) == [(0, 0, [])]


def test_minimum_cuts_can_be_requested_for_integral_values(monkeypatch):
    graph = nx.DiGraph([('r', 'a'), ('a', 't')])
    problem = _problem(graph, [['r', 't']])
    values = {(0, ('r', 'a')): 1.0, (0, ('a', 't')): 0.0}
    def unexpected(*args, **kwargs):
        pytest.fail('LP separation must retain minimum cuts when requested')
    monkeypatch.setattr(mm, '_integral_cut_certificates', unexpected)
    cuts = mm.find_violated_cuts_from_values(problem, values, {(0, 0): 1.0},
                                            integral_cuts=False)
    assert cuts == [(0, 0, [('a', 't')])]
