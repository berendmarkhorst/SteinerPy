"""Reuse a preprocessing bound only for the exact state it was computed on."""
import importlib

import networkx as nx
import pytest

from steinerpy import SteinerProblem
from steinerpy.graph_reducer import ReductionTracker

da = importlib.import_module('steinerpy.dual_ascent')


def _count_ascents(monkeypatch):
    calls = []
    original = da._multi_root_group
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(da, '_multi_root_group', record)
    return calls


def _problem():
    graph = nx.complete_graph(4)
    nx.set_edge_attributes(graph, 1.0, 'weight')
    return SteinerProblem(graph, [[0, 1, 2, 3]], heavy=False,
                          da_reduce=True, dual_ascent=True)


def test_preprocessing_ascent_is_consumed_once(monkeypatch):
    calls = _count_ascents(monkeypatch)
    problem = _problem()
    assert len(calls) == 1
    assert problem.get_solution().objective == 3.0
    assert len(calls) == 1
    assert not problem._preprocessing_ascent
    assert problem.get_solution().objective == 3.0
    assert len(calls) == 2


@pytest.mark.parametrize('change', ['costs', 'terminals', 'roots', 'nodes', 'edges', 'arcs', 'weight'])
def test_changed_inputs_invalidate_preprocessing_bound(change, monkeypatch):
    calls = _count_ascents(monkeypatch)
    problem = _problem()
    assert problem._preprocessing_ascent
    if change == 'costs':
        nx.set_edge_attributes(problem.graph, 2.0, 'weight')
    elif change == 'terminals':
        problem.terminal_groups = [[0, 1]]
    elif change == 'roots':
        problem.roots = [1]
    elif change == 'nodes':
        problem.graph.add_node('isolated')
    elif change == 'edges':
        problem.graph.remove_edge(0, 1)
        problem.edges = list(problem.graph.edges())
        problem.arcs = problem.edges + [(v, u) for u, v in problem.edges]
    elif change == 'arcs':
        problem.arcs = list(reversed(problem.arcs))
    elif change == 'weight':
        nx.set_edge_attributes(problem.graph, 5.0, 'other')
        problem.weight = 'other'
    reused = da.dual_ascent(problem)
    assert len(calls) == 2
    fresh = da.dual_ascent(problem)
    assert reused.lower_bound == fresh.lower_bound
    assert reused.upper_bound == fresh.upper_bound
    assert reused.primal_edges == fresh.primal_edges


def test_last_pass_graph_changes_do_not_leave_a_stale_bound():
    graph = nx.complete_graph(3)
    nx.set_edge_attributes(graph, 1.0, 'weight')
    graph.add_edge(0, 'leaf', weight=20)
    reuse = {'old': 'entry'}
    reduced = da.reduce_graph_with_dual_ascent(
        graph, [[0, 1, 2]], 'weight', ReductionTracker(), max_passes=1, _reuse=reuse)
    assert 'leaf' not in reduced
    assert not reuse
