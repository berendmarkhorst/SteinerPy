"""Validate the compiled long-edge path and preservation of its work cap."""
import random

import networkx as nx
import pytest

from steinerpy import graph_reducer as gr


@pytest.mark.parametrize('seed', [4, 17, 31])
def test_native_long_edges_match_python_and_shortest_paths(seed, monkeypatch):
    graph = nx.gnm_random_graph(160, 800, seed=seed)
    rng = random.Random(seed)
    for u, v in graph.edges:
        graph[u][v]['cost'] = rng.randrange(21)  # includes zero and equal-cost paths
    calls = []
    original = gr._long_edge_deletions_scipy
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(gr, '_long_edge_deletions_scipy', record)
    native = gr.long_edge_deletions(graph, weight='cost', jobs=1)
    assert calls
    monkeypatch.setattr(gr, 'HAS_SCIPY', False)
    python = gr.long_edge_deletions(graph, weight='cost', jobs=1)
    expected = set()
    for u in graph:
        distances = nx.single_source_dijkstra_path_length(graph, u, weight='cost')
        for v, attrs in graph[u].items():
            if distances[v] < attrs['cost'] - 1e-9:
                expected.add((u, v))
    assert native == python == expected
    assert list(native) == list(python)  # preserve deterministic deletion order


@pytest.mark.parametrize('nodes,edges,cap', [(160, 800, 4), (50, 200, 2000), (160, 200, 2000)])
def test_work_cap_and_small_sparse_graphs_use_python(nodes, edges, cap, monkeypatch):
    graph = nx.gnm_random_graph(nodes, edges, seed=6)
    nx.set_edge_attributes(graph, 1, 'weight')
    def unexpected(*args, **kwargs):
        pytest.fail('Native Dijkstra must not bypass the work cap or size/density gates')
    monkeypatch.setattr(gr, '_long_edge_deletions_scipy', unexpected)
    assert gr.long_edge_deletions(graph, max_settle=cap, jobs=1) == set()
