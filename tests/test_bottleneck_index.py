"""Tree-path index equivalence, disconnected cases, and storage scaling."""
import random

import networkx as nx
import pytest

from steinerpy.graph_reducer import _bottleneck_from_mst


@pytest.mark.parametrize('seed', range(10))
def test_bottleneck_queries_match_explicit_tree_paths(seed):
    rng = random.Random(seed)
    tree = nx.Graph()
    labels = ['root', ('terminal', 1)] + list(range(2, 45))
    tree.add_nodes_from(labels)
    for i in range(1, len(labels)):
        tree.add_edge(labels[i], labels[rng.randrange(i)], weight=rng.randrange(0, 30))
    index = _bottleneck_from_mst(tree, labels)
    for a in labels:
        for b in labels:
            path = nx.shortest_path(tree, a, b)
            expected = max((tree[u][v]['weight'] for u, v in zip(path, path[1:])), default=0.0)
            assert index[a][b] == expected


def test_disconnected_missing_and_empty_terminal_rows():
    forest = nx.Graph()
    forest.add_edge('a', 'b', weight=4)
    forest.add_edge('c', 'd', weight=2)
    forest.add_node('isolated')
    index = _bottleneck_from_mst(forest, ['a', 'c', 'isolated', 'missing'])
    assert dict(index['a']) == {'a': 0.0, 'b': 4}
    assert dict(index['isolated']) == {'isolated': 0.0}
    assert index['a'].get('c', float('inf')) == float('inf')
    assert index.get('missing') is None
    with pytest.raises(KeyError):
        index['a']['c']
    assert _bottleneck_from_mst(nx.Graph(), ['missing']) == {}


def test_deep_tree_does_not_materialize_all_pairs():
    # Also exercises iterative construction well beyond Python's recursion limit.
    n = 4096
    tree = nx.path_graph(n)
    nx.set_edge_attributes(tree, 1.0, 'weight')
    tree[2047][2048]['weight'] = 9.0
    index = _bottleneck_from_mst(tree, tree.nodes)
    assert index[0][n-1] == 9.0
    assert index[0][2047] == 1.0
    cells = sum(len(level) for level in index.up + index.maximum)
    assert cells <= 2 * n * n.bit_length()
    for node in range(n):
        assert index[node][n-1] == (9.0 if node <= 2047 else 1.0 if node < n-1 else 0.0)
    # Queries do not accumulate a new all-pairs cache.
    assert sum(len(level) for level in index.up + index.maximum) == cells
