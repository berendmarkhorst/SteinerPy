"""Process workers must receive graph payloads and preserve input order."""

import os

import networkx as nx

from steinerpy._parallel import get_shared, pmap


def _distance_from_shared_graph(target):
    graph, root = get_shared()
    distance = nx.shortest_path_length(graph, root, target, weight="weight")
    return target, distance, os.getpid()


def test_pmap_process_workers_match_serial_results_and_release_shared_payload():
    graph = nx.path_graph(5)
    nx.set_edge_attributes(graph, 2, "weight")
    targets = [4, 0, 3, 2]
    expected = [(4, 6), (0, 2), (3, 4), (2, 2)]
    parent_pid = os.getpid()

    for jobs in (1, 2):
        results = pmap(
            _distance_from_shared_graph,
            targets,
            jobs=jobs,
            shared=(graph, 1),
            min_items=1,
        )
        assert [(target, distance) for target, distance, _ in results] == expected
        assert all((pid == parent_pid) == (jobs == 1) for _, _, pid in results)
        assert get_shared() is None
