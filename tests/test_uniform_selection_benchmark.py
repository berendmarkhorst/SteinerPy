"""Selection evidence uses reproducible, structurally varied graph instances."""

import networkx as nx
import pytest
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from benchmarks import uniform_selection as benchmark
from benchmarks.uniform_selection import make_case, make_manifest, structural_features


def test_discovery_and_holdout_are_fixed_and_disjoint():
    discovery, heldout = make_manifest("discovery"), make_manifest("heldout")
    assert len(discovery) == len(heldout) == 52
    assert discovery == make_manifest("discovery")
    assert not {row["id"] for row in discovery} & {row["id"] for row in heldout}
    assert not {row["seed"] for row in discovery} & {row["seed"] for row in heldout}


@pytest.mark.parametrize("split", ["discovery", "heldout", "challenge"])
def test_every_instance_is_reproducible_connected_uniform_and_nontrivial(split):
    for spec in make_manifest(split):
        graph, terminals = make_case(spec)
        again, again_terminals = make_case(spec)
        assert nx.is_connected(graph)
        assert len(set(terminals)) > 10
        assert set(terminals) <= set(graph)
        assert set(graph.edges) == set(again.edges)
        assert terminals == again_terminals
        assert {data["weight"] for _, _, data in graph.edges(data=True)} == {1}


def test_incidence_and_subdivision_families_challenge_independence_rule():
    for spec in make_manifest("discovery"):
        if spec["family"] not in {"bipartite", "regular_incidence", "subdivision"}:
            continue
        graph, terminals = make_case(spec)
        features = structural_features(graph, terminals)
        assert features["aligned_bipartite"]
        assert features["steiner_independent"]
        if spec["family"] == "subdivision":
            assert features["steiner_degree2_fraction"] == 1
        elif spec["family"] == "regular_incidence":
            assert features["steiner_degree_min"] == features["steiner_degree_max"]
            assert features["steiner_degree_min"] in {3, 5}


def test_descriptors_distinguish_grid_terminal_placements():
    specs = {row["family"]: row for row in make_manifest("discovery")}
    graph, terminals = make_case(specs["checkerboard_grid"])
    assert structural_features(graph, terminals)["aligned_bipartite"]
    graph, terminals = make_case(specs["sparse_grid"])
    assert structural_features(graph, terminals)["steiner_edges"] > 0


def test_challenges_preserve_intended_counterexample_structure():
    manifest = make_manifest("challenge")
    assert len(manifest) == 12
    for spec in manifest:
        graph, terminals = make_case(spec)
        features = structural_features(graph, terminals)
        if spec["family"] == "torus":
            assert features["degree_min"] == features["degree_max"] == 4
        elif spec["family"] == "terminal_cliques":
            assert features["terminal_components"] == 2
        elif spec["family"] == "dangling_clique":
            assert features["degree_mean"] > 4.5
        elif spec["family"] == "cube_leaves":
            assert sum(degree == 1 for _, degree in graph.degree()) == 60


def test_custom_manifest_keeps_ids_and_rejects_duplicates():
    data = [{"id": "public-case", "path": "cases/input.stp"}]
    path = SimpleNamespace(read_text=lambda **kwargs: json.dumps(data))
    assert benchmark.load_manifest(path, "external") == [
        dict(data[0], family="external", split="external", seed=None)
    ]
    data.append(data[0])
    with pytest.raises(ValueError, match="unique"):
        benchmark.load_manifest(path, "external")


def test_external_stp_case_uses_existing_parser(monkeypatch):
    from benchmarks import stp_parser

    graph = nx.path_graph(3)
    monkeypatch.setattr(stp_parser, "read_stp", lambda path: (graph, [[0, 2]]))
    actual, terminals = make_case({"path": "example.stp"})
    assert actual is graph and terminals == [0, 2]
    monkeypatch.setattr(stp_parser, "read_stp", lambda path: (graph, [[0], [2]]))
    with pytest.raises(ValueError, match="one terminal group"):
        make_case({"path": "forest.stp"})


def test_worker_timeout_stops_descendants(monkeypatch):
    class Process:
        pid = 1234

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            return "", ""

    process = Process()
    stopped = []
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *args, **kwargs: process)
    if benchmark.os.name == "nt":
        monkeypatch.setattr(
            benchmark.subprocess,
            "run",
            lambda command, **kwargs: stopped.append(command),
        )
    else:
        monkeypatch.setattr(
            benchmark.os, "killpg", lambda pid, sig: stopped.append(pid)
        )
    args = SimpleNamespace(
        source_root=Path.cwd(),
        split="discovery",
        preprocess="off",
        solver="highs",
        time_limit=5,
        worker_timeout=30,
        manifest=None,
    )
    result = benchmark.run_worker(args, make_manifest("discovery")[0], "node", 0)
    assert result["status"] == "timeout"
    assert process.calls == 2 and stopped
