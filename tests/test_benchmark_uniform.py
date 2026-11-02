"""The uniform benchmark independently checks certificates and keeps failures."""

from types import SimpleNamespace
from pathlib import Path
import subprocess

import networkx as nx
import pytest

from benchmarks import benchmark_uniform as benchmark


@pytest.mark.parametrize("name", benchmark.CASES)
def test_generated_cases_exercise_uniform_mip(name):
    graph, terminals = benchmark.make_case(name)
    assert nx.is_connected(graph)
    assert len(terminals) > 10
    assert set(terminals) <= set(graph)
    assert {data["weight"] for _, _, data in graph.edges(data=True)} == {1}


@pytest.mark.parametrize(
    "edges,objective",
    [
        ([(0, 1)], 1),
        ([(0, 2)], 1),
        ([(0, 1), (1, 2), (1, 0)], 3),
        ([(0, 1), (1, 2)], 3),
    ],
)
def test_certificate_rejects_disconnected_invalid_duplicate_or_mispriced_edges(
    edges, objective
):
    with pytest.raises(AssertionError):
        benchmark.check_solution(
            nx.path_graph(3),
            [0, 2],
            SimpleNamespace(edges=edges, objective=objective, gap=0),
        )


def test_certificate_accepts_valid_unproven_tree():
    benchmark.check_solution(
        nx.path_graph(3),
        [0, 2],
        SimpleNamespace(edges=[(0, 1), (1, 2)], objective=2, gap=float("inf")),
    )


def test_pair_checker_distinguishes_incomplete_and_wrong_optima():
    common = dict(case="cycle", preprocess=False, repeat=0)
    optimal = dict(common, status="optimal", objective=10)
    benchmark.check_pairs([optimal, dict(common, status="error")])
    benchmark.check_pairs([optimal, dict(common, status="feasible", objective=12)])
    with pytest.raises(AssertionError, match="Conflicting"):
        benchmark.check_pairs([optimal, dict(common, status="optimal", objective=11)])
    with pytest.raises(AssertionError, match="beats"):
        benchmark.check_pairs([optimal, dict(common, status="feasible", objective=9)])


def test_worker_timeout_is_recorded(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(benchmark.subprocess, "run", timeout)
    row = benchmark.run_worker(
        SimpleNamespace(source_root=Path.cwd(), solver="highs", time_limit=5),
        "cycle",
        "off",
        "auto",
        0,
    )
    assert row["status"] == "timeout"
    assert row["config"] == "auto" and not row["preprocess"]
