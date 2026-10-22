"""A Slurm task must not solve the rest of its input directory."""

import csv

from benchmarks import run_benchmarks
from benchmarks.stp_parser import read_stp


def test_single_instance_does_not_run_siblings(tmp_path, monkeypatch):
    chosen = tmp_path / "chosen.stp"
    chosen.touch()
    (tmp_path / "other.stp").touch()
    calls = []

    def solve(path, *args):
        calls.append(path)
        row = dict.fromkeys(run_benchmarks.FIELDS, "")
        row.update(instance="chosen", status="ok")
        return row

    monkeypatch.setattr(run_benchmarks, "_solve", solve)
    output = tmp_path / "result.csv"
    run_benchmarks.main([
        "--instances", str(chosen), "--full", "--out", str(output)
    ])
    assert calls == [str(chosen)]
    with output.open() as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_parser_retains_declared_isolated_nodes(tmp_path):
    instance = tmp_path / "isolated.gr"
    instance.write_text(
        "SECTION Graph\nNodes 3\nEdges 1\nE 1 2 5\nEND\n"
        "SECTION Terminals\nTerminals 2\nT 1\nT 3\nEND\nEOF\n"
    )
    graph, terminals = read_stp(instance)
    assert set(graph) == {1, 2, 3}
    assert terminals == [[1, 3]]
