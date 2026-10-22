"""Keep the pilot bounded, paired, isolated and checked on original solutions."""

import argparse
from pathlib import Path

import pytest

from benchmarks import connectivity_pilot as pilot


def test_fixed_sample_and_paired_order(monkeypatch, tmp_path):
    references = [
        dict(track=track, instance=f"instance{number:03}.gr")
        for track in (1, 2, 3)
        for number in (1, 2, 3)
    ]
    monkeypatch.setattr(pilot.pace, "reference_cases", lambda *args: references)
    selected = pilot.cases(tmp_path)
    assert len(selected) == len({case["id"] for case in selected}) == 12
    assert sum(case["config"] == "composition-highs" for case in selected) == 6
    assert [case["seed"] for case in selected if case["kind"] == "combined"] == [
        42,
        43,
        44,
        44,
    ]
    roots = {label: tmp_path / label for label in ("baseline", "candidate")}
    commits = dict(baseline="before", candidate="after")
    observed = []

    def execute(case, args, identity, directory, result_path, name):
        assert args.time_limit == 120 and args.memory_gb == 8
        assert identity["threads"] == 1
        assert args.config in {"highs", "composition-highs"}
        assert Path(case["source_root"]) == roots[identity["label"]]
        assert identity["commit"] == commits[identity["label"]]
        observed.append((identity["repeat"], identity["label"]))
        return dict(status="optimal")

    monkeypatch.setattr(pilot.pace, "execute_case", execute)
    for case in selected:
        pilot.run_pair(case, roots, dict(commits=commits), tmp_path)
    assert len(observed) == 48
    assert (
        observed
        == [(0, "baseline"), (0, "candidate"), (1, "candidate"), (1, "baseline")] * 12
    )


def isolated(case, tmp_path, limit=30, root=None):
    root = root or Path(__file__).resolve().parents[1]
    args = argparse.Namespace(
        config=case["config"],
        time_limit=limit,
        memory_gb=8,
        worker_module="benchmarks.connectivity_pilot",
    )
    return pilot.pace.execute_case(
        dict(case, source_root=str(root)),
        args,
        dict(commit="test"),
        tmp_path,
        tmp_path / "result.json",
        "test",
    )


@pytest.mark.parametrize("config", ["highs", "composition-highs"])
def test_isolated_pace_worker_keeps_original_solution_checks(tmp_path, config):
    graph = tmp_path / "test.gr"
    graph.write_text(
        "SECTION Graph\nNodes 3\nEdges 3\nE 1 2 1\nE 2 3 1\nE 1 3 5\nEND\n"
        "SECTION Terminals\nTerminals 2\nT 1\nT 3\nEND\nEOF\n"
    )
    row = isolated(
        dict(
            kind="pace",
            config=config,
            path=str(graph),
            reference_lower=2,
            reference_upper=2,
        ),
        tmp_path,
    )
    assert row["status"] == "optimal" and row["objective"] == 2
    assert row["feasible"] and row["reference_optimal"]
    assert len(row["solution_edges"]) == 2
    assert not row["gurobi_started"]


def test_combined_worker_uses_public_dispatch_and_records_a_certificate(tmp_path):
    row = isolated(
        dict(kind="combined", config="highs", name="resource-tree", seed=42), tmp_path
    )
    assert row["status"] in {"optimal", "feasible"}, row
    assert row["feasible"] and row["objective"] >= 96
    assert row["implementation"] == "FeatureComposition"
    assert row["solution_edges"] and row["selected_nodes"]
    assert row["phases_seconds"]["model_build"] > 0
    assert row["backend_runs"] and len(row["input_sha256"]) == 64


def test_worker_deadline_includes_imports_and_building(tmp_path):
    row = isolated(
        dict(kind="combined", config="highs", name="resource-tree", seed=42),
        tmp_path,
        limit=0.001,
    )
    assert row["status"] == "timeout" and not row["feasible"]
    assert "objective" not in row and "solver_proven_optimal" not in row


def test_worker_imports_the_requested_checkout(tmp_path):
    package = tmp_path / "checkout" / "steinerpy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('requested checkout marker')\n"
    )
    row = isolated(dict(kind="pace", config="highs"), tmp_path, root=package.parent)
    assert row["status"] == "error"
    assert "requested checkout marker" in row["error"]


def test_changed_inputs_or_inconsistent_proofs_are_rejected():
    row = dict(id="sample", input_sha256="same", status="optimal", objective=3)
    pilot.check_results([row, dict(row, status="feasible", objective=4)])
    with pytest.raises(ValueError, match="Input changed"):
        pilot.check_results([row, dict(row, input_sha256="different")])
    with pytest.raises(AssertionError):
        pilot.check_results([row, dict(row, objective=4)])
