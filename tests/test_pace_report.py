"""Published summaries must include failures and distinguish pending work."""

import json

import pytest

from benchmarks.pace_report import load_rows, metrics


def test_pending_cases_do_not_produce_a_final_quality_score():
    rows = [
        dict(
            status="optimal",
            feasible=True,
            reference_optimal=True,
            reference_ratio=1,
            wall_seconds=2,
        ),
        dict(status="pending"),
    ]
    result = metrics(rows)
    assert result["expected"] == 2 and result["completed"] == 1
    assert result["optimal"] == 1 and result["quality"] is None


def test_failed_cases_stay_in_quality_denominator():
    rows = [
        dict(
            status="feasible",
            feasible=True,
            reference_optimal=False,
            reference_ratio=0.9,
            wall_seconds=2,
        ),
        dict(status="timeout"),
        dict(status="error", feasible=False, reference_ratio=1),
    ]
    result = metrics(rows)
    assert result["valid"] == 1 and result["optimal"] == 0
    assert result["quality"] == 30
    assert result["timeouts"] == 1 and result["errors"] == 1


def test_report_keeps_pending_cases_and_rejects_mixed_revisions(tmp_path):
    case = dict(track=1, instance="instance001.gr", instance_sha256="input")
    run = dict(commit="a", dataset_commit="data", time_limit=90, memory_gb=8)
    manifest = dict(
        run=run,
        cases=[case],
        config="highs",
        time_limit=90,
        memory_gb=8,
    )
    (tmp_path / "manifest-highs-0.json").write_text(json.dumps(manifest))
    assert load_rows(tmp_path)[0]["status"] == "pending"
    directory = tmp_path / "case"
    directory.mkdir()
    result = dict(case, status="optimal", run=dict(run, config="highs"))
    (directory / "result.json").write_text(json.dumps(result))
    assert load_rows(tmp_path)[0]["status"] == "optimal"
    result["run"]["commit"] = "changed"
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="differs from manifest"):
        load_rows(tmp_path)
