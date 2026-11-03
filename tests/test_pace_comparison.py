"""Keep the follow-up sample independent of outcomes and its limits matched."""

from pathlib import Path

import pytest

from benchmarks import pace_comparison as comparison


def test_fixed_samples_cover_additional_seeds_and_public_private_inputs(
    monkeypatch, tmp_path
):
    cases = comparison.combined_cases()
    assert len(cases) == len({r["id"] for r in cases}) == 12
    assert {r["seed"] for r in cases if r["name"] == "signed-rewards"} == set(
        range(40, 46)
    )
    assert {r["seed"] for r in cases if r["name"] == "resource-tree"} == set(
        range(42, 46)
    )
    references = [
        dict(
            track=t,
            instance=f"instance{n:03}.gr",
            split="private" if n % 2 == 0 else "public",
        )
        for t in (1, 2, 3)
        for n in (1, 50, 101, 150)
    ]
    monkeypatch.setattr(comparison.pace, "reference_cases", lambda *a, **kw: references)
    selected = comparison.pace_cases(tmp_path, tmp_path)
    assert len(selected) == 12
    for track in (1, 2, 3):
        rows = [r for r in selected if r["track"] == track]
        assert sum(r["split"] == "private" for r in rows) == 2
        assert all(
            Path(r["binary"]).name == ("track2" if track == 2 else "track1")
            for r in rows
        )
    references.pop()
    with pytest.raises(ValueError, match="all twelve inputs"):
        comparison.pace_cases(tmp_path, tmp_path)


def test_matched_pair_alternates_methods_and_uses_public_api(monkeypatch, tmp_path):
    seen = []

    def execute(case, args, identity, directory, result_path, name):
        assert args.time_limit == identity["time_limit"] == 120
        assert args.memory_gb == identity["memory_gb"] == 8
        assert identity["threads"] == 1
        assert identity["commit"] == "candidate-sha"
        assert case["source_root"] == str(tmp_path / "candidate")
        assert args.worker_module == (
            "benchmarks.pace_wata"
            if args.config == "wata"
            else "benchmarks.connectivity_pilot"
        )
        seen.append((identity["repeat"], args.config))
        return dict(status="optimal")

    monkeypatch.setattr(comparison.pace, "execute_case", execute)
    rows = comparison.run_pace_pair(
        dict(id="track1-instance001"),
        tmp_path / "candidate",
        dict(commits=dict(candidate="candidate-sha")),
        tmp_path,
    )
    assert len(rows) == 4
    assert seen == [(0, "api-highs"), (0, "wata"), (1, "wata"), (1, "api-highs")]
