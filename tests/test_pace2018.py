"""Benchmark correctness and deadline accounting, independently of PACE data."""

import argparse
import json
import os
import time

import networkx as nx
import pytest

from benchmarks.pace2018 import (
    WLS_COOLDOWN,
    comparable_run,
    reference_cases,
    run_case,
    validate_solution,
    wls_slot,
)


def test_manifest_separates_tracks_and_public_private_instances(tmp_path):
    for track in (1, 2, 3):
        (tmp_path / "Track{}".format(track)).mkdir()
        header = "paceName,opt\n" if track < 3 else "paceName,lower,upper\n"
        rows = (
            "instance001.gr ,3\ninstance002.gr ,4\n"
            if track < 3
            else "instance001.gr ,3,5\n"
        )
        (tmp_path / "track{}.csv".format(track)).write_text(header + rows)
        for number in ([1, 2] if track < 3 else [1]):
            (
                tmp_path / "Track{}".format(track) / "instance{:03}.gr".format(number)
            ).touch()
    cases = reference_cases(tmp_path, [1, 2, 3])
    assert len(cases) == 5
    assert [c["split"] for c in cases[:2]] == ["public", "private"]
    assert cases[-1]["reference_lower"] == 3 and cases[-1]["reference_upper"] == 5
    assert len({(c["track"], c["instance"]) for c in cases}) == 5


def test_sparse_dataset_still_requires_selected_instances(tmp_path):
    (tmp_path / "Track1").mkdir()
    graph = tmp_path / "Track1" / "instance001.gr"
    graph.write_text("selected instance")
    (tmp_path / "track1.csv").write_text(
        "paceName,opt\ninstance001.gr,3\ninstance002.gr,4\n"
    )
    rows = reference_cases(tmp_path, [1], instances={"instance001.gr"})
    assert len(rows) == 1 and rows[0]["reference_lower"] == 3
    assert len(rows[0]["instance_sha256"]) == 64
    with pytest.raises(FileNotFoundError):
        reference_cases(tmp_path, [1])
    graph.unlink()
    with pytest.raises(FileNotFoundError):
        reference_cases(tmp_path, [1], instances={"instance001.gr"})


def test_feasible_timeout_incumbent_is_not_claimed_optimal():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(1, 2, 2), (2, 3, 2), (1, 3, 3)])
    result = validate_solution(graph, [1, 3], [(1, 2), (2, 3)], 4, 0.25, 3, 3)
    assert result["feasible"] and result["status"] == "feasible"
    assert not result["reference_optimal"] and not result["solver_proven_optimal"]
    assert result["reference_ratio"] == 0.75
    with pytest.raises(ValueError, match="claims an optimum"):
        validate_solution(graph, [1, 3], [(1, 2), (2, 3)], 4, 0, 3, 3)


@pytest.mark.parametrize(
    "edges,objective",
    [
        ([(1, 2)], 2),
        ([(1, 3)], 4),
        ([(1, 4)], 3),
        ([(1, 2), (2, 3), (1, 3)], 7),
    ],
)
def test_invalid_original_solution_is_rejected(edges, objective):
    graph = nx.Graph()
    graph.add_weighted_edges_from([(1, 2, 2), (2, 3, 2), (1, 3, 3)])
    with pytest.raises(ValueError):
        validate_solution(graph, [1, 3], edges, objective, 0, 3, 3)


def test_resume_identity_allows_a_new_slurm_job_but_not_new_limits():
    assert comparable_run(
        dict(commit="abc", slurm_job="1", hostname="n1")
    ) == comparable_run(dict(commit="abc", slurm_job="2", hostname="n2"))
    assert comparable_run(dict(time_limit=10)) != comparable_run(dict(time_limit=20))


def make_case(tmp_path):
    path = tmp_path / "instance001.gr"
    path.write_text(
        "SECTION Graph\nNodes 3\nEdges 3\nE 1 2 1\nE 2 3 1\nE 1 3 5\nEND\nSECTION Terminals\nTerminals 2\nT 1\nT 3\nEND\nEOF\n"
    )
    case = dict(
        track=1,
        instance=path.name,
        path=str(path),
        split="public",
        reference_lower=2,
        reference_upper=2,
        instance_sha256="test",
    )
    args = argparse.Namespace(
        output=tmp_path / "results",
        config="highs",
        time_limit=30,
        memory_gb=2,
        license_slots=tmp_path / "slots",
    )
    return case, args


def test_isolated_worker_validates_original_edges_and_resumes(tmp_path):
    case, args = make_case(tmp_path)
    result = run_case(case, args, {"commit": "test"})
    assert result["status"] == "optimal" and result["objective"] == 2
    assert result["feasible"] and result["reference_optimal"]
    assert run_case(case, args, {"commit": "test"}) == result
    args.time_limit = 31
    with pytest.raises(ValueError, match="provenance changed"):
        run_case(case, args, {"commit": "test"})


def test_wall_timeout_is_recorded_without_a_solution(tmp_path):
    case, args = make_case(tmp_path)
    args.time_limit = 0.001
    result = run_case(case, args, {"commit": "test"})
    assert result["status"] == "timeout" and not result["feasible"]
    assert "objective" not in result
    stored = next(args.output.glob("*/result.json"))
    assert json.loads(stored.read_text())["status"] == "timeout"


@pytest.mark.skipif(os.name == "nt", reason="HPC uses POSIX locks")
def test_wls_reservations_cover_running_and_recently_finished_workers(tmp_path):
    import fcntl

    future = time.time() + 3600
    (tmp_path / "slot-0.lock").write_text(str(future))
    with wls_slot(tmp_path, 10):
        # Slot zero is free as an OS lock, but its token has not expired.
        path = tmp_path / "slot-1.lock"
        assert float(path.read_text()) >= time.time() + WLS_COOLDOWN
        with path.open("r+") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert float(path.read_text()) > time.time() + WLS_COOLDOWN - 2
    assert float((tmp_path / "slot-0.lock").read_text()) == future


@pytest.mark.skipif(os.name == "nt", reason="HPC uses POSIX locks")
def test_wls_waits_when_all_24_slots_are_reserved(tmp_path, monkeypatch):
    for index in range(24):
        (tmp_path / "slot-{}.lock".format(index)).write_text(str(time.time() + 3600))

    def waited(seconds):
        raise TimeoutError("all slots reserved")

    monkeypatch.setattr("benchmarks.pace2018.time.sleep", waited)
    with pytest.raises(TimeoutError, match="all slots"):
        with wls_slot(tmp_path, 10):
            pytest.fail("must not acquire a 25th token")


@pytest.mark.skipif(os.name == "nt", reason="HPC uses POSIX locks")
def test_preprocessing_only_result_does_not_occupy_a_wls_token(tmp_path):
    with wls_slot(tmp_path, 10) as lease:
        lease["used"] = False
    assert float((tmp_path / "slot-0.lock").read_text()) == 0
