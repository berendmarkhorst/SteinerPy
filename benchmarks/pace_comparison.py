"""Fixed, bounded policy and normal-API comparisons for issue #104."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import sys

from benchmarks import connectivity_pilot as pilot, pace2018 as pace


def combined_cases():
    return [
        dict(id=f"{name}-{seed}", name=name, seed=seed, config="highs", kind="combined")
        for name, seeds in (
            ("signed-rewards", range(40, 46)),
            ("resource-tree", range(42, 46)),
            ("groups-forest", (42, 44)),
        )
        for seed in seeds
    ]


def pace_cases(dataset, solver_root):
    # Two public and two private instances spread through each archive; no
    # selection by solver outcome. This is a diagnostic sample, not a ranking.
    names = {f"instance{number:03}.gr" for number in (1, 50, 101, 150)}
    selected = pace.reference_cases(dataset, [1, 2, 3], instances=names)
    if {(r["track"], r["instance"]) for r in selected} != {
        (track, name) for track in (1, 2, 3) for name in names
    }:
        raise ValueError("The fixed PACE comparison requires all twelve inputs")
    return [
        dict(
            row,
            id=f"track{row['track']}-{Path(row['instance']).stem}",
            kind="pace",
            binary=str(
                solver_root
                / "target/release"
                / ("track2" if row["track"] == 2 else "track1")
            ),
        )
        for row in selected
    ]


def run_pace_pair(case, candidate, context, output):
    rows = []
    for repeat in range(2):
        methods = ("api-highs", "wata") if repeat == 0 else ("wata", "api-highs")
        for method in methods:
            name = f"{case['id']}-{repeat}-{method}"
            directory = output / name
            directory.mkdir()
            args = argparse.Namespace(
                config=method,
                time_limit=120,
                memory_gb=8,
                worker_module=(
                    "benchmarks.pace_wata"
                    if method == "wata"
                    else "benchmarks.connectivity_pilot"
                ),
            )
            identity = dict(
                context,
                commit=context["commits"]["candidate"],
                config=method,
                repeat=repeat,
                threads=1,
                time_limit=120,
                memory_gb=8,
            )
            rows.append(
                pace.execute_case(
                    dict(case, config=method, source_root=str(candidate)),
                    args,
                    identity,
                    directory,
                    directory / "result.json",
                    name,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "baseline", "candidate", "solver-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error(
            "Use Linux to enforce both solvers' memory and process-group limits"
        )
    roots = {
        label: getattr(args, label).resolve() for label in ("baseline", "candidate")
    }
    commits = {label: pilot.clean_revision(root) for label, root in roots.items()}
    if commits["baseline"] == commits["candidate"]:
        parser.error("baseline and candidate must differ")
    context = pace.provenance(args.dataset)
    import highspy

    solver_root = args.solver_root.resolve()
    context.update(
        commits=commits,
        harness_commit=pilot.clean_revision(Path(__file__).resolve().parents[1]),
        native_highs_version=highspy.Highs().version(),
        memory_limit_enforced=True,
        competitor_commit=pilot.clean_revision(solver_root),
        compiler=subprocess.check_output(["rustc", "--version"], text=True).strip(),
        binary_sha256={
            name: hashlib.sha256(
                (solver_root / "target/release" / name).read_bytes()
            ).hexdigest()
            for name in ("track1", "track2")
        },
    )
    combined = combined_cases()
    selected = pace_cases(args.dataset, solver_root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pace.write_json(
        output / "manifest.json",
        dict(
            run=context,
            combined_cases=combined,
            pace_cases=selected,
            workers=96,
            repeats=2,
            jobs=3,
            threads=1,
            memory_gb=8,
            combined_time_limit=15,
            pace_time_limit=120,
        ),
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        combined_rows = [
            row
            for batch in pool.map(
                lambda case: pilot.run_pair(case, roots, context, output, 15), combined
            )
            for row in batch
        ]
        pace.write_json(output / "combined.json", dict(run=context, rows=combined_rows))
        pilot.check_results(combined_rows)
        pace_rows = [
            row
            for batch in pool.map(
                lambda case: run_pace_pair(case, roots["candidate"], context, output),
                selected,
            )
            for row in batch
        ]
    pace.write_json(output / "pace.json", dict(run=context, rows=pace_rows))
    pilot.check_results(pace_rows)
    if any(
        r["status"] in {"error", "memory_limit", "worker_exit"}
        for r in combined_rows + pace_rows
    ):
        raise SystemExit("Worker failures retained; inspect results and worker logs")


if __name__ == "__main__":
    main()
