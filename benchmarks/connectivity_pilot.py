"""A bounded before/after HiGHS pilot, locally or on Snellius."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import time

# Keep the same harness for both production-code checkouts.
from benchmarks import pace2018 as pace, profile_composition as profile


def cases(dataset):
    references = {
        (row["track"], row["instance"]): row
        for row in pace.reference_cases(
            dataset, [1, 2, 3], instances={"instance001.gr", "instance002.gr"}
        )
    }
    result = []
    for track in (1, 2, 3):
        for number in (1, 2):
            result.append(
                dict(
                    references[track, f"instance{number:03}.gr"],
                    id=f"general-track{track}-{number:03}",
                    config="composition-highs",
                    kind="pace",
                )
            )
    for name, seed in (
        ("resource-tree", 42),
        ("resource-tree", 43),
        ("groups-forest", 44),
        ("signed-rewards", 44),
    ):
        result.append(
            dict(
                id=f"{name}-{seed}",
                name=name,
                seed=seed,
                config="highs",
                kind="combined",
            )
        )
    for track in (1, 2):
        result.append(
            dict(
                references[track, "instance001.gr"],
                id=f"classic-track{track}-001",
                config="highs",
                kind="pace",
            )
        )
    return result


def worker(request_path):
    request = json.loads(request_path.read_text())
    case = request["case"]
    source = Path(case["source_root"]).resolve()
    if not (source / "steinerpy" / "__init__.py").is_file():
        raise ValueError("Missing production-code checkout")
    sys.path.insert(0, str(source))
    # Fail rather than silently measure an installed or the wrong checkout.
    if "steinerpy" in sys.modules:
        raise RuntimeError("Production code was imported before checkout selection")
    if request["config"] not in {"highs", "composition-highs"}:
        raise ValueError("This pilot only runs HiGHS")
    if case["kind"] == "pace":
        pace.worker(request_path)
        return

    started = time.monotonic()
    if sys.platform == "linux":
        import resource

        bound = int(request["memory_gb"] * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (bound, bound))
    progress = Path(request["progress"])
    row = dict(case, status="running", phase="imports", feasible=False)
    pace.write_json(progress, row)
    logging.disable(logging.CRITICAL)
    timer = profile.PhaseTimer()
    try:
        import steinerpy
        from steinerpy import Problem
        from steinerpy._feature_composition import FeatureComposition

        Path(steinerpy.__file__).resolve().relative_to(source)
        graph, options = profile.make_case(case["name"], case["seed"])
        options["preprocess"] = False
        inputs = dict(edges=list(graph.edges(data=True)), options=options)
        row.update(
            input_sha256=hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
            nodes=len(graph),
            edges=len(graph.edges),
            phase="solve",
        )
        pace.write_json(progress, row)
        with timer.instrument(), timer.measure("other"):
            problem = Problem(graph, **options)
            if not isinstance(problem.implementation, FeatureComposition):
                raise ValueError("Expected the public API to select the general model")
            remaining = max(0, request["time_limit"] - (time.monotonic() - started) - 2)
            solution = problem.get_solution(
                time_limit=remaining, solver="highs", threads=1
            )
        row.update(
            phase="validate",
            implementation=type(problem.implementation).__name__,
            solution_edges=list(solution.edges),
            selected_nodes=list(solution.selected_nodes),
        )
        pace.write_json(progress, row)
        checker = FeatureComposition(problem._spec)
        checked = checker.solution(
            checker.decode(solution.edges, list(solution.selected_nodes)),
            solution.gap,
            solution.runtime,
        )
        if not math.isfinite(solution.objective) or not math.isclose(
            checked.objective, solution.objective, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("Objective differs from the checked original selection")
        row.update(
            status="optimal" if solution.gap == 0 else "feasible",
            phase="complete",
            feasible=True,
            solver_proven_optimal=solution.gap == 0,
            objective=solution.objective,
            total_cost=checked.total_cost,
            gap=solution.gap if math.isfinite(solution.gap) else None,
            objective_sense="max" if checker.objective.report_sign < 0 else "min",
            lexicographic=checker.coverage_mode,
        )
    except MemoryError:
        row.update(status="memory_limit", error="MemoryError")
    except Exception as error:
        status = "error"
        if type(error).__name__ == "NoIncumbent":
            status = "infeasible" if error.status == "infeasible" else "no_incumbent"
        row.update(status=status, error=f"{type(error).__name__}: {error}")
    row.update(
        worker_seconds=time.monotonic() - started,
        phases_seconds=dict(timer.seconds),
        backend_runs=timer.backend_runs,
        peak_rss_mb=profile.peak_rss_mb(),
    )
    pace.write_json(progress, row)


def clean_revision(root):
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(f"Use a clean checkout: {root}")
    return revision


def run_pair(case, roots, context, output, time_limit=120):
    rows = []
    for repeat in range(2):
        labels = list(roots) if repeat == 0 else list(reversed(roots))
        for label in labels:
            name = f"{case['id']}-{repeat}-{label}"
            directory = output / name
            directory.mkdir()
            args = argparse.Namespace(
                config=case["config"],
                time_limit=time_limit,
                memory_gb=8,
                worker_module="benchmarks.connectivity_pilot",
            )
            identity = dict(
                context,
                label=label,
                commit=context["commits"][label],
                repeat=repeat,
                threads=1,
                time_limit=time_limit,
                memory_gb=8,
                config=case["config"],
            )
            rows.append(
                pace.execute_case(
                    dict(case, source_root=str(roots[label])),
                    args,
                    identity,
                    directory,
                    directory / "result.json",
                    name,
                )
            )
    return rows


def check_results(rows):
    for case_id in {row["id"] for row in rows}:
        hashes = {
            row["input_sha256"]
            for row in rows
            if row["id"] == case_id and "input_sha256" in row
        }
        if len(hashes) > 1:
            raise ValueError(
                f"Input changed between versions or repetitions: {case_id}"
            )
    profile.check_results(
        [
            dict(
                row,
                case=row["id"],
                lexicographic=row.get("lexicographic", False),
                objective_sense=row.get("objective_sense", "min"),
            )
            for row in rows
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120,
        help="Whole-worker seconds (at most 120); use 15 for a short local pilot",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not math.isfinite(args.time_limit) or not 0 < args.time_limit <= 120:
        parser.error("--time-limit must be finite, positive and at most 120 seconds")
    if any(
        getattr(args, name) is None
        for name in ("dataset", "baseline", "candidate", "output")
    ):
        parser.error("require --dataset, --baseline, --candidate and --output")
    roots = dict(baseline=args.baseline.resolve(), candidate=args.candidate.resolve())
    commits = {label: clean_revision(root) for label, root in roots.items()}
    if commits["baseline"] == commits["candidate"]:
        parser.error("baseline and candidate must differ")
    context = pace.provenance(args.dataset)
    import highspy

    context.update(
        commits=commits,
        harness_commit=clean_revision(Path(__file__).resolve().parents[1]),
        native_highs_version=highspy.Highs().version(),
        memory_limit_enforced=sys.platform == "linux",
    )
    selected = cases(args.dataset)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    pace.write_json(
        args.output / "manifest.json",
        dict(
            run=context,
            cases=selected,
            jobs=args.jobs,
            repeats=2,
            workers=48,
            threads=1,
            time_limit=args.time_limit,
            memory_gb=8,
            solver="highs",
        ),
    )
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        batches = pool.map(
            lambda case: run_pair(case, roots, context, args.output, args.time_limit),
            selected,
        )
        rows = [row for batch in batches for row in batch]
    pace.write_json(args.output / "results.json", dict(run=context, rows=rows))
    check_results(rows)
    if any(row["status"] in {"error", "worker_exit", "memory_limit"} for row in rows):
        raise SystemExit(
            "Worker failures retained; inspect results.json and worker logs"
        )


if __name__ == "__main__":
    main()
