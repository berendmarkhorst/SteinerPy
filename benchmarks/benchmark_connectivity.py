"""Paired general-model timings and separate LP bounds on identical inputs."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

# Import the current harness before selecting the production-code checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import profile_composition as profile


def worker(args):
    sys.path.insert(0, str(args.source_root.resolve()))
    if not args.lp:
        make_case = profile.make_case
        with patch.object(
            profile, "make_case", lambda name: make_case(name, args.seed)
        ):
            return profile.worker(args.case, args.config, args.time_limit)

    from steinerpy import Problem
    from steinerpy._problem_spec import normalize_problem
    from steinerpy._problem_validation import validate_problem
    from steinerpy._feature_composition import FeatureComposition
    from steinerpy._selection_model import SelectionModel

    graph, options = profile.make_case(args.case, args.seed)
    options["preprocess"] = args.config == "flow-filtered"
    spec = normalize_problem(
        graph, None, options.pop("terminals", None), None, options, Problem.variants
    )
    validate_problem(spec, None)
    strategy = FeatureComposition(spec)
    model = SelectionModel(spec, "highs", args.time_limit, "", 1)
    try:
        coverage = strategy.configure(model)
        # Diagnose the stronger relaxation separately from adaptive MIP timing.
        if hasattr(model, "strengthen_connectivity"):
            model.strengthen_connectivity()
        model.model.setOptionValue("solve_relaxation", True)
        started = perf_counter()
        model.model.minimize(
            -model.sum(coverage) if strategy.coverage_mode else model.cost
        )
        status = model.model.modelStatusToString(model.model.getModelStatus())
        return dict(
            status=status,
            # Always the minimization objective, including negated coverage/rewards.
            bound=model.model.getObjectiveValue() if status == "Optimal" else None,
            seconds=perf_counter() - started,
            columns=model.model.getNumCol(),
            rows=model.model.getNumRow(),
        )
    finally:
        model.close()


def check_results(rows):
    for seed in {r["seed"] for r in rows}:
        for config in {r["config"] for r in rows}:
            subset = [r for r in rows if r["seed"] == seed and r["config"] == config]
            profile.check_results([r for r in subset if not r["lp"]])
            for case in {r["case"] for r in subset}:
                bounds = {
                    r["label"]: r["bound"]
                    for r in subset
                    if r["case"] == case and r["lp"] and r["status"] == "Optimal"
                }
                if len(bounds) == 2:
                    assert bounds["candidate"] >= bounds["baseline"] - 1e-7, case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--compare-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=5)
    parser.add_argument(
        "--cases", nargs="+", choices=profile.CASES, default=profile.CASES
    )
    parser.add_argument(
        "--configs", nargs="+", choices=["flow", "flow-filtered"], default=["flow"]
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=profile.CASES)
    parser.add_argument("--config", choices=["flow", "flow-filtered"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lp", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("repeats and time limit must be positive and finite")
    if args.worker:
        if args.case is None or args.config is None or args.seed is None:
            parser.error("worker needs case, config and seed")
        print(json.dumps(worker(args), allow_nan=False))
        return
    if args.compare_root is None or args.output is None:
        parser.error("require --compare-root and --output")
    roots = {
        "baseline": args.compare_root.resolve(),
        "candidate": args.source_root.resolve(),
    }
    rows = []
    report = dict(
        commits={
            label: subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            for label, root in roots.items()
        },
        dirty={
            label: bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain"], text=True
                ).strip()
            )
            for label, root in roots.items()
        },
        platform=platform.platform(),
        cpu=platform.processor(),
        python=sys.version.split()[0],
        packages={p: version(p) for p in ("networkx", "highspy", "scipy")},
        seeds=args.seeds,
        repeats=args.repeats,
        time_limit=args.time_limit,
        threads=1,
        solver="highs",
        worker_timeout=max(30, args.time_limit + 15),
        rows=rows,
    )
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        STEINERPY_THREADS="1",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for case in args.cases:
            for config in args.configs:
                # LP runs are separate fresh processes, never inside MIP timing.
                for repeat in range(-1, args.repeats):
                    labels = list(roots) if repeat % 2 == 0 else list(reversed(roots))
                    for label in labels:
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--source-root",
                            str(roots[label]),
                            "--case",
                            case,
                            "--seed",
                            str(seed),
                            "--config",
                            config,
                            "--time-limit",
                            str(args.time_limit),
                        ]
                        if repeat == -1:
                            command.append("--lp")
                        try:
                            result = subprocess.run(
                                command,
                                env=env,
                                capture_output=True,
                                text=True,
                                timeout=report["worker_timeout"],
                            )
                            row = (
                                json.loads(result.stdout)
                                if not result.returncode
                                else dict(status="error", error=result.stderr)
                            )
                        except subprocess.TimeoutExpired:
                            row = dict(status="timeout")
                        rows.append(
                            dict(
                                row,
                                seed=seed,
                                case=case,
                                config=config,
                                repeat=repeat,
                                label=label,
                                lp=repeat == -1,
                            )
                        )
                        args.output.write_text(
                            json.dumps(report, indent=2, allow_nan=False) + "\n"
                        )
            print(f"Completed seed {seed}, {case}", flush=True)
    check_results(rows)
    print(dict(Counter(r["status"] for r in rows)), flush=True)
    if any(r["status"] in {"error", "timeout", "no_incumbent"} for r in rows):
        raise SystemExit("Incomplete workers retained; inspect results.")


if __name__ == "__main__":
    main()
