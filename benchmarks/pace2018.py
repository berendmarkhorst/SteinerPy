"""PACE 2018 sweeps with isolated workers, checked solutions and resumable JSON.

Run on compute nodes, e.g. python -m benchmarks.pace2018 --dataset ../pace2018
--output results --config highs --jobs 12. Every instance gets a fresh process;
the wall limit includes imports, parsing, preprocessing, model building and solving.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

CONFIGS = ("highs", "gurobi", "heuristic", "composition-highs", "composition-gurobi")
WLS_COOLDOWN = 310  # A closed environment's five-minute token can still be occupied.


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reference_cases(dataset, tracks):
    """Use the organizers' manifests, including the omission of Track 3 #058."""
    result = []
    for track in tracks:
        with (dataset / "track{}.csv".format(track)).open(newline="") as stream:
            for row in csv.DictReader(stream):
                name = row["paceName"].strip()
                path = dataset / "Track{}".format(track) / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                lower = float(row.get("opt", row.get("lower")))
                upper = float(row.get("opt", row.get("upper")))
                number = int(path.stem[len("instance") :])
                result.append(
                    dict(
                        track=track,
                        instance=name,
                        split="private" if number % 2 == 0 else "public",
                        path=str(path.resolve()),
                        reference_lower=lower,
                        reference_upper=upper,
                        instance_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    return result


def validate_solution(graph, terminals, edges, objective, gap, lower, upper):
    """Check the original graph, independently of the solver's internal reductions."""
    import networkx as nx

    if not math.isfinite(objective):
        raise ValueError("non-finite objective")
    tree = nx.Graph()
    tree.add_nodes_from(terminals)
    for u, v in edges:
        if not graph.has_edge(u, v) or tree.has_edge(u, v):
            raise ValueError("invalid or duplicated original edge")
        tree.add_edge(u, v)
    if not nx.is_tree(tree):
        raise ValueError("solution is not a tree spanning all terminals")
    actual = sum(graph[u][v].get("weight", 1) for u, v in tree.edges)
    if not math.isclose(actual, objective, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("reported objective differs from original edge costs")
    if objective < lower - 1e-6:
        raise ValueError("objective below published lower bound")
    reference_optimal = lower == upper and math.isclose(
        objective, upper, abs_tol=1e-6, rel_tol=0
    )
    if gap == 0 and objective > upper + 1e-6:
        raise ValueError(
            "solver claims an optimum worse than a published feasible bound"
        )
    return dict(
        feasible=True,
        reference_optimal=reference_optimal,
        solver_proven_optimal=gap == 0,
        reference_ratio=upper / objective if objective else 1.0,
        status="optimal" if reference_optimal or gap == 0 else "feasible",
    )


def worker(request_path):
    """One process, one instance, at most one Gurobi environment."""
    request = json.loads(Path(request_path).read_text())
    progress = Path(request["progress"])
    row = dict(request["case"])
    started = time.monotonic()
    if sys.platform == "linux":
        import resource

        bound = int(request["memory_gb"] * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (bound, bound))
    row.update(status="running", phase="imports", feasible=False)
    write_json(progress, row)
    try:
        import logging
        import random

        logging.disable(logging.CRITICAL)
        random.seed(0)
        from benchmarks.stp_parser import read_stp
        from steinerpy import Problem, SteinerProblem

        row["phase"] = "parse"
        write_json(progress, row)
        graph, groups = read_stp(row["path"])
        row.update(
            nodes=len(graph),
            edges=graph.number_of_edges(),
            terminals=len(groups[0]),
            parse_seconds=time.monotonic() - started,
            phase="preprocess",
        )
        write_json(progress, row)
        prep = time.monotonic()
        config = request["config"]
        if config.startswith("composition-"):
            problem = Problem(graph, terminals=groups, node_costs={}, preprocess=True)
        else:
            # Default classic solver, with one model per process. Decomposition and
            # DA fallback can create additional environments and are not used here.
            problem = SteinerProblem(
                graph, groups, preprocess=True, dual_ascent=False, decompose=False
            )
        row.update(preprocess_seconds=time.monotonic() - prep, phase="solve")
        if hasattr(problem, "graph"):
            row.update(
                reduced_nodes=len(problem.graph),
                reduced_edges=problem.graph.number_of_edges(),
            )
        write_json(progress, row)
        remaining = max(0, request["time_limit"] - (time.monotonic() - started) - 2)
        solve = time.monotonic()
        kwargs = dict(
            time_limit=remaining,
            threads=1,
            solver="gurobi" if config.endswith("gurobi") else "highs",
        )
        if config == "heuristic":
            kwargs["exact"] = False
        solution = problem.get_solution(**kwargs)
        row.update(
            solve_seconds=time.monotonic() - solve,
            solver_seconds=solution.runtime,
            objective=float(solution.objective),
            gap=float(solution.gap) if math.isfinite(solution.gap) else None,
            phase="validate",
        )
        edges = list(solution.original_selected_edges)
        row.update(
            validate_solution(
                graph,
                groups[0],
                edges,
                solution.objective,
                solution.gap,
                row["reference_lower"],
                row["reference_upper"],
            )
        )
        row["solution_edges"] = edges
        row["phase"] = "complete"
    except MemoryError:
        row.update(status="memory_limit", error="MemoryError")
    except Exception as error:
        row.update(status="error", error="{}: {}".format(type(error).__name__, error))
    row["worker_seconds"] = time.monotonic() - started
    if sys.platform != "win32":
        import resource

        row["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024 if sys.platform != "darwin" else 1024**2
        )
    write_json(progress, row)


@contextmanager
def wls_slot(directory, time_limit):
    """Cap this account's benchmark sessions across arrays, including token expiry.

    The reservation survives a killed supervisor. Each classic worker disables
    decomposition and DA retries, so it creates at most one Gurobi environment.
    Other applications using the same license must leave room for these 24 slots.
    """
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(24):
            stream = (directory / "slot-{}.lock".format(index)).open("a+")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            stream.seek(0)
            available = float(stream.read() or 0)
            if available > time.time():
                stream.close()
                continue

            def reserve(until):
                stream.seek(0)
                stream.truncate()
                stream.write(str(until))
                stream.flush()
                os.fsync(stream.fileno())

            reserve(time.time() + time_limit + WLS_COOLDOWN + 60)
            try:
                yield
            finally:
                reserve(time.time() + WLS_COOLDOWN)
                stream.close()
            return
        time.sleep(1)


def comparable_run(run):
    return {
        key: value
        for key, value in run.items()
        if key not in {"hostname", "slurm_job", "slurm_array_task"}
    }


def run_case(case, args, provenance):
    name = "track{}-{}-{}".format(
        case["track"], Path(case["instance"]).stem, args.config
    )
    directory = args.output / name
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    identity = dict(
        provenance,
        config=args.config,
        time_limit=args.time_limit,
        memory_gb=args.memory_gb,
        threads=1,
        seed=0,
        instance_sha256=case["instance_sha256"],
    )
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if comparable_run(previous["run"]) != comparable_run(identity):
            raise ValueError(
                "Result provenance changed; choose a new output directory: " + name
            )
        return previous
    if args.config.endswith("gurobi"):
        if os.name == "nt":
            raise RuntimeError(
                "The WLS concurrency guard requires Linux; run Gurobi sweeps on the HPC."
            )
        with wls_slot(args.license_slots, args.time_limit):
            return execute_case(case, args, identity, directory, result_path, name)
    return execute_case(case, args, identity, directory, result_path, name)


def execute_case(case, args, identity, directory, result_path, name):
    progress = directory / "progress.json"
    if progress.exists():
        progress.unlink()
    request = dict(
        case=case,
        config=args.config,
        time_limit=args.time_limit,
        memory_gb=args.memory_gb,
        progress=str(progress.resolve()),
    )
    request_path = directory / "request.json"
    write_json(request_path, request)
    # Set WLS token expiry and deterministic MIP settings before Env.start().
    (directory / "gurobi.env").write_text(
        "WLSTokenDuration 5\nSeed 0\nThreads 1\nOutputFlag 0\n"
    )
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        STEINERPY_THREADS="1",
    )
    command = [
        sys.executable,
        "-m",
        "benchmarks.pace2018",
        "--worker",
        str(request_path.resolve()),
    ]
    started = time.monotonic()
    peak = 0.0
    timed_out = False
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=os.name != "nt",
        )
        while process.poll() is None:
            if sys.platform == "linux":
                try:
                    status = Path("/proc/{}/status".format(process.pid)).read_text()
                    peak = max(
                        peak,
                        next(
                            float(line.split()[1]) / 1024
                            for line in status.splitlines()
                            if line.startswith("VmHWM:")
                        ),
                    )
                except (FileNotFoundError, ProcessLookupError, StopIteration):
                    pass
            if time.monotonic() - started >= args.time_limit:
                timed_out = True
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                break
            time.sleep(0.1)
    elapsed = time.monotonic() - started
    row = (
        json.loads(progress.read_text())
        if progress.exists()
        else dict(case, phase="startup", feasible=False)
    )
    if timed_out:
        row.update(status="timeout", feasible=False)
        for key in (
            "objective",
            "gap",
            "reference_optimal",
            "solver_proven_optimal",
            "reference_ratio",
            "solution_edges",
        ):
            row.pop(key, None)
    elif process.returncode:
        row.update(status="worker_exit", returncode=process.returncode, feasible=False)
    row.update(
        run=identity,
        wall_seconds=elapsed,
        peak_rss_mb=max(peak, row.get("peak_rss_mb", 0)),
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(result_path, row)
    print("{}: {} ({:.2f}s)".format(name, row["status"], elapsed), flush=True)
    return row


def provenance(dataset):
    def revision(path):
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()

    packages = {}
    for package in ("networkx", "numpy", "scipy", "highspy", "gurobipy"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    cpu = platform.processor()
    if sys.platform == "linux":
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return dict(
        commit=revision(Path(__file__).resolve().parents[1]),
        dataset_commit=revision(dataset),
        python=platform.python_version(),
        packages=packages,
        cpu=cpu,
        platform=platform.platform(),
        hostname=platform.node(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", choices=CONFIGS, default="highs")
    parser.add_argument("--tracks", default="1,2,3")
    parser.add_argument("--time-limit", type=float, default=1800)
    parser.add_argument("--memory-gb", type=float, default=8)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--license-slots", type=Path, default=Path.home() / ".cache" / "steinerpy-wls"
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--first", type=int, help="smoke test: first N cases per track")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.worker)
        return
    if args.dataset is None or args.output is None:
        parser.error("--dataset and --output are required")
    if not 1 <= args.jobs <= 24 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require 1..24 jobs and 0 <= shard-index < shard-count")
    if (
        not math.isfinite(args.time_limit)
        or args.time_limit <= 0
        or not math.isfinite(args.memory_gb)
        or args.memory_gb <= 0
    ):
        parser.error("time and memory limits must be finite and positive")
    tracks = [int(t) for t in args.tracks.split(",")]
    if not tracks or len(set(tracks)) != len(tracks) or not set(tracks) <= {1, 2, 3}:
        parser.error("tracks must be distinct values from 1,2,3")
    cases = reference_cases(args.dataset, tracks)
    if args.first is not None:
        if args.first < 1:
            parser.error("--first must be positive")
        cases = [
            case
            for track in tracks
            for case in [c for c in cases if c["track"] == track][: args.first]
        ]
    cases = cases[args.shard_index :: args.shard_count]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    context = provenance(args.dataset)
    write_json(
        args.output / ("manifest-{}-{}.json".format(args.config, args.shard_index)),
        dict(
            run=context,
            cases=cases,
            config=args.config,
            time_limit=args.time_limit,
            memory_gb=args.memory_gb,
        ),
    )
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        list(executor.map(lambda case: run_case(case, args, context), cases))


if __name__ == "__main__":
    main()
