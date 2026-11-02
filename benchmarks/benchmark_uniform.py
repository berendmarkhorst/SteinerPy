"""Paired uniform-cost formulation comparison in isolated worker processes.

Every returned solution is checked on the original graph. Timings compare
automatic selection with the node model disabled, on the same source checkout.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

CASES = ("cycle", "grid", "hypercube", "dense")
CONFIGS = ("auto", "directed")


def make_case(name):
    """Generate fixed uniform graphs with more than ten mandatory terminals."""
    import networkx as nx

    if name == "cycle":
        graph = nx.cycle_graph(60)
        terminals = list(range(0, 60, 5))
    elif name == "grid":
        graph = nx.grid_2d_graph(8, 8)
        terminals = [node for node in graph if all(x % 2 == 0 for x in node)]
    elif name == "hypercube":
        graph = nx.hypercube_graph(5)
        terminals = [node for node in graph if sum(node) % 2 == 0]
    elif name == "dense":
        graph = nx.complete_bipartite_graph(16, 32)
        terminals = list(range(16))
    else:
        raise ValueError("Unknown uniform benchmark case: " + name)
    nx.set_edge_attributes(graph, 1, "weight")
    return graph, terminals


def check_solution(graph, terminals, solution):
    """Reject invalid original-space trees, costs, and optimality gaps."""
    import networkx as nx

    edges = list(solution.edges)
    keys = {frozenset(edge) for edge in edges}
    if len(keys) != len(edges) or any(not graph.has_edge(*edge) for edge in edges):
        raise AssertionError("Solution contains duplicate or non-original edges")
    selected = nx.Graph()
    selected.add_nodes_from(terminals)
    selected.add_edges_from(edges)
    if not nx.is_tree(selected):
        raise AssertionError("Solution is disconnected or cyclic")
    cost = sum(graph[u][v].get("weight", 1) for u, v in edges)
    if not math.isfinite(solution.objective) or not math.isclose(
        cost, solution.objective, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise AssertionError("Reported objective differs from original edge cost")
    if math.isnan(solution.gap) or solution.gap < 0:
        raise AssertionError("Invalid optimality gap")


def check_pairs(rows):
    """Compare certified optima, retaining failed or incomplete measurements."""
    groups = {}
    for row in rows:
        key = (row["case"], row["preprocess"], row["repeat"])
        groups.setdefault(key, []).append(row)
    for key, pair in groups.items():
        optima = {row["objective"] for row in pair if row["status"] == "optimal"}
        if len(optima) > 1:
            raise AssertionError("Conflicting proven optima for {!r}".format(key))
        if optima:
            optimum = next(iter(optima))
            if any(
                row["objective"] < optimum - 1e-9
                for row in pair
                if row["status"] in {"optimal", "feasible"}
            ):
                raise AssertionError(
                    "Incumbent beats a proven optimum for {!r}".format(key)
                )


def worker(args):
    """Measure construction and solving; graph generation and imports precede timing."""
    import logging

    sys.path.insert(0, str(args.source_root.resolve()))
    import steinerpy.objects as objects
    import steinerpy.dreyfus_wagner as dw
    from benchmarks.benchmark_dispatch import peak_rss_mb

    logging.disable(logging.CRITICAL)
    graph, terminals = make_case(args.case)
    row = dict(
        nodes=len(graph), edges=graph.number_of_edges(), terminals=len(terminals)
    )
    paths = []

    def record_path(function, name):
        def call(*args, **kwargs):
            paths.append(name)
            return function(*args, **kwargs)

        return call

    started = None
    try:
        with ExitStack() as stack:
            for owner, attr, name in (
                (objects, "build_model", "directed_cut"),
                (objects, "build_model_gurobi", "directed_cut"),
                (dw, "dreyfus_wagner", "dreyfus_wagner"),
            ):
                stack.enter_context(
                    patch.object(owner, attr, record_path(getattr(owner, attr), name))
                )
            started = perf_counter()
            problem = objects.SteinerProblem(
                graph, [terminals], preprocess=args.preprocess == "on"
            )
            constructed = perf_counter()
            row.update(
                construction_seconds=constructed - started,
                reduced_nodes=problem.graph.number_of_nodes(),
                reduced_edges=problem.graph.number_of_edges(),
                reduced_terminals=len(set(problem.terminal_groups[0])),
            )
            try:
                solution = problem.get_solution(
                    time_limit=args.time_limit,
                    solver=args.solver,
                    threads=1,
                    decompose=False,
                )
            finally:
                finished = perf_counter()
                stats = problem.cut_stats
                row.update(
                    solve_seconds=finished - constructed,
                    total_seconds=finished - started,
                    formulation=stats.get("formulation")
                    or (paths[-1] if paths else "shortcut"),
                    cut_stats=stats,
                )
            check_solution(graph, terminals, solution)
            row.update(
                status="optimal" if solution.gap == 0 else "feasible",
                objective=solution.objective,
                gap=solution.gap if math.isfinite(solution.gap) else None,
                selected_edges=len(solution.edges),
                certificate_valid=True,
            )
    except Exception as error:
        unavailable = args.solver == "gurobi" and (
            isinstance(error, ImportError) or getattr(error, "errno", None) == 10009
        )
        incomplete = isinstance(
            error, RuntimeError
        ) and "stopped before finding" in str(error)
        row.update(
            status=(
                "unavailable"
                if unavailable
                else "incomplete" if incomplete else "error"
            ),
            error="{}: {}".format(type(error).__name__, error),
        )
        if started is not None:
            row.setdefault("total_seconds", perf_counter() - started)
    row["peak_rss_mb"] = peak_rss_mb()
    return row


def run_worker(args, case, preprocess, config, repeat):
    """Keep unsuccessful measurements as explicit rows in the output."""
    row = dict(case=case, preprocess=preprocess == "on", config=config, repeat=repeat)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        STEINERPY_THREADS="1",
        STEINERPY_DW_MAX_TERMINALS="10",
        STEINERPY_DECOMPOSE="0",
        STEINERPY_UNIFORM_NODE_MODEL="auto" if config == "auto" else "0",
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source-root",
        str(args.source_root.resolve()),
        "--case",
        case,
        "--config",
        config,
        "--preprocess",
        preprocess,
        "--solver",
        args.solver,
        "--time-limit",
        str(args.time_limit),
    ]
    try:
        completed = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=30, check=False
        )
        if completed.returncode:
            row.update(status="error", error=completed.stderr[-3000:])
        else:
            row.update(json.loads(completed.stdout.strip().splitlines()[-1]))
    except subprocess.TimeoutExpired:
        row.update(status="timeout", error="Worker exceeded 30 seconds")
    except (ValueError, IndexError) as error:
        row.update(status="error", error="Invalid worker output: " + str(error))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--solver", choices=("highs", "gurobi"), default="highs")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--preprocess", choices=("off", "on"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--config", choices=CONFIGS)
    args = parser.parse_args()
    if args.repeats < 1 or not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("repeats and time limit must be positive and finite")
    if args.worker:
        if args.case is None or args.config is None or args.preprocess is None:
            parser.error("worker requires case, config, and preprocessing")
        os.environ["STEINERPY_UNIFORM_NODE_MODEL"] = (
            "auto" if args.config == "auto" else "0"
        )
        print(json.dumps(worker(args), allow_nan=False))
        return
    if args.output is None:
        parser.error("require --output")
    packages = {}
    for package in ("networkx", "highspy", "scipy", "gurobipy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    rows = []
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        commit=subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        dirty=bool(
            subprocess.check_output(
                ["git", "-C", str(args.source_root), "status", "--porcelain"], text=True
            ).strip()
        ),
        platform=platform.platform(),
        cpu=platform.processor(),
        python=sys.version.split()[0],
        packages=packages,
        threads=1,
        solver=args.solver,
        repeats=args.repeats,
        time_limit=args.time_limit,
        worker_timeout=30,
        timing_scope="construction including preprocessing; solve including model build",
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for case in args.cases:
            for preprocess in (args.preprocess,) if args.preprocess else ("off", "on"):
                order = CONFIGS if repeat % 2 == 0 else tuple(reversed(CONFIGS))
                for config in order:
                    row = run_worker(args, case, preprocess, config, repeat)
                    rows.append(row)
                    print(
                        "{} {} {} repeat {}: {}".format(
                            case, preprocess, config, repeat + 1, row["status"]
                        ),
                        flush=True,
                    )
                    args.output.write_text(
                        json.dumps(report, indent=2, allow_nan=False) + "\n"
                    )
    check_pairs(rows)
    if any(row["status"] == "error" for row in rows):
        raise SystemExit("Benchmark recorded worker errors; inspect the output")


if __name__ == "__main__":
    main()
