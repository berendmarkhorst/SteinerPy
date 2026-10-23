"""Paired LP-cut diagnostics for issue #85, with a hard timeout per worker."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter

CASES = ("issue85", "positive", "feasible", "classic", "forest", "sap")


def worker(args):
    sys.path.insert(0, str(args.source_root.resolve()))
    if args.highs_path:
        sys.path.insert(0, str(args.highs_path.resolve()))
    import highspy as hp
    import networkx as nx
    from steinerpy import Problem, SteinerProblem
    from steinerpy import mathematical_model as mm

    Path(mm.__file__).resolve().relative_to(args.source_root.resolve())
    logging.disable(logging.CRITICAL)
    native_runs = []
    original = hp.Highs.minimize

    def minimize(model, *positional, **kwargs):
        integer = any(
            t != hp.HighsVarType.kContinuous for t in model.getLp().integrality_
        )
        started = perf_counter()
        result = original(model, *positional, **kwargs)
        native_runs.append(
            dict(
                phase="mip" if integer else "lp",
                seconds=perf_counter() - started,
                parallel=model.getOptionValue("parallel")[1],
                threads=model.getOptionValue("threads")[1],
                status=model.modelStatusToString(model.getModelStatus()),
            )
        )
        return result

    hp.Highs.minimize = minimize
    if args.case in {"issue85", "positive", "feasible"}:
        graph = nx.cycle_graph(4)
        graph.add_edge(0, 2)
        nx.set_edge_attributes(
            graph,
            {e: i % 4 + (args.case == "positive") for i, e in enumerate(graph.edges)},
            "weight",
        )
        problem = Problem(
            graph,
            groups=[[0], [2], [3]],
            node_costs={n: (n * 3) % 7 for n in graph},
            max_degree=2 if args.case == "feasible" else 1,
        )
        demands = [[0, 2, 3]]
    else:
        graph = nx.gnm_random_graph(40, 100, seed=42)
        graph.add_edges_from(nx.cycle_graph(40).edges)
        rng = random.Random(42)
        nx.set_edge_attributes(
            graph, {e: rng.randint(1, 9) for e in graph.edges}, "weight"
        )
        if args.case == "sap":
            graph = graph.to_directed()
        demands = (
            [[0, 5, 10, 15], [20, 25, 30, 35]]
            if args.case == "forest"
            else [list(range(0, 40, 5))]
        )
        problem = SteinerProblem(graph, demands, preprocess=False, dual_ascent=False)
    row = dict(
        case=args.case, highspy=hp.Highs().version(), networkx=version("networkx")
    )
    started = perf_counter()
    try:
        if args.case == "sap":
            gap, _, objective, edges, status = mm.solve_sap_highs(
                problem, time_limit=3, threads=0, return_status=True
            )
            assert status == "optimal"
        else:
            options = dict(time_limit=3, threads=0)
            if not isinstance(problem, Problem):
                options["decompose"] = False
            solution = problem.get_solution(**options)
            gap, objective, edges = solution.gap, solution.objective, solution.edges
        solved = perf_counter()
        selected = nx.DiGraph() if graph.is_directed() else nx.Graph()
        selected.add_edges_from(edges)
        assert all(graph.has_edge(*edge) for edge in edges)
        assert all(
            nx.has_path(selected, group[0], n) for group in demands for n in group
        )
        cost = sum(graph[u][v]["weight"] for u, v in edges)
        if args.case == "feasible":
            assert max(dict(selected.degree()).values()) <= 2
            cost += sum((n * 3) % 7 for n in selected)
            assert cost == 9
        assert cost == objective
        row.update(status="optimal" if gap == 0 else "feasible", objective=objective)
    except RuntimeError as error:
        solved = perf_counter()
        if "instance is infeasible" not in str(
            error
        ) and "connectivity-valid feasible incumbent" not in str(error):
            raise
        row.update(
            status=(
                "infeasible" if "instance is infeasible" in str(error) else "incomplete"
            ),
            error=str(error),
        )
    row.update(seconds=solved - started, native_runs=native_runs)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--compare-root", type=Path)
    parser.add_argument("--highs-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.case is None:
            parser.error("worker requires --case")
        print(json.dumps(worker(args), allow_nan=False))
        return
    if args.compare_root is None or args.output is None or args.repeats < 1:
        parser.error("require --compare-root, --output and positive repeats")
    roots = dict(
        baseline=args.compare_root.resolve(), candidate=args.source_root.resolve()
    )
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
        python=platform.python_version(),
        logical_cpus=os.cpu_count(),
        threads=0,
        time_limit=3,
        worker_timeout=12,
        repeats=args.repeats,
        rows=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        for repeat in range(args.repeats):
            labels = list(roots) if repeat % 2 == 0 else list(reversed(roots))
            for label in labels:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--case",
                    case,
                    "--source-root",
                    str(roots[label]),
                ]
                if args.highs_path:
                    command.extend(["--highs-path", str(args.highs_path.resolve())])
                env = dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    STEINERPY_DW_MAX_TERMINALS="0",
                    PYTHONHASHSEED="0",
                )
                try:
                    result = subprocess.run(
                        command, capture_output=True, text=True, timeout=12, env=env
                    )
                    row = (
                        json.loads(result.stdout)
                        if result.returncode == 0
                        else dict(status="error", error=result.stderr)
                    )
                except subprocess.TimeoutExpired:
                    row = dict(status="worker_timeout")
                row.update(case=case, label=label, repeat=repeat)
                report["rows"].append(row)
                args.output.write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
                print(
                    case,
                    label,
                    row["status"],
                    round(row.get("seconds", 0), 3),
                    flush=True,
                )
    for case in CASES:
        rows = [r for r in report["rows"] if r["case"] == case]
        optima = {r["objective"] for r in rows if r["status"] == "optimal"}
        assert len(optima) <= 1, case
        if optima:
            optimum = next(iter(optima))
            assert all(
                r["objective"] >= optimum for r in rows if r["status"] == "feasible"
            ), case
            assert all(r["status"] != "infeasible" for r in rows), case
    print(dict(Counter(r["status"] for r in report["rows"])))
    if any(r["status"] in {"error", "worker_timeout"} for r in report["rows"]):
        raise SystemExit("Worker failures retained; inspect the output")


if __name__ == "__main__":
    main()
