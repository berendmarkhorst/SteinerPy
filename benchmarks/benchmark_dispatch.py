"""Bounded before/after dispatch comparison; each measurement is a fresh process.

Use --compare-root to alternate old/new checkouts within each case and repeat.
Compare matching case objectives and gaps before timings. An engineering regression
threshold for unchanged paths is +10% or +2 ms in median construction+solve time,
whichever is larger. This small sample does not establish general performance.
"""

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

CASES = (
    "plain-tree",
    "plain-forest",
    "zero-tree",
    "zero-forest",
    "zero-unreduced",
    "paid-degree-hop",
    "groups-budget",
    "reward-forest",
)


def peak_rss_mb():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak",
                    "current",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        measure = ctypes.windll.psapi.GetProcessMemoryInfo
        measure.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not measure(get_process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return counters.peak / 1024**2
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == "darwin" else 1024
    )


def worker(args):
    import logging
    import random
    from time import perf_counter
    from importlib.metadata import version

    sys.path.insert(0, str(Path(args.source_root).resolve()))
    import networkx as nx
    from steinerpy import Problem

    logging.disable(logging.CRITICAL)
    graph = nx.cycle_graph(12)
    graph.add_edges_from([(0, 4), (1, 7), (2, 9), (3, 8), (5, 10), (6, 11)])
    rng = random.Random(42)
    for u, v in graph.edges:
        graph[u][v]["weight"] = rng.randint(1, 9)
    options = dict(terminals=[0, 4, 8], preprocess=True)
    if "forest" in args.case:
        options["terminals"] = [[0, 4], [7, 10]]
    if args.case.startswith("zero"):
        options["node_costs"] = {}
    if args.case == "zero-unreduced":
        options["preprocess"] = False
    if args.case == "paid-degree-hop":
        options.update(node_costs={3: 2}, max_degree=3, hop_limit=8)
    if args.case == "groups-budget":
        options = dict(
            groups=[[0, 1], [10, 11]],
            node_costs={3: 2},
            budget=12,
            max_degree=3,
            preprocess=True,
        )
    if args.case == "reward-forest":
        options.update(node_costs={3: 2}, node_prizes={6: 4}, hop_limit=8)
    started = perf_counter()
    problem = Problem(graph, **options)
    constructed = perf_counter()
    solution = problem.get_solution(time_limit=5, solver="highs", threads=1)
    finished = perf_counter()
    assert solution.gap == 0, "case did not prove optimality within its limit"
    assert all(graph.has_edge(*e) for e in solution.edges)
    return dict(
        case=args.case,
        objective=solution.objective,
        gap=solution.gap,
        implementation=type(problem.implementation).__name__,
        construction_seconds=constructed - started,
        solve_seconds=finished - constructed,
        total_seconds=finished - started,
        peak_rss_mb=peak_rss_mb(),
        python=sys.version.split()[0],
        packages={p: version(p) for p in ("networkx", "highspy", "scipy")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--compare-root", help="baseline checkout to interleave with source-root"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(worker(args)))
        return
    if not args.output or args.repeats < 1:
        parser.error("require --output and a positive --repeats")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        STEINERPY_THREADS="1",
        STEINERPY_DW_MAX_TERMINALS="10",
        STEINERPY_DECOMPOSE="0",
    )
    sources = {"candidate": args.source_root}
    if args.compare_root:
        sources["baseline"] = args.compare_root
    commits = {
        label: subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip()
        for label, root in sources.items()
    }
    rows = []
    for repeat in range(args.repeats):
        for case in CASES:
            order = list(sources)
            if repeat % 2:
                order.reverse()
            pair = []
            for label in order:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--source-root",
                    sources[label],
                    "--case",
                    case,
                ]
                result = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if result.returncode:
                    raise RuntimeError(case + ": " + result.stderr)
                row = dict(json.loads(result.stdout), repeat=repeat, label=label)
                pair.append(row)
                rows.append(row)
            assert len({(r["objective"], r["gap"]) for r in pair}) == 1, case
        print("Completed repeat {}/{}".format(repeat + 1, args.repeats), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                commits=commits,
                platform=platform.platform(),
                cpu=platform.processor(),
                threads=1,
                seed=42,
                solver="highs",
                time_limit=5,
                timing_scope="construction (including preprocessing) + solve (including model build)",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
