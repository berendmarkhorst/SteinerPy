"""Reproducible Dreyfus-Wagner runtime and peak-RSS benchmark.

Each repetition runs in a fresh subprocess so ``ru_maxrss`` is isolated. Use
the same interpreter with two source checkouts to compare implementations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_K = (6, 7, 8, 9, 10)
DEFAULT_SIZES = (300, 1200)
DEFAULT_DENSITIES = (3, 8)


def _connected_graph(nx, n: int, edge_factor: int, k: int, seed: int):
    """Deterministic connected sparse graph with exactly about factor*n edges."""
    import random

    rng = random.Random(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    order = list(range(n))
    rng.shuffle(order)
    for i in range(1, n):
        parent = order[rng.randrange(i)]
        graph.add_edge(order[i], parent, weight=rng.randint(1, 20))

    target = min(n * (n - 1) // 2, edge_factor * n)
    while graph.number_of_edges() < target:
        u, v = rng.sample(range(n), 2)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v, weight=rng.randint(1, 20))
    terminals = rng.sample(range(n), k)
    return graph, terminals


def _worker(args) -> None:
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    import networkx as nx
    import scipy

    from steinerpy.dreyfus_wagner import dreyfus_wagner

    graph, terminals = _connected_graph(
        nx, args.nodes, args.edge_factor, args.k, args.seed
    )
    started = time.perf_counter()
    objective, edges = dreyfus_wagner(graph, terminals)
    runtime = time.perf_counter() - started

    selected = nx.Graph()
    selected.add_nodes_from(terminals)
    selected.add_edges_from(edges)
    feasible = len(terminals) <= 1 or all(
        nx.has_path(selected, terminals[0], terminal) for terminal in terminals[1:]
    )
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = raw_rss if sys.platform == "darwin" else raw_rss * 1024
    print(
        json.dumps(
            {
                "runtime_s": runtime,
                "peak_rss_mib": rss_bytes / (1024 * 1024),
                "objective": objective,
                "edge_count": len(edges),
                "feasible": feasible,
                "python": sys.version.split()[0],
                "networkx": nx.__version__,
                "scipy": scipy.__version__,
            }
        )
    )


def _parse_ints(value: str):
    return tuple(int(item) for item in value.split(",") if item)


def _coordinator(args) -> None:
    rows = []
    script = str(Path(__file__).resolve())
    for k in args.k:
        for n in args.sizes:
            for factor in args.densities:
                runtimes = []
                peaks = []
                reference = None
                for _repetition in range(args.repeats):
                    seed = args.seed + 100_000 * k + 1_000 * n + factor
                    command = [
                        sys.executable,
                        script,
                        "--worker",
                        "--source-root",
                        args.source_root,
                        "--k",
                        str(k),
                        "--nodes",
                        str(n),
                        "--edge-factor",
                        str(factor),
                        "--seed",
                        str(seed),
                    ]
                    completed = subprocess.run(
                        command,
                        check=True,
                        text=True,
                        capture_output=True,
                        env={**os.environ, "PYTHONHASHSEED": "0"},
                    )
                    result = json.loads(completed.stdout.strip().splitlines()[-1])
                    if not result["feasible"]:
                        raise RuntimeError(
                            f"infeasible reconstruction for k={k}, n={n}, "
                            f"factor={factor}"
                        )
                    signature = (result["objective"], result["edge_count"])
                    if reference is None:
                        reference = signature
                    elif signature != reference:
                        raise RuntimeError(
                            f"non-deterministic result for k={k}, n={n}, "
                            f"factor={factor}: {signature} != {reference}"
                        )
                    runtimes.append(result["runtime_s"])
                    peaks.append(result["peak_rss_mib"])

                assert reference is not None
                rows.append(
                    {
                        "label": args.label,
                        "k": k,
                        "nodes": n,
                        "edge_factor": factor,
                        "edges": min(n * (n - 1) // 2, factor * n),
                        "seed": seed,
                        "repeats": args.repeats,
                        "median_runtime_s": statistics.median(runtimes),
                        "median_peak_rss_mib": statistics.median(peaks),
                        "objective": reference[0],
                        "python": result["python"],
                        "networkx": result["networkx"],
                        "scipy": result["scipy"],
                    }
                )

    fieldnames = list(rows[0])
    destination = Path(args.output) if args.output else None
    handle = destination.open("w", newline="") if destination else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if destination:
            handle.close()


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--label", default="current")
    parser.add_argument("--output")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=_parse_ints, default=DEFAULT_K)
    parser.add_argument("--sizes", type=_parse_ints, default=DEFAULT_SIZES)
    parser.add_argument("--densities", type=_parse_ints, default=DEFAULT_DENSITIES)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--edge-factor", type=int)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    if args.worker:
        if isinstance(args.k, tuple):
            if len(args.k) != 1:
                raise SystemExit("--worker requires one --k")
            args.k = args.k[0]
        _worker(args)
    else:
        _coordinator(args)
