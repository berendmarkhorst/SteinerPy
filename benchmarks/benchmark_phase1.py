"""Reproducible Phase-1 research benchmarks against an arbitrary checkout.

Each configuration/repetition runs in a fresh Python process so peak RSS is
isolated. Run the same command once with ``--source-root`` pointing at a clean
checkout of current main and once at the candidate branch, appending both to
the same CSV. The report rejects objective/certificate mismatches before
printing median runtimes.

Examples::

    python benchmarks/benchmark_phase1.py --feature pc \
      --source-root /tmp/steinerpy-main --label main --configs none,pcd \
      --output /tmp/phase1-pc.csv
    python benchmarks/benchmark_phase1.py --feature pc \
      --source-root . --label candidate \
      --configs none,pcd,pcd+trd,pcd+trd+nodes \
      --output /tmp/phase1-pc.csv --append
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys


FIELDS = [
    "label",
    "commit",
    "feature",
    "instance",
    "seed",
    "repeat",
    "config",
    "python",
    "highspy",
    "networkx",
    "threads",
    "objective",
    "gap",
    "preprocess_time",
    "heuristic_time",
    "lp_resolve_time",
    "mip_resolve_time",
    "solve_time",
    "total_time",
    "nodes_removed",
    "edges_removed",
    "active_rows",
    "peak_rows",
    "cuts_purged",
    "cuts_reintroduced",
    "separation_rounds",
    "peak_rss_mb",
    "solved_in_preprocessing",
    "status",
]

DEFAULT_CONFIGS = {
    "pc": "none,pcd,pcd+trd,pcd+trd+nodes",
    "primal": "baseline,local,implied,portfolio",
    "cut": "off,age3,age5,age10",
}


def _synthetic_graph(feature, seed):
    import random

    import networkx as nx

    rng = random.Random(seed)
    if feature == "pc":
        n, m = 18, 42
    elif feature == "primal":
        n, m = 36, 85
    else:
        n, m = 28, 68
    attempt = seed
    while True:
        graph = nx.gnm_random_graph(n, m, seed=attempt)
        if nx.is_connected(graph):
            break
        attempt += 10007
    for u, v in graph.edges():
        graph[u][v]["weight"] = rng.randint(1, 20)
    return graph, rng


def _peak_rss_mb():
    import resource

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return peak / divisor


def _worker(args):
    source_root = os.path.abspath(args.source_root)
    sys.path.insert(0, source_root)

    from importlib import metadata

    import networkx as nx

    from steinerpy import PrizeCollectingProblem, SteinerProblem

    graph, rng = _synthetic_graph(args.feature, args.seed)
    row = {field: "" for field in FIELDS}
    row.update(
        feature=args.feature,
        instance="synthetic-{:02d}".format(args.seed),
        seed=args.seed,
        repeat=args.repeat,
        config=args.config,
        python="{}.{}.{}".format(*sys.version_info[:3]),
        highspy=metadata.version("highspy"),
        networkx=nx.__version__,
        threads=args.threads,
        status="ok",
    )

    import logging
    import time

    logging.disable(logging.CRITICAL)
    started = time.perf_counter()

    if args.feature == "pc":
        prizes = {
            node: (rng.randint(3, 30) if rng.random() < 0.6 else 0)
            for node in graph.nodes()
        }
        if not any(prizes.values()):
            prizes[0] = 1
        anchor = next(node for node, prize in prizes.items() if prize > 0)
        level = {
            "none": False,
            "pcd": True,
            "pcd+trd": "pcd+trd",
            "pcd+trd+nodes": "pcd+trd+nodes",
        }[args.config]
        problem = PrizeCollectingProblem(
            graph,
            [[anchor]],
            prizes,
            penalty_cost=0,
            preprocess=False,
            pc_transform=True,
            pc_reduce=level,
        )
        solution = problem.get_solution(
            time_limit=args.time_limit, threads=args.threads
        )
        stats = getattr(problem, "pc_reduction_stats", {})
        row.update(
            preprocess_time=stats.get("preprocessing_time", 0.0),
            nodes_removed=stats.get("nodes_removed", 0),
            edges_removed=stats.get("edges_removed", 0),
            solved_in_preprocessing=stats.get("solved_in_preprocessing", False),
        )
    elif args.feature == "primal":
        terminals = rng.sample(sorted(graph.nodes()), 12)
        flags = {
            "baseline": (False, False),
            "local": (True, False),
            "implied": (False, True),
            "portfolio": (True, True),
        }[args.config]
        problem = SteinerProblem(
            graph,
            [terminals],
            preprocess=False,
            dual_ascent=True,
            primal_local_search=flags[0],
            implied_profit=flags[1],
        )
        solution = problem.get_solution(
            time_limit=args.time_limit, threads=args.threads
        )
        stats = getattr(problem, "heuristic_stats", {})
        row["heuristic_time"] = stats.get("runtime", 0.0)
    else:
        terminals = rng.sample(sorted(graph.nodes()), 8)
        groups = [terminals[:4], terminals[4:]]
        age = {"off": 0, "age3": 3, "age5": 5, "age10": 10}[args.config]
        os.environ["STEINERPY_CUT_PURGE_AGE"] = str(age)
        os.environ["STEINERPY_DW_MAX_TERMINALS"] = "0"
        problem = SteinerProblem(graph, groups, preprocess=False)
        solution = problem.get_solution(
            time_limit=args.time_limit, threads=args.threads
        )

    cut_stats = getattr(problem, "cut_stats", {})
    row.update(
        objective=solution.objective,
        gap=solution.gap,
        lp_resolve_time=cut_stats.get("lp_resolve_time", 0.0),
        mip_resolve_time=cut_stats.get("mip_resolve_time", 0.0),
        solve_time=solution.runtime,
        total_time=time.perf_counter() - started,
        active_rows=cut_stats.get("active_model_rows", ""),
        peak_rows=cut_stats.get("peak_model_rows", ""),
        cuts_purged=cut_stats.get("cuts_purged", 0),
        cuts_reintroduced=cut_stats.get("cuts_reintroduced", 0),
        separation_rounds=cut_stats.get("separation_rounds", 0),
        peak_rss_mb=_peak_rss_mb(),
    )
    print(json.dumps(row, sort_keys=True))


def _commit(source_root):
    try:
        return subprocess.check_output(
            ["git", "-C", source_root, "rev-parse", "--short=12", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run_one(script, args, config, seed, repeat):
    command = [
        sys.executable,
        script,
        "--worker",
        "--feature",
        args.feature,
        "--source-root",
        args.source_root,
        "--config",
        config,
        "--seed",
        str(seed),
        "--repeat",
        str(repeat),
        "--time-limit",
        str(args.time_limit),
        "--threads",
        str(args.threads),
    ]
    env = os.environ.copy()
    env.update(
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
    )
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, env=env
    )
    if completed.returncode:
        return {
            "feature": args.feature,
            "instance": "synthetic-{:02d}".format(seed),
            "seed": seed,
            "repeat": repeat,
            "config": config,
            "status": "error: {}".format(
                completed.stderr.strip().splitlines()[-1]
                if completed.stderr.strip()
                else "worker failed"
            ),
        }
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _validated_summary(rows):
    by_instance = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["instance"], row["repeat"])
        by_instance.setdefault(key, []).append(row)
    for grouped in by_instance.values():
        objectives = [float(row["objective"]) for row in grouped]
        gaps = [float(row["gap"]) for row in grouped]
        if max(objectives) - min(objectives) > 1e-6 or any(
            abs(gap) > 1e-7 for gap in gaps
        ):
            for row in grouped:
                row["status"] = "MISMATCH"

    print("config              n   median total   median RSS   status")
    for config in sorted({row["config"] for row in rows}):
        group = [row for row in rows if row["config"] == config]
        valid = [row for row in group if row["status"] == "ok"]
        if valid:
            total = statistics.median(float(row["total_time"]) for row in valid)
            rss = statistics.median(float(row["peak_rss_mb"]) for row in valid)
            status = "ok"
        else:
            total, rss = float("nan"), float("nan")
            status = group[0]["status"]
        print(
            "{:<19} {:>3} {:>13.4f}s {:>11.1f}MB   {}".format(
                config, len(valid), total, rss, status
            )
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", required=True, choices=sorted(DEFAULT_CONFIGS))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--configs")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args)
        return
    if not args.output:
        parser.error("--output is required")

    configs = (args.configs or DEFAULT_CONFIGS[args.feature]).split(",")
    valid_configs = set(DEFAULT_CONFIGS[args.feature].split(","))
    unknown = set(configs) - valid_configs
    if unknown:
        parser.error("unknown config(s): {}".format(sorted(unknown)))
    seeds = [int(value) for value in args.seeds.split(",")]
    script = os.path.abspath(__file__)
    commit = _commit(args.source_root)
    rows = []
    for repeat in range(args.repeats):
        for seed in seeds:
            for config in configs:
                row = {field: "" for field in FIELDS}
                row.update(_run_one(script, args, config, seed, repeat))
                row.update(label=args.label, commit=commit)
                rows.append(row)
                print(
                    "{} {} seed={} repeat={} [{}]".format(
                        args.feature, config, seed, repeat, row["status"]
                    )
                )

    _validated_summary(rows)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    mode = "a" if args.append else "w"
    write_header = mode == "w" or not os.path.exists(output)
    with open(output, mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
