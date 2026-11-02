"""Discovery and held-out comparisons for uniform-cost formulation selection.

Manifests are deterministic and can be written before measuring either split.
Every measurement is a fresh process and reuses the original uniform benchmark's
certificate checks. ``node`` forces the eligible node model; ``auto`` exercises
the package's selection rule. Timing excludes structural-feature measurement.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

FAMILIES = (
    "bipartite",
    "regular_incidence",
    "subdivision",
    "checkerboard_grid",
    "sparse_grid",
    "parity_cube",
    "random_cube",
    "regular",
    "random",
    "small_world",
    "geometric",
    "clustered",
    "bridged_incidence",
)
CHALLENGE_FAMILIES = (
    "torus",
    "king_grid",
    "grid3",
    "periodic_grid3",
    "dangling_clique",
    "terminal_cliques",
    "incidence_chain",
    "cube_leaves",
)
CONFIGS = ("node", "directed", "auto")
SOURCE_FILES = (
    "steinerpy/_uniform_node.py",
    "steinerpy/objects.py",
    "steinerpy/mathematical_model.py",
    "benchmarks/benchmark_uniform.py",
    "benchmarks/uniform_selection.py",
)


def make_manifest(split):
    """Fix both sizes and seeds before collecting measurements."""
    if split == "challenge":
        cases = (
            [("torus", side, variant) for side in (8, 10) for variant in (0, 1)]
            + [("king_grid", side, 0) for side in (8, 10)]
            + [
                ("grid3", 6, 0),
                ("periodic_grid3", 6, 0),
                ("dangling_clique", 8, 0),
                ("terminal_cliques", 8, 0),
                ("incidence_chain", 12, 0),
                ("cube_leaves", 5, 0),
            ]
        )
        return [
            dict(
                id="challenge-{}-{}-{}".format(family, size, variant),
                split=split,
                family=family,
                size=size,
                seed=17,
                variant=variant,
            )
            for family, size, variant in cases
        ]
    if split not in {"discovery", "heldout"}:
        raise ValueError("A custom split requires --manifest")
    heldout = split == "heldout"
    seeds = (101, 102) if heldout else (0, 1)
    rows = []
    for family in FAMILIES:
        if family in {
            "bipartite",
            "regular_incidence",
            "subdivision",
            "bridged_incidence",
        }:
            sizes = (16, 24) if heldout else (12, 20)
        elif family in {"checkerboard_grid", "sparse_grid"}:
            sizes = (8, 10) if heldout else (6, 8)
            if family == "sparse_grid":
                sizes = tuple(size + 1 for size in sizes)
        elif family in {"parity_cube", "random_cube"}:
            sizes = (6, 7) if heldout else (5, 6)
        else:
            sizes = (64, 96) if heldout else (40, 64)
        for size in sizes:
            for variant, seed in enumerate(seeds):
                rows.append(
                    dict(
                        id="{}-{}-{}-{}".format(split, family, size, seed),
                        split=split,
                        family=family,
                        size=size,
                        seed=seed,
                        variant=variant,
                    )
                )
    return rows


def _connected(graph, rng):
    """Connect independent random graph components without deleting vertices."""
    import networkx as nx

    components = list(nx.connected_components(graph))
    for left, right in zip(components, components[1:]):
        graph.add_edge(rng.choice(sorted(left)), rng.choice(sorted(right)))
    return graph


def _regular_incidence(k, degree, rng):
    import networkx as nx

    for _ in range(100):
        graph = nx.Graph()
        graph.add_nodes_from(range(2 * k))
        for _ in range(degree):
            for _ in range(10000):
                permutation = rng.sample(range(k), k)
                if all(not graph.has_edge(i, k + j) for i, j in enumerate(permutation)):
                    graph.add_edges_from((i, k + j) for i, j in enumerate(permutation))
                    break
            else:
                break
        if graph.number_of_edges() == degree * k and nx.is_connected(graph):
            return graph
    raise RuntimeError("Could not generate connected regular incidence graph")


def make_case(spec):
    """Generate one connected, positive uniform instance without solver calls."""
    import networkx as nx

    if "path" in spec:
        from benchmarks.stp_parser import read_stp

        graph, groups = read_stp(str(spec["path"]))
        if len(groups) != 1:
            raise ValueError("Uniform selection benchmarks require one terminal group")
        return graph, groups[0]

    family, size = spec["family"], spec["size"]
    rng = random.Random(spec["seed"])
    variant = spec["variant"]
    heldout = spec["split"] == "heldout"
    if family in {"torus", "king_grid", "dangling_clique", "terminal_cliques"}:
        graph = nx.grid_2d_graph(size, size, periodic=family == "torus")
        terminals = [node for node in graph if all(value % 2 == 0 for value in node)]
        if family == "torus" and variant:
            neighbors = sorted(
                {neighbor for node in terminals for neighbor in graph[node]}
                - set(terminals)
            )
            terminals.extend(
                rng.sample(neighbors, round(0.3 * len(graph)) - len(terminals))
            )
        elif family == "king_grid":
            for x in range(size - 1):
                for y in range(size - 1):
                    graph.add_edges_from(
                        (((x, y), (x + 1, y + 1)), ((x + 1, y), (x, y + 1)))
                    )
        elif family == "dangling_clique":
            clique = [("clique", index) for index in range(12)]
            graph.add_edges_from(nx.complete_graph(clique).edges)
            graph.add_edge((0, 0), clique[0])
        elif family == "terminal_cliques":
            terminals = []
            for label, corner in (("left", (0, 0)), ("right", (size - 1, size - 1))):
                clique = [(label, index) for index in range(10)]
                graph.add_edges_from(nx.complete_graph(clique).edges)
                graph.add_edge(corner, clique[0])
                terminals.extend(clique)
    elif family in {"grid3", "periodic_grid3"}:
        graph = nx.grid_graph((size,) * 3, periodic=family == "periodic_grid3")
        terminals = [
            node
            for node in graph
            if (
                sum(node) % 2 == 0
                if family == "periodic_grid3"
                else all(value % 2 == 0 for value in node)
            )
        ]
    elif family == "incidence_chain":
        graph = nx.disjoint_union_all(
            [_regular_incidence(size, 6, rng) for _ in range(3)]
        )
        terminals = [
            offset + node for offset in (0, 2 * size, 4 * size) for node in range(size)
        ]
        for index in range(2):
            middle = 6 * size + index
            graph.add_edges_from(
                ((2 * size * index, middle), (middle, 2 * size * (index + 1)))
            )
    elif family == "cube_leaves":
        graph = nx.hypercube_graph(size)
        original = list(graph)
        terminals = [node for node in original if sum(node) % 2 == 0]
        graph.add_edges_from(
            (rng.choice(original), ("leaf", index)) for index in range(60)
        )
    elif family == "regular_incidence":
        degree = (4, 6)[variant] if heldout else (3, 5)[variant]
        graph = _regular_incidence(size, degree, rng)
        terminals = list(range(size))
    elif family == "bridged_incidence":
        graph = nx.disjoint_union(
            _regular_incidence(size, 3 if heldout else 4, rng),
            _regular_incidence(size, 3 if heldout else 4, rng),
        )
        terminals = list(range(size)) + list(range(2 * size, 3 * size))
        for index in range(2 if heldout else 1):
            if variant == 0:
                graph.add_edge(index, 2 * size + index)
            else:
                middle = 4 * size + index
                graph.add_edges_from(((index, middle), (middle, 2 * size + index)))
    elif family == "bipartite":
        k, count = size, 2 * size
        for _ in range(1000):
            graph = nx.Graph()
            graph.add_nodes_from(range(k + count))
            for node in range(k, k + count):
                degree = rng.randint(2, max(3, k // (2 if variant else 4)))
                graph.add_edges_from(
                    (terminal, node) for terminal in rng.sample(range(k), degree)
                )
            if nx.is_connected(graph):
                break
        else:
            raise RuntimeError("Could not generate connected bipartite graph")
        terminals = list(range(k))
    elif family == "subdivision":
        if heldout and variant == 0:
            original = nx.cycle_graph(size)
        elif heldout:
            original = nx.grid_2d_graph(4, size // 4)
            original = nx.convert_node_labels_to_integers(original)
        else:
            original = nx.random_regular_graph((3, 5)[variant], size, seed=rng)
            original = _connected(original, rng)
        graph = nx.Graph()
        graph.add_nodes_from(original)
        for index, (left, right) in enumerate(original.edges):
            middle = len(original) + index
            graph.add_edges_from(((left, middle), (middle, right)))
        terminals = list(original)
    elif family in {"checkerboard_grid", "sparse_grid"}:
        width = size + variant
        graph = nx.grid_2d_graph(size, width)
        if family == "checkerboard_grid":
            terminals = [node for node in graph if sum(node) % 2 == 0]
        else:
            terminals = [
                node
                for node in graph
                if all(coordinate % 2 == 0 for coordinate in node)
            ]
    elif family in {"parity_cube", "random_cube"}:
        graph = nx.hypercube_graph(size)
        if family == "parity_cube":
            parity = variant
            terminals = [node for node in graph if sum(node) % 2 == parity]
        else:
            fraction = (0.4, 0.6)[variant]
            terminals = rng.sample(list(graph), max(12, round(fraction * len(graph))))
    else:
        if family == "regular":
            degree = (3, 8)[variant] if heldout else (4, 6)[variant]
            graph = nx.random_regular_graph(degree, size, seed=rng)
        elif family == "random":
            probability = (0.09, 0.28)[variant] if heldout else (0.12, 0.25)[variant]
            graph = nx.gnp_random_graph(size, probability, seed=rng)
        elif family == "small_world":
            graph = nx.watts_strogatz_graph(size, (4, 8)[variant], 0.25, seed=rng)
        elif family == "geometric":
            graph = nx.random_geometric_graph(size, (0.25, 0.4)[variant], seed=rng)
        elif family == "clustered":
            partitions = [size // 4] * 4
            graph = nx.stochastic_block_model(
                partitions,
                [
                    [0.65 if i == j else (0.01, 0.04)[variant] for j in range(4)]
                    for i in range(4)
                ],
                seed=rng,
            )
        else:
            raise ValueError("Unknown family: " + family)
        graph = _connected(graph, rng)
        terminals = rng.sample(
            list(graph), max(12, round((0.3, 0.5)[variant] * len(graph)))
        )
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_label")
    labels = {data["original_label"]: node for node, data in graph.nodes(data=True)}
    terminals = [labels[node] for node in terminals]
    nx.set_edge_attributes(graph, 1, "weight")
    return graph, terminals


def structural_features(graph, terminals):
    """Record inexpensive graph descriptors without assuming a selection rule."""
    import networkx as nx

    terminals = set(terminals)
    steiner = set(graph) - terminals
    n, m = len(graph), graph.number_of_edges()
    ss = sum(left in steiner and right in steiner for left, right in graph.edges)
    tt = sum(left in terminals and right in terminals for left, right in graph.edges)
    degrees = list(dict(graph.degree()).values())
    s_degrees = [graph.degree(node) for node in steiner]
    t_neighbors = [
        sum(neighbor in terminals for neighbor in graph[node]) for node in steiner
    ]
    costs = {data.get("weight", 1) for _, _, data in graph.edges(data=True)}
    result = dict(
        nodes=n,
        edges=m,
        terminals=len(terminals),
        steiner_nodes=len(steiner),
        density=nx.density(graph),
        terminal_fraction=len(terminals) / n if n else 0,
        steiner_edges=ss,
        terminal_edges=tt,
        cross_edges=m - ss - tt,
        steiner_independent=ss == 0,
        terminal_independent=tt == 0,
        aligned_bipartite=ss == tt == 0,
        bipartite=nx.is_bipartite(graph),
        terminal_components=nx.number_connected_components(graph.subgraph(terminals)),
        steiner_components=nx.number_connected_components(graph.subgraph(steiner)),
        steiner_degree2_fraction=(
            sum(degree == 2 for degree in s_degrees) / len(s_degrees)
            if s_degrees
            else 0
        ),
        uniform_positive=(
            len(costs) == 1 and all(math.isfinite(cost) and cost > 0 for cost in costs)
        ),
    )
    for prefix, values in (
        ("degree", degrees),
        ("steiner_degree", s_degrees),
        ("steiner_terminal_neighbors", t_neighbors),
    ):
        result.update(
            {
                prefix + "_min": min(values, default=0),
                prefix + "_max": max(values, default=0),
                prefix + "_mean": sum(values) / len(values) if values else 0,
            }
        )
    return result


def worker(args, spec):
    """Reuse the original worker and remove descriptor overhead from timings."""
    sys.path.insert(0, str(args.source_root.resolve()))
    from benchmarks import benchmark_uniform as base
    from steinerpy import SteinerProblem

    if "path" in spec:
        spec = dict(spec, path=str((args.source_root / spec["path"]).resolve()))
    features = {}
    elapsed = [0.0]
    original = SteinerProblem.get_solution

    def record(problem, *solve_args, **solve_kwargs):
        started = perf_counter()
        features.update(structural_features(problem.graph, problem.terminal_groups[0]))
        elapsed[0] += perf_counter() - started
        return original(problem, *solve_args, **solve_kwargs)

    with patch.object(base, "make_case", lambda name: make_case(spec)), patch.object(
        SteinerProblem, "get_solution", record
    ):
        row = base.worker(args)
    row["features"] = features
    row["feature_seconds"] = elapsed[0]
    for key in ("solve_seconds", "total_seconds"):
        if key in row:
            row[key] = max(0.0, row[key] - elapsed[0])
    return row


def run_worker(args, spec, config, repeat):
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        STEINERPY_THREADS="1",
        STEINERPY_DW_MAX_TERMINALS="10",
        STEINERPY_DECOMPOSE="0",
        STEINERPY_UNIFORM_NODE_MODEL={"node": "1", "directed": "0", "auto": "auto"}[
            config
        ],
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source-root",
        str(args.source_root.resolve()),
        "--split",
        args.split,
        "--case",
        spec["id"],
        "--preprocess",
        args.preprocess,
        "--solver",
        args.solver,
        "--time-limit",
        str(args.time_limit),
    ]
    if args.manifest:
        command.extend(("--manifest", str(args.manifest.resolve())))
    row = dict(
        case=spec["id"],
        family=spec["family"],
        split=spec["split"],
        seed=spec["seed"],
        config=config,
        preprocess=args.preprocess == "on",
        repeat=repeat,
    )
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name != "nt",
        )
        try:
            stdout, stderr = process.communicate(timeout=args.worker_timeout)
        except subprocess.TimeoutExpired:
            # The Windows venv launcher has a child interpreter; stop both.
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
            row.update(status="timeout", error="Worker deadline exceeded")
        else:
            if process.returncode:
                row.update(status="error", error=stderr[-3000:])
            else:
                row.update(json.loads(stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError) as error:
        row.update(status="error", error="Invalid worker output: " + str(error))
    return row


def source_hashes(root, specs):
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }
    hashes.update(
        {
            "input:"
            + spec["id"]: hashlib.sha256((root / spec["path"]).read_bytes()).hexdigest()
            for spec in specs
            if "path" in spec
        }
    )
    return hashes


def load_manifest(path, split):
    """Load predeclared generated or STP cases without altering the source file."""
    specs = json.loads(path.read_text(encoding="utf-8"))
    if len({spec["id"] for spec in specs}) != len(specs):
        raise ValueError("Manifest instance IDs must be unique")
    return [
        dict(
            spec,
            split=spec.get("split", split),
            family=spec.get("family", "external"),
            seed=spec.get("seed"),
        )
        for spec in specs
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-manifests", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="discovery")
    parser.add_argument("--solver", choices=("highs", "gurobi"), default="highs")
    parser.add_argument(
        "--configs", nargs="+", choices=CONFIGS, default=("node", "directed")
    )
    parser.add_argument("--families", nargs="+", choices=FAMILIES + CHALLENGE_FAMILIES)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--preprocess", choices=("off", "on"), default="off")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=5)
    parser.add_argument("--worker-timeout", type=float, default=30)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.write_manifests:
        args.write_manifests.mkdir(parents=True, exist_ok=True)
        for split in ("discovery", "heldout"):
            (args.write_manifests / (split + "-manifest.json")).write_text(
                json.dumps(make_manifest(split), indent=2) + "\n", encoding="utf-8"
            )
        if args.output is None:
            return
    if args.repeats < 1 or not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("repeats and time limit must be positive and finite")
    specs = (
        load_manifest(args.manifest, args.split)
        if args.manifest
        else make_manifest(args.split)
    )
    if args.worker:
        spec = next(spec for spec in specs if spec["id"] == args.case)
        print(json.dumps(worker(args, spec), allow_nan=False))
        return
    if args.output is None:
        parser.error("require --output")
    if args.families:
        specs = [spec for spec in specs if spec["family"] in args.families]
    if args.cases:
        unknown = set(args.cases) - {spec["id"] for spec in specs}
        if unknown:
            parser.error("Unknown cases: " + ", ".join(sorted(unknown)))
        specs = [spec for spec in specs if spec["id"] in args.cases]
    sys.path.insert(0, str(args.source_root.resolve()))
    from benchmarks.benchmark_uniform import check_pairs

    packages = {}
    for package in ("networkx", "highspy", "scipy", "gurobipy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    hashes = source_hashes(args.source_root, specs)
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        split=args.split,
        manifest=specs,
        manifest_sha256=hashlib.sha256(
            json.dumps(specs, sort_keys=True).encode()
        ).hexdigest(),
        source_hashes=hashes,
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
        solver=args.solver,
        threads=1,
        time_limit=args.time_limit,
        repeats=args.repeats,
        worker_timeout=args.worker_timeout,
        rows=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for index, spec in enumerate(specs):
            order = (
                args.configs
                if (repeat + index) % 2 == 0
                else list(reversed(args.configs))
            )
            for config in order:
                row = run_worker(args, spec, config, repeat)
                report["rows"].append(row)
                print(
                    "{} {}: {} {:.3f}s".format(
                        spec["id"],
                        config,
                        row["status"],
                        row.get("total_seconds", math.nan),
                    ),
                    flush=True,
                )
                args.output.write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
    if hashes != source_hashes(args.source_root, specs):
        raise SystemExit(
            "Source changed during benchmark; measurements must not be pooled"
        )
    check_pairs(report["rows"])
    if any(row["status"] == "error" for row in report["rows"]):
        raise SystemExit("Benchmark recorded worker errors; inspect output")


if __name__ == "__main__":
    main()
