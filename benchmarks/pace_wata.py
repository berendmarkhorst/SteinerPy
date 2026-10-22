"""Run the original wata&sigma solver on the nine-case PACE comparison sample.

Build https://github.com/wata-orz/steiner_tree separately with cargo build
--release. This uses the same isolated wall-deadline supervisor as pace2018.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

from benchmarks.pace2018 import provenance, reference_cases, run_case, write_json


def parse_solution(text):
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if not lines or len(lines[0]) != 2 or lines[0][0] != "VALUE":
        raise ValueError("missing VALUE header")
    edges = []
    for line in lines[1:]:
        if len(line) != 2:
            raise ValueError("expected two endpoints per edge")
        edges.append(tuple(map(int, line)))
    return int(lines[0][1]), edges


def worker(path):
    import resource

    request = json.loads(path.read_text())
    bound = int(request["memory_gb"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (bound, bound))
    row = dict(request["case"], status="running", phase="solve", feasible=False)
    progress = Path(request["progress"])
    write_json(progress, row)
    began = time.monotonic()
    try:
        # The binary inherits the process group and address-space limit; the
        # common parent kills both processes if the full wall deadline expires.
        with Path(row["path"]).open() as source:
            result = subprocess.run(
                [row["binary"]],
                stdin=source,
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
        row.update(solve_seconds=time.monotonic() - began, phase="validate")
        write_json(progress, row)
        from benchmarks.pace2018 import validate_solution
        from benchmarks.stp_parser import read_stp

        graph, groups = read_stp(row["path"])
        objective, edges = parse_solution(result.stdout)
        row.update(
            nodes=len(graph),
            edges=graph.number_of_edges(),
            terminals=len(groups[0]),
            objective=objective,
            gap=0.0,
            solution_edges=edges,
        )
        row.update(
            validate_solution(
                graph,
                groups[0],
                edges,
                objective,
                0,
                row["reference_lower"],
                row["reference_upper"],
            )
        )
        row["phase"] = "complete"
    except MemoryError:
        row.update(status="memory_limit", error="MemoryError")
    except Exception as error:
        row.update(status="error", error="{}: {}".format(type(error).__name__, error))
    # Maximum individual-process RSS, not the sum of concurrently live processes.
    row["peak_rss_mb"] = (
        max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        / 1024
    )
    write_json(progress, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--solver-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not all((args.dataset, args.solver_root, args.output)):
        parser.error("--dataset, --solver-root and --output are required")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.config = "wata"
    args.time_limit = 90.0
    args.memory_gb = 8.0
    args.worker_module = "benchmarks.pace_wata"
    context = provenance(args.dataset)
    context["competitor_commit"] = subprocess.check_output(
        ["git", "-C", str(args.solver_root), "rev-parse", "HEAD"], text=True
    ).strip()
    context["compiler"] = subprocess.check_output(
        ["rustc", "--version"], text=True
    ).strip()
    context["binary_sha256"] = {}
    cases = []
    for track in (1, 2, 3):
        binary = (
            args.solver_root / "target/release" / ("track2" if track == 2 else "track1")
        ).resolve()
        context["binary_sha256"][binary.name] = hashlib.sha256(
            binary.read_bytes()
        ).hexdigest()
        for case in reference_cases(args.dataset, [track])[:3]:
            cases.append(dict(case, binary=str(binary)))
    write_json(
        args.output / "manifest-wata-0.json",
        dict(
            run=context,
            cases=cases,
            config=args.config,
            time_limit=args.time_limit,
            memory_gb=args.memory_gb,
        ),
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda case: run_case(case, args, context), cases))


if __name__ == "__main__":
    main()
