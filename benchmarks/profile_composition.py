"""Bounded local profiling of compatible general formulations (issue #92).

Run as ``python -m benchmarks.profile_composition --output results.json``.
Each case/configuration runs in a fresh process, using one HiGHS thread.
"""

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from functools import wraps
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

from benchmarks.benchmark_dispatch import peak_rss_mb

CASES = (
    "mandatory-costs",
    "optional-costs",
    "groups",
    "forest",
    "groups-forest",
    "directed-groups",
    "degree-hops",
    "terminal-leaves",
    "prize-forest",
    "node-budget",
    "coverage-budget",
    "coverage-feasible",
    "penalty-budget",
    "signed-rewards",
)
CUT_CASES = {
    "mandatory-costs",
    "optional-costs",
    "groups",
    "directed-groups",
    "degree-hops",
}
PHASES = (
    "preprocessing",
    "compilation",
    "model_build",
    "optimization",
    "separation",
    "decoding",
    "other",
)


def make_case(name, seed=42):
    import networkx as nx

    size = 48 if name in {"mandatory-costs", "optional-costs"} else 24
    graph = nx.gnm_random_graph(size, 2 * size, seed=seed)
    graph.add_edges_from(nx.cycle_graph(size).edges)
    rng = random.Random(seed)
    nx.set_edge_attributes(graph, {e: rng.randint(1, 9) for e in graph.edges}, "weight")
    options = dict(
        terminals=[0, size // 3, 2 * size // 3],
        node_costs={n: 2 + n % 3 for n in graph if n % 4 == 3},
    )
    if name == "mandatory-costs":
        options.update(terminals=list(range(0, size, 4)))
        options["node_costs"] = {n: 2 + n % 5 for n in options["terminals"]}
    if name in {"groups", "groups-forest", "directed-groups"}:
        options.update(groups=[[5, 6], [18, 19]])
    if "forest" in name:
        options["terminals"] = [[0, 4], [12, 16]]
    if name == "directed-groups":
        graph = graph.to_directed()
        options.update(root=0, terminals=[8, 16])
    if name == "degree-hops":
        options.update(max_degree=3, hop_limit=12)
    if name == "terminal-leaves":
        options.update(partial_terminals=[8, 16], max_degree=3)
    if name in {"prize-forest", "node-budget"}:
        options.update(node_prizes={6: 7, 14: 9, 22: 5})
    if name == "node-budget":
        options.update(node_budget=3, budget=40)
    if name in {"coverage-budget", "coverage-feasible"}:
        options.update(groups=[[5, 6], [18, 19]], budget=12, node_budget=3)
        if name == "coverage-feasible":
            options["terminals"] = [0]
    if name == "penalty-budget":
        options.update(penalty_cost=3, penalty_budget=1, hop_limit=8)
    if name == "signed-rewards":
        options = dict(
            root=0,
            node_costs={n: 1 + n % 3 for n in graph},
            node_weights={n: (5 if n % 4 == 0 else -1) for n in graph},
            budget=20,
            node_budget=12,
            max_degree=3,
        )
    return graph, options


class PhaseTimer:
    """Attribute nested calls exclusively; restore all instrumentation on exit."""

    def __init__(self, clock=perf_counter):
        self.clock = clock
        self.seconds = Counter(dict.fromkeys(PHASES, 0.0))
        self.calls = Counter()
        self.stack = []
        self.backend_runs = []

    @contextmanager
    def measure(self, phase):
        frame = [phase, self.clock(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - frame[1]
            self.stack.pop()
            self.seconds[phase] += elapsed - frame[2]
            self.calls[phase] += 1
            if self.stack:
                self.stack[-1][2] += elapsed

    @contextmanager
    def instrument(self):
        import highspy
        from steinerpy import mathematical_model as mm
        from steinerpy import _composition as cuts, _composition_budget as budget
        from steinerpy import _selection_model as selection
        from steinerpy._feature_composition import FeatureComposition

        def measured(original, phase, materialize=False, backend=False):
            @wraps(original)
            def call(*args, **kwargs):
                with self.measure(phase):
                    result = original(*args, **kwargs)
                    if materialize:
                        result = list(result)
                if backend:
                    model = args[0]
                    info = model.getInfo()
                    self.backend_runs.append(
                        dict(
                            columns=model.getNumCol(),
                            rows=model.getNumRow(),
                            status=model.modelStatusToString(model.getModelStatus()),
                            mip_nodes=info.mip_node_count,
                        )
                    )
                return result

            return call

        hooks = [
            (selection, "feasible_edges", "preprocessing"),
            (selection.SelectionModel, "__init__", "model_build"),
            (FeatureComposition, "configure", "model_build"),
            (FeatureComposition, "decode", "decoding"),
            (mm, "build_model", "model_build"),
            (mm, "build_budget_model", "model_build"),
            (mm, "find_violated_cuts_from_values", "separation"),
            (highspy.Highs, "solve", "optimization"),
        ]
        if hasattr(selection.SelectionModel, "strengthen_connectivity"):
            hooks.append(
                (selection.SelectionModel, "strengthen_connectivity", "model_build")
            )
        for module in (cuts, budget):
            hooks.extend(
                (module, name, "compilation")
                for name in ("compile_groups", "make_view")
            )
            hooks.append((module, "decode_groups", "decoding"))
        hooks.append((budget, "decode_budget", "decoding"))
        with ExitStack() as stack:
            for owner, name, phase in hooks:
                stack.enter_context(
                    patch.object(
                        owner,
                        name,
                        measured(
                            getattr(owner, name),
                            phase,
                            materialize=name == "feasible_edges",
                            backend=name == "solve",
                        ),
                    )
                )
            yield


def worker(case, config, limit, profiled=True):
    from steinerpy import Problem
    from steinerpy._problem_spec import normalize_problem
    from steinerpy._problem_validation import validate_problem
    from steinerpy._feature_composition import FeatureComposition
    from steinerpy._composition import GroupComposition
    from steinerpy._selection_model import NoIncumbent

    if config == "cuts" and case not in CUT_CASES:
        raise ValueError("Cut composition does not support every requested feature.")
    graph, options = make_case(case)
    options["preprocess"] = config == "flow-filtered"
    timer = PhaseTimer()
    row = dict(
        case=case,
        config=config,
        nodes=len(graph),
        edges=len(graph.edges),
        profiled=profiled,
    )
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    started = perf_counter()
    try:
        with ExitStack() as stack:
            if profiled:
                stack.enter_context(timer.instrument())
            with timer.measure("other"):
                # Force the formulation after validation, independently of public dispatch.
                spec = normalize_problem(
                    graph,
                    None,
                    options.pop("terminals", None),
                    None,
                    options,
                    Problem.variants,
                )
                validate_problem(spec, None)
                strategy = (
                    GroupComposition if config == "cuts" else FeatureComposition
                )(spec)
                solution = strategy.get_solution(
                    time_limit=limit, solver="highs", threads=1
                )
        finished = perf_counter()
        phases = dict(timer.seconds)
        # Check after timing, on the original graph; no solver-model values are trusted.
        checker = FeatureComposition(spec)
        decoded = checker.decode(solution.edges, list(solution.selected_nodes))
        checked = checker.solution(decoded, solution.gap, solution.runtime)
        assert math.isfinite(solution.objective) and math.isclose(
            checked.objective, solution.objective, rel_tol=1e-9, abs_tol=1e-9
        )
        row.update(
            status="optimal" if solution.gap == 0 else "feasible",
            objective=solution.objective,
            total_cost=checked.total_cost,
            objective_sense=(
                "max"
                if checker.coverage_mode or checker.objective.report_sign < 0
                else "min"
            ),
            lexicographic=checker.coverage_mode,
            gap=solution.gap if math.isfinite(solution.gap) else None,
        )
    except Exception as error:
        finished = perf_counter()
        phases = dict(timer.seconds)
        status = "error"
        if isinstance(error, NoIncumbent):
            status = "infeasible" if error.status == "infeasible" else "no_incumbent"
        row.update(status=status, error=f"{type(error).__name__}: {error}")
    elapsed = finished - started
    phases["other"] += elapsed - sum(phases.values())
    row.update(
        total_seconds=elapsed,
        phases_seconds=phases if profiled else None,
        phase_calls=dict(timer.calls) if profiled else None,
        backend_runs=timer.backend_runs if profiled else None,
        peak_rss_mb=peak_rss_mb(),
    )
    logging.disable(previous_logging)
    return row


def check_results(rows):
    """Compare proof claims, including the secondary cost of coverage optima."""
    for case in {r["case"] for r in rows}:
        valid = [
            r
            for r in rows
            if r["case"] == case and r["status"] in {"optimal", "feasible"}
        ]
        if any(r["case"] == case and r["status"] == "infeasible" for r in rows):
            assert not valid, f"{case}: infeasibility contradicts a valid selection"
        optima = [r for r in valid if r["status"] == "optimal"]
        if not optima:
            continue
        best = optima[0]
        for row in valid:
            if row["lexicographic"] and math.isclose(
                row["objective"], best["objective"]
            ):
                assert row["total_cost"] >= best["total_cost"] - 1e-7, case
            if row["status"] == "optimal":
                assert math.isclose(row["objective"], best["objective"]), case
                if row["lexicographic"]:
                    assert math.isclose(row["total_cost"], best["total_cost"]), case
            elif row["objective_sense"] == "min":
                assert row["objective"] >= best["objective"] - 1e-7, case
            else:
                assert row["objective"] <= best["objective"] + 1e-7, case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=5)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--config", choices=("flow", "flow-filtered", "cuts"))
    args = parser.parse_args()
    if args.time_limit <= 0 or not math.isfinite(args.time_limit) or args.repeats < 1:
        parser.error("limits and repeats must be positive and finite")
    if args.worker:
        if not args.config:
            parser.error("worker requires a configuration")
        print(
            json.dumps(
                worker(args.worker, args.config, args.time_limit, not args.no_profile),
                allow_nan=False,
            )
        )
        return
    if not args.output:
        parser.error("require --output")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        STEINERPY_THREADS="1",
    )
    rows = []
    report = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        platform=platform.platform(),
        cpu=platform.processor(),
        python=sys.version.split()[0],
        packages={p: version(p) for p in ("networkx", "highspy", "scipy")},
        threads=1,
        repeats=args.repeats,
        cases=list(args.cases),
        profiled=not args.no_profile,
        seed=42,
        solver="highs",
        time_limit=args.time_limit,
        worker_timeout=max(30, args.time_limit + 15),
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for case in args.cases:
            configs = ["flow", "flow-filtered"] + (
                ["cuts"] if case in CUT_CASES else []
            )
            if repeat % 2:
                configs.reverse()
            for config in configs:
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.profile_composition",
                    "--worker",
                    case,
                    "--config",
                    config,
                    "--time-limit",
                    str(args.time_limit),
                ]
                if args.no_profile:
                    command.append("--no-profile")
                try:
                    result = subprocess.run(
                        command,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=report["worker_timeout"],
                    )
                    if result.returncode:
                        row = dict(status="error", error=result.stderr)
                    else:
                        row = json.loads(result.stdout)
                except subprocess.TimeoutExpired:
                    row = dict(status="timeout")
                rows.append(dict(row, case=case, config=config, repeat=repeat))
                # Preserve errors and earlier results even when the run is interrupted.
                args.output.write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
        print(f"Completed repeat {repeat + 1}/{args.repeats}", flush=True)
    check_results(rows)
    print(dict(Counter(r["status"] for r in rows)), flush=True)


if __name__ == "__main__":
    main()
