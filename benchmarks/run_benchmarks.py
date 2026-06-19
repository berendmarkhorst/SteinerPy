"""Benchmark runner: SteinerPy baseline vs the dual-ascent accelerator.

Usage
-----
    python -m benchmarks.run_benchmarks \
        --instances benchmarks/data/B \
        --optima benchmarks/optima.csv \
        --time-limit 300 --solver highs \
        --config both --out benchmarks/results/B.csv [--jobs 4] [--full]

Design for HPC use:
* **headless** — no plotting/interactive dependencies;
* **resumable** — instances already present in ``--out`` are skipped and the CSV
  is append-only, so a preempted sweep can be restarted with the same command;
* **parallel** — ``--jobs N`` fans instances out over processes (best effort;
  for hard per-instance wall-clock isolation prefer the Slurm array, one
  instance per task — see ``slurm_array.sh``);
* each solve honours the solver ``time_limit`` as its primary guard.

Correctness: when an instance has a published optimum and is solved to
optimality (gap ~ 0), we assert ``objective == optimum``; a disagreement between
the baseline and the accelerated run is flagged as a correctness bug.
"""

import argparse
import csv
import glob
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .stp_parser import read_stp, instance_stats
from .optima import load_optima, optimum_for

FIELDS = [
    "instance", "nodes", "edges", "terminals", "opt",
    "base_obj", "base_rt", "base_gap",
    "da_obj", "da_rt", "da_gap",
    "speedup", "n_fixed", "status",
]


def _solve(stp_path, time_limit, solver, config):
    """Worker: solve one instance under the requested config(s). Returns a row dict."""
    logging.disable(logging.CRITICAL)  # silence solver chatter in workers
    from steinerpy import SteinerProblem  # imported in-worker for spawn safety

    graph, tg = read_stp(stp_path)
    stats = instance_stats(graph, tg)
    name = os.path.splitext(os.path.basename(stp_path))[0]
    row = {f: "" for f in FIELDS}
    row.update(instance=name, nodes=stats["nodes"], edges=stats["edges"],
               terminals=stats["terminals"], status="ok")

    def run(use_da):
        t0 = time.time()
        prob = SteinerProblem(graph.copy(), [list(tg[0])], preprocess=True, dual_ascent=use_da)
        sol = prob.get_solution(time_limit=time_limit)
        return sol, time.time() - t0

    try:
        if config in ("baseline", "both"):
            sol, rt = run(False)
            row.update(base_obj=round(sol.objective, 4), base_rt=round(rt, 3),
                       base_gap=round(sol.gap, 6))
        if config in ("da", "both"):
            # Report how many variables dual ascent fixes (model shrinkage).
            from steinerpy.dual_ascent import dual_ascent, reduced_cost_fixing
            probe = SteinerProblem(graph.copy(), [list(tg[0])], preprocess=True, dual_ascent=True)
            da_res = dual_ascent(probe)
            row["n_fixed"] = reduced_cost_fixing(probe, da_res).total()
            sol_da, rt_da = run(True)
            row.update(da_obj=round(sol_da.objective, 4), da_rt=round(rt_da, 3),
                       da_gap=round(sol_da.gap, 6))
    except Exception as exc:  # never let one instance abort the sweep
        row["status"] = f"error:{type(exc).__name__}"
        return row

    if config == "both" and row["base_obj"] != "" and row["da_obj"] != "":
        if abs(float(row["base_obj"]) - float(row["da_obj"])) > 1e-4:
            row["status"] = "MISMATCH"  # correctness bug: configs disagree
        elif row["base_rt"] and float(row["da_rt"]) > 0:
            row["speedup"] = round(float(row["base_rt"]) / max(1e-9, float(row["da_rt"])), 2)
    return row


def _validate_optimum(row, optima):
    opt = optimum_for(row["instance"], optima)
    if opt is None:
        return
    row["opt"] = opt
    for key in ("base_obj", "da_obj"):
        val = row.get(key)
        if val != "" and abs(float(val) - opt) < 1e-4:
            continue
        if val != "" and row.get(f"{key[:-4]}_gap") in (0, 0.0, "0", "0.0"):
            # solved to optimality but objective != published optimum
            row["status"] = f"WRONG_OPT({key})"


def _already_done(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add(r["instance"])
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="SteinerPy benchmark runner")
    ap.add_argument("--instances", required=True, help="directory of .stp files")
    ap.add_argument("--optima", default=None, help="optional instance,optimum CSV")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--solver", default="highs", choices=["highs", "gurobi"])
    ap.add_argument("--config", default="both", choices=["baseline", "da", "both"])
    ap.add_argument("--out", required=True, help="output CSV (append-only, resumable)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--full", action="store_true",
                    help="include all instances (default skips very large ones)")
    args = ap.parse_args(argv)

    optima = load_optima(args.optima)
    paths = sorted(glob.glob(os.path.join(args.instances, "*.stp")))
    done = _already_done(args.out)
    paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] not in done]
    if not args.full:
        # Skip very large instances that a HiGHS lazy-cut solver won't finish.
        def small(p):
            g, _ = read_stp(p)
            return g.number_of_nodes() <= 1000
        paths = [p for p in paths if small(p)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()

        def finalize(row):
            _validate_optimum(row, optima)
            writer.writerow(row)
            fh.flush()
            print(f"{row['instance']:<10} opt={row['opt']!s:>6} "
                  f"base={row['base_obj']!s:>8}/{row['base_rt']!s:>7}s "
                  f"da={row['da_obj']!s:>8}/{row['da_rt']!s:>7}s "
                  f"fixed={row['n_fixed']!s} speedup={row['speedup']!s} [{row['status']}]")

        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = {ex.submit(_solve, p, args.time_limit, args.solver, args.config): p
                        for p in paths}
                for fut in as_completed(futs):
                    finalize(fut.result())
        else:
            for p in paths:
                finalize(_solve(p, args.time_limit, args.solver, args.config))


if __name__ == "__main__":
    main()
