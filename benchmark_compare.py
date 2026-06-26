"""
benchmark_compare.py — SteinerPy vs NetworkX on literature instances
====================================================================

Run from the project root inside the test_env (or .venv):

    source test_env/bin/activate
    python benchmark_compare.py                       # bundled SteinLib B-series
    python benchmark_compare.py --instances path/to/E # any directory of .stp files
    python benchmark_compare.py --time-limit 30       # exact solves use Gurobi
    python benchmark_compare.py --solver highs        # fall back to HiGHS

What this compares
------------------
Steiner-tree methods, run on standard *literature* instances (SteinLib ``.stp``
files with published optima) rather than random graphs:

* networkx approximation.steiner_tree, method="kou"      — existing heuristic
* networkx approximation.steiner_tree, method="mehlhorn" — existing heuristic
* steinerpy exact ILP (Gurobi by default), no dual-ascent — exact baseline
* steinerpy exact ILP + dual_ascent=True                 — exact accelerator
* steinerpy get_solution(exact=False)                    — heuristic-only mode

Metrics captured per instance
-----------------------------
* Wall-clock solve time (seconds)
* Total Steiner-tree weight (objective)
* **Optimality gap measured against the published optimum**,
  ``gap = (obj - opt) / opt``, computed uniformly for *every* method so that
  networkx, the SteinerPy heuristic, and the exact solver sit on one scale.

Why measure against the published optimum?
------------------------------------------
networkx's ``steiner_tree`` is a 2-approximation that returns no lower bound, so
on its own it can't tell you how far from optimal it is. SteinLib publishes
proven optima (see ``benchmarks/optima.py``); using them as ground truth lets us
score the networkx heuristics on the same optimality-gap axis as SteinerPy.

In addition, SteinerPy carries its *own* certified gap (``Solution.gap``):
  * exact / exact+DA — the MIP optimality gap (0.0 == proven optimal);
  * exact=False      — a valid bound on how far the heuristic tree could be from
                       the optimum (0.0 == provably optimal).
networkx provides no such certificate; that column is shown for the SteinerPy
heuristic to make the "provably within X%" guarantee visible.

The infrastructure (parser, instances, optima) lives under ``benchmarks/`` and is
reused here as-is:
  * benchmarks/stp_parser.py — read_stp(), instance_stats()
  * benchmarks/optima.py     — load_optima(), optimum_for()
  * benchmarks/data/B/       — 18 SteinLib B-series instances (|V| = 50..100)
"""

import argparse
import os
import sys
import time

import networkx as nx
from networkx.algorithms.approximation import steiner_tree as nx_steiner_tree

# Make ``benchmarks`` importable regardless of the current working directory
# (benchmarks/ is a package, normally driven via ``python -m benchmarks...``).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from steinerpy import SteinerProblem
from benchmarks.stp_parser import read_stp, instance_stats
from benchmarks.optima import load_optima, optimum_for

DEFAULT_INSTANCES = os.path.join("benchmarks", "data", "B")
NX_METHODS = ("kou", "mehlhorn")

# Method registry: key -> column header. Order drives the table layout.
METHODS = [
    ("nx_kou", "NX-kou"),
    ("nx_meh", "NX-meh"),
    ("sp_exact", "SP-exact"),
    ("sp_da", "SP+DA"),
    ("sp_heur", "SP-heur"),
]


# ---------------------------------------------------------------------------
# Gap helpers
# ---------------------------------------------------------------------------

def gap_vs_opt(obj, opt):
    """Relative gap of an objective above a reference optimum, as a fraction."""
    if obj is None or opt is None or opt == 0:
        return None
    return (obj - opt) / opt


def _record(result, key, obj, runtime, opt, cgap=None):
    """Store a method's objective/time, its gap vs. the optimum, and (optional)
    its own certified gap, under the given key prefix."""
    result[f"{key}_obj"] = obj
    result[f"{key}_time"] = runtime
    result[f"{key}_gap"] = gap_vs_opt(obj, opt)
    result[f"{key}_cgap"] = cgap


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(stp_path, optima, time_limit, solver):
    graph, tg = read_stp(stp_path)
    stats = instance_stats(graph, tg)
    name = os.path.splitext(os.path.basename(stp_path))[0]
    terminals = list(tg[0])
    opt = optimum_for(name, optima)
    directed = isinstance(graph, nx.DiGraph)

    result = {
        "label": name,
        "n": stats["nodes"],
        "edges": stats["edges"],
        "k": stats["terminals"],
        "opt": opt,
        "opt_is_published": opt is not None,
        "status": "ok",
    }

    try:
        # ---- networkx approximations (existing methods; undirected only) ----
        for method, key in zip(NX_METHODS, ("nx_kou", "nx_meh")):
            if directed:
                _record(result, key, None, None, opt)
                continue
            t0 = time.perf_counter()
            tree = nx_steiner_tree(graph, terminals, weight="weight", method=method)
            rt = time.perf_counter() - t0
            _record(result, key, tree.size(weight="weight"), rt, opt)

        # ---- steinerpy exact ILP (no dual-ascent) ----
        t0 = time.perf_counter()
        sol = SteinerProblem(graph.copy(), [terminals], preprocess=True).get_solution(
            time_limit=time_limit, solver=solver, dual_ascent=False)
        _record(result, "sp_exact", sol.objective, time.perf_counter() - t0, opt,
                cgap=sol.gap)

        # ---- steinerpy exact ILP + dual-ascent accelerator ----
        t0 = time.perf_counter()
        sol = SteinerProblem(graph.copy(), [terminals], preprocess=True).get_solution(
            time_limit=time_limit, solver=solver, dual_ascent=True)
        _record(result, "sp_da", sol.objective, time.perf_counter() - t0, opt,
                cgap=sol.gap)

        # ---- steinerpy heuristic-only (exact=False): dual-ascent primal, no ILP ----
        t0 = time.perf_counter()
        sol = SteinerProblem(graph.copy(), [terminals], preprocess=True).get_solution(
            exact=False)
        _record(result, "sp_heur", sol.objective, time.perf_counter() - t0, opt,
                cgap=sol.gap)
    except Exception as exc:  # never let one instance abort the whole sweep
        result["status"] = f"error:{type(exc).__name__}: {exc}"

    # If no published optimum, fall back to the best exact objective found as a
    # reference, and (re)compute every method's gap against it.
    if opt is None and result["status"] == "ok":
        candidates = [result.get("sp_exact_obj"), result.get("sp_da_obj")]
        ref = min([c for c in candidates if c is not None], default=None)
        if ref is not None:
            result["opt"] = ref
            for key, _ in METHODS:
                result[f"{key}_gap"] = gap_vs_opt(result.get(f"{key}_obj"), ref)

    return result


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def _fmt_time(v):
    return f"{v:>8.3f}" if isinstance(v, (int, float)) else f"{'-':>8}"


def _fmt_pct(v):
    if not isinstance(v, (int, float)):
        return f"{'-':>6}"
    p = v * 100
    if abs(p) < 0.005:  # avoid printing "-0.00" from float rounding
        p = 0.0
    return f"{p:>6.2f}"


def _build_header():
    head = f"{'Instance':<10} {'|V|':>4} {'|E|':>5} {'k':>3} {'opt':>6} |"
    for _, hdr in METHODS:
        head += f" {hdr + ' t':>8} {'gap%':>6} |"
    head += f" {'heur cgap%':>10}"
    return head


def print_row(r):
    line = (
        f"{r['label']:<10} {r['n']:>4} {r['edges']:>5} {r['k']:>3} "
        f"{(r['opt'] if r['opt'] is not None else '-')!s:>6} |"
    )
    for key, _ in METHODS:
        line += f" {_fmt_time(r.get(f'{key}_time'))} {_fmt_pct(r.get(f'{key}_gap'))} |"
    # SteinerPy heuristic's certified optimality bound (networkx has none).
    line += f" {_fmt_pct(r.get('sp_heur_cgap')):>10}"
    if not r.get("opt_is_published", True):
        line += "  (opt=ref)"
    if r.get("status") != "ok":
        line += f"  [{r['status']}]"
    print(line)


def _avg(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="SteinerPy vs NetworkX on literature (SteinLib) instances.")
    ap.add_argument("--instances", default=DEFAULT_INSTANCES,
                    help="directory of .stp files (default: bundled B-series)")
    ap.add_argument("--optima", default=None,
                    help="optional instance,optimum CSV merged into the known optima")
    ap.add_argument("--time-limit", type=float, default=60.0,
                    help="seconds per SteinerPy solve (default: 60)")
    ap.add_argument("--solver", default="gurobi", choices=["highs", "gurobi"],
                    help="MIP solver for the exact runs (default: gurobi)")
    args = ap.parse_args(argv)

    optima = load_optima(args.optima)
    paths = sorted(
        os.path.join(args.instances, f)
        for f in os.listdir(args.instances) if f.lower().endswith(".stp")
    ) if os.path.isdir(args.instances) else []
    if not paths:
        ap.error(f"no .stp instances found in {args.instances!r}")

    header = _build_header()
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("SteinerPy vs NetworkX — Steiner Tree on literature instances")
    print("=" * len(header))
    print(f"\nInstances : {args.instances}  ({len(paths)} files)")
    print(f"Solver    : {args.solver}   time-limit: {args.time_limit:g}s/solve")
    print("""
Columns:
  NX-kou / NX-meh = networkx approximation.steiner_tree (2-approx, no bound)
  SP-exact        = steinerpy exact ILP, selected solver (no acceleration)
  SP+DA           = steinerpy exact ILP + dual-ascent accelerator
  SP-heur         = steinerpy get_solution(exact=False) (no ILP, certified gap)
  '<m> t'         = wall-clock seconds
  gap%            = (obj - opt) / opt  vs. the PUBLISHED optimum (lower is better)
  heur cgap%      = SP-heur's own CERTIFIED optimality gap (provably within this %);
                    networkx offers no such certificate.

SP-exact / SP+DA gap% should be ~0 (they return the published optimum). gap% is
measured against literature optima, not the solver's internal gap. '(opt=ref)'
marks an instance with no published optimum (gap is vs. the best exact value).
""")
    print(header)
    print(sep)

    all_results = []
    for path in paths:
        r = run_case(path, optima, args.time_limit, args.solver)
        all_results.append(r)
        print_row(r)

    print(sep)

    # ---- Summary ----
    print("\nSummary (averages over instances where the method ran):")
    print(f"  {'method':<10} {'avg time':>10} {'avg gap%':>9} {'#optimal':>9}")
    n_inst = len(all_results)
    for key, hdr in METHODS:
        avg_t = _avg(r.get(f"{key}_time") for r in all_results)
        avg_g = _avg(r.get(f"{key}_gap") for r in all_results)
        n_opt = sum(
            1 for r in all_results
            if isinstance(r.get(f"{key}_gap"), (int, float))
            and abs(r[f"{key}_gap"]) < 5e-5  # below the 0.01% table precision
        )
        avg_t_s = f"{avg_t:>9.3f}s" if avg_t is not None else f"{'-':>10}"
        avg_g_pct = 0.0 if avg_g is not None and abs(avg_g) < 5e-5 else avg_g
        avg_g_s = f"{avg_g_pct * 100:>8.2f}%" if avg_g is not None else f"{'-':>9}"
        print(f"  {hdr:<10} {avg_t_s} {avg_g_s} {f'{n_opt}/{n_inst}':>9}")

    # Exact-vs-DA speedup.
    speedups = [
        r["sp_exact_time"] / r["sp_da_time"]
        for r in all_results
        if isinstance(r.get("sp_exact_time"), (int, float))
        and isinstance(r.get("sp_da_time"), (int, float)) and r["sp_da_time"] > 0
    ]
    if speedups:
        print(f"\nAverage exact-vs-DA speedup (SP-exact / SP+DA): "
              f"{sum(speedups) / len(speedups):.2f}x")

    print("\nNotes:")
    print("  • SP-exact and SP+DA return the proven optimum (gap% ~ 0); they")
    print("    differ only in runtime — SP+DA fixes variables before the ILP.")
    print("  • NX-kou / NX-meh are fast but suboptimal and carry no bound.")
    print("  • SP-heur (exact=False) is ILP-free yet reports a CERTIFIED gap")
    print("    (heur cgap%): 0.00 == provably optimal, >0 bounds the distance.")


if __name__ == "__main__":
    main()
