"""
benchmark_pcstp.py — SteinerPy vs pcst_fast on prize-collecting instances
=========================================================================

Run from the project root (env with steinerpy + pcst_fast installed):

    python benchmark_pcstp.py                              # bundled JMP set
    python benchmark_pcstp.py --instances path/to/PCSPG    # any .stp PCSPG dir
    python benchmark_pcstp.py --heuristics-only            # skip the exact ILP

What this compares
------------------
The Prize-Collecting Steiner Problem in Graphs (PCSPG / PCSTP): each edge has a
cost and each terminal a prize; minimise ``sum_{e in T} c(e) + sum_{v not in T}
p(v)`` (edge cost plus the prizes of terminals the tree does not span).

* steinerpy PrizeCollectingProblem, pc_transform=True  — exact (proven optimal)
* steinerpy PrizeCollectingProblem, exact=False        — heuristic (certified gap)
* pcst_fast (Hegde, Indyk & Schmidt 2014)              — Goemans-Williamson
  2-approximation, the de-facto fast PCST competitor (used in network biology /
  sparse approximation). Strong pruning. No optimality certificate.

Metrics per instance
---------------------
* Wall-clock solve time (seconds)
* PCSTP objective (computed identically for every method from its selected
  edges/nodes, so the three are directly comparable)
* Optimality gap vs. the exact optimum, ``(obj - opt) / opt``

Instances: DIMACS 2014 "JMP" set (Johnson, Minkoff & Phillips 1999), bundled
under benchmarks/data/PCSPG-JMP. The exact solver provides the reference optimum
(its own certified gap ~ 0 confirms optimality); pcst_fast and the steinerpy
heuristic are scored against it.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from steinerpy import PrizeCollectingProblem
from benchmarks.stp_parser import read_pcspg
import pcst_fast

DEFAULT_INSTANCES = os.path.join("benchmarks", "data", "PCSPG-JMP")
WEIGHT = "weight"


# ---------------------------------------------------------------------------
# Canonical PCSTP objective — applied uniformly to every method's structure
# ---------------------------------------------------------------------------

def pcstp_objective(graph, total_prize, prizes, sel_edges, sel_nodes) -> float:
    """edge cost of the tree + prizes of the terminals it does NOT span."""
    edge_cost = sum(graph[u][v][WEIGHT] for (u, v) in sel_edges)
    collected = sum(prizes.get(n, 0) for n in sel_nodes)
    return edge_cost + (total_prize - collected)


def gap_vs(obj, opt):
    if obj is None or opt is None or opt == 0:
        return None
    return (obj - opt) / opt


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def run_pcst_fast(graph, prizes, total_prize):
    """Goemans-Williamson PCST via pcst_fast (unrooted, strong pruning)."""
    nodes = sorted(graph.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edge_list = list(graph.edges())
    edges = np.array([[idx[u], idx[v]] for (u, v) in edge_list], dtype=np.int64)
    costs = np.array([graph[u][v][WEIGHT] for (u, v) in edge_list], dtype=np.float64)
    pr = np.zeros(len(nodes), dtype=np.float64)
    for n, p in prizes.items():
        pr[idx[n]] = p
    t0 = time.perf_counter()
    v_sel, e_sel = pcst_fast.pcst_fast(edges, pr, costs, -1, 1, "strong", 0)
    rt = time.perf_counter() - t0
    sel_edges = [edge_list[i] for i in e_sel]
    sel_nodes = [nodes[i] for i in v_sel]
    return pcstp_objective(graph, total_prize, prizes, sel_edges, sel_nodes), rt


def run_steinerpy(graph, prizes, root, total_prize, exact, time_limit, solver):
    t0 = time.perf_counter()
    prob = PrizeCollectingProblem(graph.copy(), [[root]], prizes,
                                  weight=WEIGHT, penalty_cost=0)
    if exact:
        sol = prob.get_solution(pc_transform=True, time_limit=time_limit, solver=solver)
    else:
        sol = prob.get_solution(exact=False)
    rt = time.perf_counter() - t0
    obj = pcstp_objective(graph, total_prize, prizes, sol.edges, sol.selected_nodes)
    return obj, rt, sol.gap


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(stp_path, time_limit, solver, heuristics_only):
    graph, prizes = read_pcspg(stp_path)
    name = os.path.splitext(os.path.basename(stp_path))[0]
    total_prize = sum(prizes.values())
    root = max(prizes, key=prizes.get)  # nominal root (transform builds its own)

    r = {"label": name, "n": graph.number_of_nodes(), "edges": graph.number_of_edges(),
         "p": len(prizes), "status": "ok", "opt": None, "opt_proven": False}
    try:
        if not heuristics_only:
            obj, rt, g = run_steinerpy(graph, prizes, root, total_prize,
                                       True, time_limit, solver)
            r.update(ex_obj=obj, ex_time=rt, ex_cgap=g)
            r["opt"] = obj
            r["opt_proven"] = g is not None and abs(g) < 1e-6

        obj, rt, g = run_steinerpy(graph, prizes, root, total_prize,
                                   False, time_limit, solver)
        r.update(he_obj=obj, he_time=rt, he_cgap=g)

        obj, rt = run_pcst_fast(graph, prizes, total_prize)
        r.update(pf_obj=obj, pf_time=rt)
    except Exception as exc:  # never let one instance abort the sweep
        r["status"] = f"error:{type(exc).__name__}: {exc}"

    # In --heuristics-only mode there is no exact optimum; use the best heuristic
    # objective as a (non-proven) reference so the two heuristics stay comparable.
    if r["opt"] is None and r["status"] == "ok":
        cands = [r.get("he_obj"), r.get("pf_obj")]
        r["opt"] = min([c for c in cands if c is not None], default=None)

    opt = r["opt"]
    r["he_gap"] = gap_vs(r.get("he_obj"), opt)
    r["pf_gap"] = gap_vs(r.get("pf_obj"), opt)
    r["ex_gap"] = gap_vs(r.get("ex_obj"), opt)  # ~0 by construction
    return r


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _t(v):
    return f"{v:>8.3f}" if isinstance(v, (int, float)) else f"{'-':>8}"


def _pct(v):
    if not isinstance(v, (int, float)):
        return f"{'-':>7}"
    p = v * 100
    if abs(p) < 0.005:
        p = 0.0
    return f"{p:>7.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="SteinerPy vs pcst_fast on PCSPG (prize-collecting) instances.")
    ap.add_argument("--instances", default=DEFAULT_INSTANCES,
                    help="directory of PCSPG .stp files (default: bundled JMP set)")
    ap.add_argument("--time-limit", type=float, default=60.0,
                    help="seconds per exact solve (default: 60)")
    ap.add_argument("--solver", default="gurobi", choices=["highs", "gurobi"],
                    help="MIP solver for the exact SAP transform (default: gurobi)")
    ap.add_argument("--heuristics-only", action="store_true",
                    help="skip the exact solve; compare SP-heur vs pcst_fast only")
    args = ap.parse_args(argv)

    paths = sorted(
        os.path.join(args.instances, f)
        for f in os.listdir(args.instances) if f.lower().endswith(".stp")
    ) if os.path.isdir(args.instances) else []
    if not paths:
        ap.error(f"no .stp instances found in {args.instances!r}")

    show_exact = not args.heuristics_only
    header = (f"{'Instance':<10} {'|V|':>4} {'|E|':>5} {'|P|':>4} {'opt':>9} |"
              + (f" {'SP-ex t':>8} {'gap%':>7} |" if show_exact else "")
              + f" {'SP-he t':>8} {'gap%':>7} {'cgap%':>7} |"
              + f" {'pcstf t':>8} {'gap%':>7}")
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("SteinerPy vs pcst_fast — Prize-Collecting Steiner (PCSPG)")
    print("=" * len(header))
    print(f"\nInstances : {args.instances}  ({len(paths)} files)")
    print(f"Mode      : {'heuristics only' if args.heuristics_only else 'heuristic + exact'}"
          f"   solver: {args.solver}   time-limit: {args.time_limit:g}s")
    print("""
Columns:
  SP-ex   = steinerpy exact (pc_transform), proven optimal (the reference)
  SP-he   = steinerpy heuristic (exact=False); cgap% = its CERTIFIED gap
  pcstf   = pcst_fast Goemans-Williamson 2-approx (no certificate)
  gap%    = (obj - opt) / opt vs. the exact optimum  |  '<m> t' = seconds
""")
    print(header)
    print(sep)

    results = []
    for path in paths:
        r = run_case(path, args.time_limit, args.solver, args.heuristics_only)
        results.append(r)
        opt = r["opt"]
        opt_s = (f"{opt:>9.0f}" if isinstance(opt, (int, float)) else f"{'-':>9}")
        if isinstance(opt, (int, float)) and not r["opt_proven"]:
            opt_s = opt_s[:-1] + "~"  # exact hit the time limit; reference only
        line = f"{r['label']:<10} {r['n']:>4} {r['edges']:>5} {r['p']:>4} {opt_s} |"
        if show_exact:
            line += f" {_t(r.get('ex_time'))} {_pct(r.get('ex_gap'))} |"
        line += f" {_t(r.get('he_time'))} {_pct(r.get('he_gap'))} {_pct(r.get('he_cgap'))} |"
        line += f" {_t(r.get('pf_time'))} {_pct(r.get('pf_gap'))}"
        if r["status"] != "ok":
            line += f"  [{r['status']}]"
        print(line)

    print(sep)

    # ---- Summary ----
    def avg(key):
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    def n_opt(key):
        return sum(1 for r in results
                   if isinstance(r.get(key), (int, float)) and abs(r[key]) < 5e-5)

    n = len(results)
    print("\nSummary (averages over solved instances):")
    print(f"  {'method':<10} {'avg time':>10} {'avg gap%':>9} {'#optimal':>9}")
    rows = [("SP-heur", "he_time", "he_gap"), ("pcst_fast", "pf_time", "pf_gap")]
    if show_exact:
        rows.insert(0, ("SP-exact", "ex_time", "ex_gap"))
    for label, tkey, gkey in rows:
        at, ag = avg(tkey), avg(gkey)
        at_s = f"{at:>9.3f}s" if at is not None else f"{'-':>10}"
        ag_s = f"{(0.0 if ag is not None and abs(ag) < 5e-5 else ag) * 100:>8.2f}%" \
            if ag is not None else f"{'-':>9}"
        print(f"  {label:<10} {at_s} {ag_s} {f'{n_opt(gkey)}/{n}':>9}")

    print("\nNotes:")
    if show_exact:
        print("  • SP-exact returns the PROVEN optimum (reference); '~' marks an")
        print("    instance where it hit the time limit (used as a best-known ref).")
    else:
        print("  • No exact run: gaps are vs. the best heuristic objective per")
        print("    instance (a non-proven reference), so the best method shows 0%.")
    print("  • pcst_fast is extremely fast but gives no optimality certificate.")
    print("  • SP-heur is ILP-free and reports a CERTIFIED gap (cgap%); its PCSTP")
    print("    primal (dual-ascent SAP) is weak on P-type instances — unlike the")
    print("    classic-Steiner heuristic, this path has no MST refinement yet.")


if __name__ == "__main__":
    main()
