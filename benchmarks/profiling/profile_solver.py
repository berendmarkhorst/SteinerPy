"""Reproducible Gurobi timing and cProfile harness; run from repository root."""
import argparse
import cProfile
import functools
import importlib
import json
import logging
import os
import platform
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', nargs='+', default=['b18', 'c05', 'c15', 'd05', 'd15'])
    ap.add_argument('--configs', nargs='+', default=['default', 'accelerated'])
    ap.add_argument('--limit', type=float, default=30)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--serial', action='store_true')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--tag', default='timing')
    args = ap.parse_args()
    if args.serial:
        for key in ['STEINERPY_REDUCE_JOBS', 'STEINERPY_ASCENT_JOBS', 'STEINERPY_SEP_THREADS']:
            os.environ[key] = '1'
    import gurobipy as gp
    import scipy
    import networkx as nx
    from steinerpy import SteinerProblem
    from benchmarks.stp_parser import read_stp
    from benchmarks.optima import load_optima
    mm = importlib.import_module('steinerpy.mathematical_model')
    ob = importlib.import_module('steinerpy.objects')
    da = importlib.import_module('steinerpy.dual_ascent')
    logging.disable(logging.CRITICAL)
    out = ROOT / 'benchmarks' / 'profiling' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(args=vars(args), python=sys.version, platform=platform.platform(), cpus=os.cpu_count(), gurobi=gp.gurobi.version(), scipy=scipy.__version__, networkx=nx.__version__, env={k:v for k,v in os.environ.items() if k.startswith('STEINERPY_')})
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    totals, counts, models = {}, {}, []
    def instrument(module, name):
        original = getattr(module, name)
        @functools.wraps(original)
        def measured(*a, **kw):
            start = time.perf_counter()
            try:
                result = original(*a, **kw)
                if name == 'build_model_gurobi':
                    models.append(result[0])
                return result
            finally:
                totals[name] = totals.get(name, 0) + time.perf_counter() - start
                counts[name] = counts.get(name, 0) + 1
        setattr(module, name, measured)
        if getattr(ob, name, None) is original:
            setattr(ob, name, measured)
    for module, names in [(mm, ['build_model_gurobi', 'run_model_gurobi', 'find_violated_cuts_from_values']), (da, ['dual_ascent', 'reduced_cost_fixing', 'reduce_graph_with_dual_ascent'])]:
        for name in names:
            instrument(module, name)
    optima = load_optima()
    with (out / 'results.jsonl').open('w') as output:
        for instance in args.instances:
            graph, tg = read_stp(ROOT / 'benchmarks' / 'data' / instance[0].upper() / (instance + '.stp'))
            for config in args.configs:
                totals.clear(); counts.clear(); models.clear()
                row = dict(instance=instance, config=config, nodes=len(graph), edges=graph.number_of_edges(), terminals=len(tg[0]), optimum=optima[instance])
                profiler = cProfile.Profile()
                start = time.perf_counter()
                if args.profile:
                    profiler.enable()
                try:
                    problem = SteinerProblem(graph.copy(), [list(tg[0])], dual_ascent=config=='accelerated', da_reduce=config=='accelerated')
                    row['constructor_s'] = time.perf_counter() - start
                    row['reduced_nodes'] = len(problem.graph)
                    row['reduced_edges'] = problem.graph.number_of_edges()
                    sol = problem.get_solution(solver='gurobi', time_limit=args.limit, threads=args.threads)
                    row.update(objective=sol.objective, gap=sol.gap, matches_optimum=abs(sol.objective-optima[instance])<1e-6)
                except Exception as exc:
                    row['error'] = repr(exc)
                finally:
                    if args.profile:
                        profiler.disable()
                    row['wall_s'] = time.perf_counter() - start
                row['timers_s'] = dict(totals)
                row['calls'] = dict(counts)
                row['models'] = []
                for m in models:
                    row['models'].append({key:getattr(m, key) for key in ['Runtime', 'NodeCount', 'SolCount', 'Status', 'NumVars', 'NumConstrs', 'ObjBound']})
                if args.profile:
                    stem = out / (instance+'-'+config)
                    profiler.dump_stats(str(stem)+'.prof')
                    stats = pstats.Stats(profiler)
                    row['callback_s'] = sum(v[3] for k,v in stats.stats.items() if k[2]=='_cut_callback')
                    with open(str(stem)+'.txt', 'w') as f:
                        stats.stream=f
                        stats.sort_stats('cumulative').print_stats(55)
                        stats.sort_stats('tottime').print_stats(35)
                output.write(json.dumps(row)+'\n'); output.flush()
                print(json.dumps(row), flush=True)
                for m in models:
                    m.dispose()

if __name__ == '__main__':
    main()
