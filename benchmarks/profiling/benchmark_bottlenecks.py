"""Isolate bottleneck-index construction, queries, and Python allocation peak."""
import argparse
import gc
import json
from pathlib import Path
import random
import statistics
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import networkx as nx
from steinerpy.graph_reducer import _bottleneck_from_mst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[512, 2048])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--queries', type=int, default=10000)
    args = parser.parse_args()
    for n in args.sizes:
        rng = random.Random(42)
        tree = nx.Graph()
        tree.add_nodes_from(range(n))
        for node in range(1, n):
            tree.add_edge(node, rng.randrange(node), weight=float(rng.randrange(1000)))
        pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(args.queries)]
        build_times, query_times = [], []
        checksum = None
        for _ in range(args.repeats):
            gc.collect()
            start = time.perf_counter()
            index = _bottleneck_from_mst(tree, tree.nodes)
            build_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            checksum = sum(index[a][b] for a, b in pairs)
            query_times.append(time.perf_counter() - start)
            del index
        gc.collect()
        # Separate memory pass: tracemalloc must not inflate the timing results.
        tracemalloc.start()
        index = _bottleneck_from_mst(tree, tree.nodes)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del index
        print(json.dumps(dict(terminals=n, repeats=args.repeats, queries=args.queries,
                              build_s=statistics.median(build_times),
                              query_s=statistics.median(query_times),
                              peak_python_bytes=peak, checksum=checksum)), flush=True)


if __name__ == '__main__':
    main()
