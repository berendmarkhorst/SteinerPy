"""Shared-memory parallel helpers (thesis Ch. 6.3.1).

Implements the "collect-then-apply" parallelisation pattern used by the reduction
tests and the multi-root dual ascent: independent, read-only tests run in worker
processes (each worker receives the shared read-only payload **once** via an
initializer, so the heavy graph data is pickled per worker rather than per task),
their *candidates* are collected, and the caller applies them serially.

Everything degrades to a plain serial loop when parallelism is disabled or the
input is too small to amortise process-pool overhead, so small instances (and the
test-suite) are unaffected and deterministic.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List, Optional


def _env_jobs(var: str, default: int) -> int:
    val = os.environ.get(var)
    if val is not None:
        try:
            return max(1, int(val))
        except ValueError:
            pass
    return default


def cpu_jobs() -> int:
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:  # pragma: no cover
        return 1


def reduce_jobs() -> int:
    """Worker count for the parallel reduction tests (default: all cores)."""
    return _env_jobs("STEINERPY_REDUCE_JOBS", cpu_jobs())


def ascent_jobs() -> int:
    """Worker count for parallel multi-root dual ascent (default: 1 = off).

    Off by default: the serial multi-root ascent has an adaptive LB==UB
    early-exit that parallel execution forfeits, and pickling the graph to
    workers only pays off on very large instances.  Set ``STEINERPY_ASCENT_JOBS``
    to opt in.
    """
    return _env_jobs("STEINERPY_ASCENT_JOBS", 1)


# Per-worker read-only payload, populated by the pool initializer.
_SHARED = None


def _init_worker(shared) -> None:  # pragma: no cover - runs in child process
    global _SHARED
    _SHARED = shared


def get_shared():
    """Return the read-only payload set for this worker (or the serial run)."""
    return _SHARED


def pmap(fn: Callable, items: Iterable, jobs: int, shared,
         min_items: int, chunksize: Optional[int] = None) -> List:
    """Map ``fn`` over ``items`` (results in input order).

    Runs in a :class:`ProcessPoolExecutor` of ``jobs`` workers when ``jobs > 1``
    and there are at least ``min_items`` items; otherwise serial.  ``fn`` must be
    a top-level (picklable) function that reads the shared payload via
    :func:`get_shared`.
    """
    items = list(items)
    if jobs <= 1 or len(items) < min_items:
        _init_worker(shared)
        try:
            return [fn(it) for it in items]
        finally:
            _init_worker(None)
    if chunksize is None:
        chunksize = max(1, len(items) // (jobs * 4) or 1)
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                             initargs=(shared,)) as ex:
        return list(ex.map(fn, items, chunksize=chunksize))
