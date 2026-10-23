"""LP cut rounds must finish and preserve the subsequent MIP configuration."""

import os
from pathlib import Path
import subprocess
import sys

import highspy as hp
import networkx as nx
import pytest

from steinerpy import SteinerProblem
from steinerpy import mathematical_model as mm


@pytest.mark.parametrize("parallel", ["on", "off", "choose"])
@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("native_version", [(1, 12), (1, 14), (1, 15), (2, 0)])
def test_lp_configuration_is_restored_even_on_failure(
    monkeypatch, parallel, fails, native_version
):
    monkeypatch.setattr(hp, "HIGHS_VERSION_MAJOR", native_version[0])
    monkeypatch.setattr(hp, "HIGHS_VERSION_MINOR", native_version[1])
    model = mm.make_model(3, threads=2)
    model.setOptionValue("parallel", parallel)

    def minimize(objective):
        assert model.getOptionValue("parallel")[1] == (
            "choose" if parallel == "on" and native_version < (1, 15) else parallel
        )
        assert model.getOptionValue("threads")[1] == 2
        if fails:
            raise RuntimeError("interrupted LP")
        return hp.HighsStatus.kOk

    monkeypatch.setattr(model, "minimize", minimize)
    if fails:
        with pytest.raises(RuntimeError, match="interrupted LP"):
            mm._minimize_highs_lp(model, None)
    else:
        assert mm._minimize_highs_lp(model, None) == hp.HighsStatus.kOk
    assert model.getOptionValue("parallel")[1] == parallel
    assert model.getOptionValue("threads")[1] == 2


@pytest.mark.parametrize("sap", [False, True])
@pytest.mark.parametrize("lp_rounds", [0, 5])
def test_both_cut_loops_keep_integer_parallelism_and_exact_solution(
    monkeypatch, sap, lp_rounds
):
    monkeypatch.setenv("STEINERPY_LP_CUT_ROUNDS", str(lp_rounds))
    monkeypatch.delenv("STEINERPY_THREADS", raising=False)
    graph = nx.cycle_graph(4)
    nx.set_edge_attributes(graph, 1, "weight")
    if sap:
        graph = graph.to_directed()
    view = SteinerProblem(graph, [[0, 2]], preprocess=False)
    calls = []
    minimize = hp.Highs.minimize

    def record(model, *args, **kwargs):
        integer = any(
            t != hp.HighsVarType.kContinuous for t in model.getLp().integrality_
        )
        calls.append(integer)
        expect_parallel = integer or (
            hp.HIGHS_VERSION_MAJOR,
            hp.HIGHS_VERSION_MINOR,
        ) >= (1, 15)
        assert model.getOptionValue("parallel")[1] == (
            "on" if expect_parallel else "choose"
        )
        return minimize(model, *args, **kwargs)

    monkeypatch.setattr(hp.Highs, "minimize", record)
    if sap:
        gap, _, objective, edges = mm.solve_sap_highs(view, time_limit=3)
    else:
        model, x, _, y2, z = mm.build_model(view, time_limit=3)
        gap, _, objective, edges = mm.run_model(model, view, x, y2, z)
    assert gap == 0 and objective == 2
    selected = nx.DiGraph() if sap else nx.Graph()
    selected.add_edges_from(edges)
    assert nx.has_path(selected, 0, 2)
    assert sum(graph[u][v]["weight"] for u, v in edges) == 2
    assert True in calls and (False in calls) == (lp_rounds > 0)


@pytest.mark.parametrize("threads", [None, 1, 2])
def test_issue85_proves_infeasibility_within_a_bounded_fresh_process(threads):
    script = """
import networkx as nx
from steinerpy import Problem
graph = nx.cycle_graph(4)
graph.add_edge(0, 2)
nx.set_edge_attributes(graph, {e: i % 4 for i, e in enumerate(graph.edges)}, 'weight')
problem = Problem(graph, groups=[[0], [2], [3]],
                  node_costs={n: (n * 3) % 7 for n in graph}, max_degree=1)
try:
    problem.get_solution(time_limit=3, threads=THREADS)
except RuntimeError as error:
    assert 'instance is infeasible' in str(error), str(error)
else:
    raise AssertionError('Three connected nodes cannot all have degree <= 1')
""".replace("THREADS", repr(threads))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("STEINERPY_THREADS", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=12,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
