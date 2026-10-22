"""The diagnostic harness must not double-count phases or overstate proofs."""

import pytest

from benchmarks.profile_composition import PhaseTimer, check_results, worker


def test_nested_phases_are_exclusive_even_on_failure():
    times = iter([0, 2, 5, 9])
    timer = PhaseTimer(clock=lambda: next(times))
    with pytest.raises(ValueError):
        with timer.measure("model_build"):
            with timer.measure("preprocessing"):
                raise ValueError("interrupted")
    assert timer.seconds["preprocessing"] == 3
    assert timer.seconds["model_build"] == 6
    assert sum(timer.seconds.values()) == 9
    assert not timer.stack


def test_instrumentation_restores_functions_and_consumes_preprocessing():
    import highspy
    import networkx as nx
    from steinerpy import Problem
    from steinerpy import _selection_model as selection

    original_run, original_filter = highspy.Highs.solve, selection.feasible_edges
    problem = Problem(nx.path_graph(3), terminals=[0, 2], node_costs={}, budget=1)
    timer = PhaseTimer()
    with pytest.raises(ValueError):
        with timer.instrument():
            assert highspy.Highs.solve is not original_run
            assert list(selection.feasible_edges(problem._spec)) == [(0, 1), (1, 2)]
            assert timer.calls["preprocessing"] == 1
            raise ValueError("interrupted")
    assert highspy.Highs.solve is original_run
    assert selection.feasible_edges is original_filter


def result(objective, status="optimal", total_cost=4, sense="min", coverage=False):
    return dict(
        case="tiny",
        objective=objective,
        status=status,
        total_cost=total_cost,
        objective_sense=sense,
        lexicographic=coverage,
    )


def test_proof_comparison_handles_partial_results_and_both_objective_directions():
    check_results([result(3), result(4, "feasible")])
    check_results([result(3, sense="max"), result(2, "feasible", sense="max")])
    with pytest.raises(AssertionError):
        check_results([result(3), result(2, "feasible")])
    with pytest.raises(AssertionError):
        check_results([result(3, sense="max"), result(4, "feasible", sense="max")])
    with pytest.raises(AssertionError):
        check_results([result(3), result(4)])


def test_coverage_proof_checks_secondary_cost():
    with pytest.raises(AssertionError):
        check_results(
            [result(3, coverage=True), result(3, total_cost=5, coverage=True)]
        )
    with pytest.raises(AssertionError, match="infeasibility"):
        check_results([result(3), dict(case="tiny", status="infeasible")])
    with pytest.raises(AssertionError):
        check_results(
            [
                result(3, coverage=True),
                result(3, "feasible", total_cost=3, coverage=True),
            ]
        )


def test_incompatible_cut_formulation_is_rejected():
    with pytest.raises(ValueError, match="every requested feature"):
        worker("forest", "cuts", 1)


def test_proved_infeasibility_is_distinguished_from_a_broken_run(monkeypatch):
    from steinerpy._feature_composition import FeatureComposition
    from steinerpy._selection_model import NoIncumbent
    import logging

    def infeasible(*args, **kwargs):
        raise NoIncumbent("infeasible")

    previous = logging.root.manager.disable
    monkeypatch.setattr(FeatureComposition, "get_solution", infeasible)
    row = worker("coverage-budget", "flow", 1)
    assert row["status"] == "infeasible"
    assert logging.root.manager.disable == previous


@pytest.mark.parametrize("config", ["flow", "flow-filtered", "cuts"])
def test_profiled_solve_validates_result_and_records_native_solver_calls(
    monkeypatch, config
):
    import networkx as nx
    from benchmarks import profile_composition as profile

    monkeypatch.setattr(
        profile,
        "make_case",
        lambda name: (
            nx.path_graph(3),
            dict(terminals=[0, 2], node_costs={1: 3}),
        ),
    )
    # Verify the worker requests one thread, but keep this shared test process
    # on the scheduler setting used by the rest of the suite.
    from steinerpy import mathematical_model as mm

    original_make_model = mm.make_model

    def make_model(time_limit, logfile="", threads=None):
        assert threads == 1
        return original_make_model(time_limit, logfile, threads=None)

    monkeypatch.setattr(mm, "make_model", make_model)
    row = worker("optional-costs", config, 2)
    assert row["status"] == "optimal", row
    assert row["objective"] == 5 and row["gap"] == 0
    assert row["backend_runs"] and row["phase_calls"]["optimization"] > 0
    assert all(v >= 0 for v in row["phases_seconds"].values())
    assert sum(row["phases_seconds"].values()) == pytest.approx(row["total_seconds"])
    assert row["phase_calls"].get("preprocessing", 0) == (config == "flow-filtered")
