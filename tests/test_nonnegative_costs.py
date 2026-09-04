"""Non-negative edge-cost preconditions for exact Steiner algorithms."""

from types import SimpleNamespace

import networkx as nx
import pytest

from steinerpy import (
    BudgetedMaxWeightConnectedSubgraph,
    DirectedSteinerProblem,
    HopConstrainedSteinerProblem,
    MaxWeightConnectedSubgraph,
    NodeWeightedSteinerProblem,
    PartialTerminalSteinerProblem,
    PrizeCollectingProblem,
    SteinerProblem,
)
from steinerpy.dreyfus_wagner import dreyfus_wagner
from steinerpy.dual_ascent import dual_ascent
from steinerpy.pc_transform import (
    transform_directed_pcstp_to_sap,
    transform_mwcsp_to_pcstp,
    transform_pcstp_to_sap,
)


@pytest.mark.parametrize("preprocess", [False, True])
def test_disconnected_negative_cycle_is_rejected(preprocess):
    """A disconnected negative cycle must never lower a Steiner objective."""
    graph = nx.Graph()
    graph.add_edge("s", "m", weight=4)
    graph.add_edge("m", "t", weight=4)
    graph.add_edge("a", "b", weight=-3)
    graph.add_edge("b", "c", weight=-3)
    graph.add_edge("c", "a", weight=-3)

    with pytest.raises(ValueError, match="non-negative edge/arc"):
        SteinerProblem(graph, [["s", "t"]], preprocess=preprocess)


def test_directed_and_prize_collecting_negative_edges_are_rejected():
    directed = nx.DiGraph()
    directed.add_edge("r", "t", cost=-1)
    with pytest.raises(ValueError, match="attribute 'cost'"):
        DirectedSteinerProblem(directed, root="r", terminals=["t"], weight="cost")

    graph = nx.Graph()
    graph.add_edge(0, 1, weight=-1)
    with pytest.raises(ValueError, match="non-negative edge/arc"):
        PrizeCollectingProblem(graph, [[0]], {0: 1, 1: 1}, penalty_cost=0)


def test_transformed_variants_validate_before_dropping_or_changing_edges():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=-1)
    graph.add_edge(1, 2, weight=2)
    with pytest.raises(ValueError, match="non-negative"):
        PartialTerminalSteinerProblem(graph, [[0, 1, 2]], partial_terminals=[0])

    directed = nx.DiGraph()
    directed.add_edge("r", "t", weight=-1)
    with pytest.raises(ValueError, match="non-negative"):
        HopConstrainedSteinerProblem(directed, root="r", terminals=["t"], hop_limit=1)


def test_direct_algorithm_and_transform_entry_points_reject_negative_costs():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=-1)
    with pytest.raises(ValueError, match="non-negative"):
        dreyfus_wagner(graph, [0, 1])
    with pytest.raises(ValueError, match="non-negative"):
        transform_pcstp_to_sap(graph, {0: 1, 1: 1})

    view = SimpleNamespace(graph=graph, weight="weight")
    with pytest.raises(ValueError, match="non-negative"):
        dual_ascent(view)

    directed = nx.DiGraph()
    directed.add_edge("r", "t", weight=-1)
    with pytest.raises(ValueError, match="non-negative"):
        transform_directed_pcstp_to_sap(directed, {"r": 0, "t": 1}, root="r")


def test_zero_and_missing_edge_costs_remain_supported():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=0)
    graph.add_edge(1, 2)  # missing costs keep the historical default of 1

    problem = SteinerProblem(graph, [[0, 2]], preprocess=False)

    assert problem.graph[0][1]["weight"] == 0
    assert "weight" not in problem.graph[1][2]


def test_negative_node_weights_remain_supported_for_mwcs_variants():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 0, "weight")
    weights = {0: 5, 1: -2, 2: 4}

    mwcs = MaxWeightConnectedSubgraph(graph, weights, root=0)
    topology_only = graph.copy()
    nx.set_edge_attributes(topology_only, -99, "weight")
    budgeted = BudgetedMaxWeightConnectedSubgraph(
        topology_only,
        weights,
        {0: 0, 1: 1, 2: 1},
        node_budget=2,
        root=0,
    )

    assert mwcs._mwcs_node_weights[1] == -2
    assert budgeted._mwcs_node_weights[1] == -2


def test_all_positive_mwcs_transform_keeps_edge_costs_nonnegative():
    graph = nx.path_graph(3)
    transformed, prizes, constant = transform_mwcsp_to_pcstp(graph, {0: 5, 1: 2, 2: 4})

    assert set(nx.get_edge_attributes(transformed, "weight").values()) == {0}
    assert prizes == {0: 5, 1: 2, 2: 4}
    assert constant == 11


def test_negative_node_costs_are_rejected_for_node_weighted_steiner():
    graph = nx.path_graph(3)
    with pytest.raises(ValueError, match="non-negative node costs"):
        NodeWeightedSteinerProblem(graph, [[0, 2]], node_weights={0: 0, 1: -1, 2: 0})
