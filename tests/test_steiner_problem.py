"""Basic tests for SteinerProblem class."""

import networkx as nx
import pytest
from steinerpy import (
    SteinerProblem, Solution, PrizeCollectingProblem, PrizeCollectingSolution,
    NodeWeightedSteinerProblem, NodeWeightedSolution, MaxWeightConnectedSubgraph,
    DegreeConstrainedSteinerProblem,
    BudgetConstrainedSteinerProblem, BudgetSolution,
    DirectedSteinerProblem,
)


def test_steiner_problem_initialization():
    """Test SteinerProblem initialization."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=2)
    G.add_edge('C', 'D', weight=1)
    
    terminal_groups = [['A', 'D']]
    
    problem = SteinerProblem(G, terminal_groups, preprocess=True)

    assert problem.original_graph.number_of_nodes() == 4
    assert problem.original_graph.number_of_edges() == 3
    assert problem.graph.number_of_nodes() == 2
    assert problem.graph.number_of_edges() == 1
    assert problem.terminal_groups == terminal_groups
    assert problem.weight == "weight"
    assert len(problem.edges) == 1
    assert len(problem.arcs) == 2  # bidirectional
    assert len(problem.nodes) == 2
    assert problem.steiner_points == set()
    assert problem.roots == ['A']

    problem = SteinerProblem(G, terminal_groups, preprocess=False)

    assert problem.original_graph.number_of_nodes() == 4
    assert problem.original_graph.number_of_edges() == 3
    assert problem.graph.number_of_nodes() == 4
    assert problem.graph.number_of_edges() == 3
    assert problem.terminal_groups == terminal_groups
    assert problem.weight == "weight"
    assert len(problem.edges) == 3
    assert len(problem.arcs) == 6  # bidirectional
    assert len(problem.nodes) == 4
    assert problem.steiner_points == {"B", "C"}
    assert problem.roots == ['A']


def test_steiner_problem_repr():
    """Test SteinerProblem string representation."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    
    terminal_groups = [['A', 'B']]
    problem = SteinerProblem(G, terminal_groups)
    
    repr_str = repr(problem)
    assert "2 nodes" in repr_str
    assert "1 edges" in repr_str
    assert str(terminal_groups) in repr_str


def test_solution_initialization():
    """Test Solution initialization."""
    gap = 0.01
    runtime = 1.5
    objective = 10.5
    selected_edges = [('A', 'B'), ('B', 'C')]
    
    solution = Solution(gap, runtime, objective, selected_edges)
    
    assert solution.gap == gap
    assert solution.runtime == runtime
    assert solution.objective == objective
    assert solution.selected_edges == selected_edges


def test_steiner_problem_with_custom_weight():
    """Test SteinerProblem with custom weight attribute."""
    G = nx.Graph()
    G.add_edge('A', 'B', cost=5)
    
    terminal_groups = [['A', 'B']]
    problem = SteinerProblem(G, terminal_groups, weight="cost")
    
    assert problem.weight == "cost"


def test_steiner_problem_multiple_terminal_groups():
    """Test SteinerProblem with multiple terminal groups."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)
    G.add_edge('C', 'D', weight=1)
    G.add_edge('D', 'E', weight=1)
    
    terminal_groups = [['A', 'B'], ['D', 'E']]
    problem = SteinerProblem(G, terminal_groups, preprocess=True)
    
    assert len(problem.terminal_groups) == 2
    assert problem.roots == ['A', 'D']
    assert problem.steiner_points == set()

    problem = SteinerProblem(G, terminal_groups, preprocess=False)
    
    assert len(problem.terminal_groups) == 2
    assert problem.roots == ['A', 'D']
    assert problem.steiner_points == {"C"}


def test_steiner_tree_example_from_notebook():
    """Test the first example from example.ipynb - Steiner Tree connecting A, B, D."""
    # Create the same graph as in the notebook example
    graph = nx.Graph()
    edges = [("A", "C"), ("A", "D"), ("B", "C"), ("C", "D")]
    weights = [1, 10, 1, 1]
    
    for i, edge in enumerate(edges):
        graph.add_edge(edge[0], edge[1])
        graph.edges[edge]["weight"] = weights[i]
    
    # Create SteinerProblem to connect terminals A, B, D
    problem = SteinerProblem(graph, [["A", "B", "D"]])
    
    # Verify problem setup
    assert problem.graph.number_of_nodes() == 4
    assert problem.graph.number_of_edges() == 4
    assert problem.terminal_groups == [["A", "B", "D"]]
    assert problem.steiner_points == {'C'}  # C is the only Steiner point
    
    # Solve the problem
    solution = problem.get_solution(time_limit=30)
    
    # Verify solution properties
    assert isinstance(solution, Solution)
    assert solution.objective is not None
    assert solution.runtime is not None
    assert solution.gap is not None
    assert isinstance(solution.selected_edges, list)
    
    # The optimal solution should have cost 3 (AC=1, BC=1, CD=1)
    # allowing for small numerical tolerance
    assert abs(solution.objective - 3.0) < 1e-6, f"Expected objective ~3.0, got {solution.objective}"
    
    # The optimal solution should use exactly 3 edges
    assert len(solution.selected_edges) == 3, f"Expected 3 edges, got {len(solution.selected_edges)}"
    
    # Convert selected edges to a set of undirected edges for easier comparison
    selected_undirected = set()
    for edge in solution.selected_edges:
        # Normalize edge direction (smaller node first)
        normalized_edge = tuple(sorted(edge))
        selected_undirected.add(normalized_edge)
    
    # The expected optimal edges are AC, BC, CD
    expected_edges = {('A', 'C'), ('B', 'C'), ('C', 'D')}
    assert selected_undirected == expected_edges, f"Expected edges {expected_edges}, got {selected_undirected}"

def test_prize_collecting_problem_initialization():
    """Test PrizeCollectingProblem initialization."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=2)
    G.add_edge('B', 'C', weight=3)
    G.add_edge('A', 'C', weight=5)
    
    node_prizes = {'A': 10, 'B': 15, 'C': 8}
    terminal_groups = [['A', 'C']]
    
    problem = PrizeCollectingProblem(
        graph=G,
        terminal_groups=terminal_groups,
        node_prizes=node_prizes,
        penalty_cost=20,
        penalty_budget=10,
        preprocess=False
    )
    
    # Test base attributes
    assert problem.graph.number_of_nodes() == 3
    assert problem.graph.number_of_edges() == 3
    assert problem.terminal_groups == terminal_groups
    assert problem.weight == "weight"
    assert problem.steiner_points == {'B'}
    
    # Test prize collecting specific attributes
    assert problem.node_prizes == node_prizes
    assert problem.penalty_cost == 20
    assert problem.penalty_budget == 10


def test_prize_collecting_problem_with_preprocessing():
    """Test PrizeCollectingProblem with preprocessing enabled."""
    # Create a graph with degree-2 nodes that can be reduced
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=2)  # B can be contracted if it's not a terminal
    G.add_edge('C', 'D', weight=1)
    
    node_prizes = {'A': 10, 'B': 5, 'C': 3, 'D': 8}
    terminal_groups = [['A', 'D']]  # B and C are not terminals
    
    problem = PrizeCollectingProblem(
        graph=G,
        terminal_groups=terminal_groups,
        node_prizes=node_prizes,
        preprocess=True
    )
    
    # Should inherit preprocessing behavior from SteinerProblem
    assert problem.original_graph.number_of_nodes() == 4
    assert problem.preprocess == True


def test_prize_collecting_solution_properties():
    """Test PrizeCollectingSolution properties and methods."""
    selected_edges = [('A', 'B'), ('B', 'C')]
    original_selected_edges = [('A', 'X'), ('X', 'B'), ('B', 'C')]
    selected_nodes = ['A', 'B', 'C']
    penalties = {'group_0_D': 5.0}
    
    solution = PrizeCollectingSolution(
        gap=0.0,
        runtime=1.5,
        objective=25.0,
        selected_edges=selected_edges,
        original_selected_edges=original_selected_edges,
        selected_nodes=selected_nodes,
        penalties=penalties,
        total_prize=30.0,
        edge_cost=3.0,
        was_preprocessed=True
    )
    
    # Test inherited properties
    assert solution.gap == 0.0
    assert solution.runtime == 1.5
    assert solution.objective == 25.0
    assert solution.edges == original_selected_edges  # Should return original edges
    assert solution.was_preprocessed == True
    
    # Test prize collecting specific properties
    assert solution.selected_nodes == selected_nodes
    assert solution.penalties == penalties
    assert solution.total_prize == 30.0
    assert solution.edge_cost == 3.0
    assert solution.net_value == 22.0  # 30.0 - 3.0 edge costs - 5.0 penalties


def test_prize_collecting_simple_example():
    """Test a simple prize collecting example."""
    # Create a simple triangle graph
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)
    G.add_edge('A', 'C', weight=3)  # More expensive direct path
    
    # Set prizes - make B valuable enough to include
    node_prizes = {'A': 0, 'B': 10, 'C': 0}  # Only B has a prize
    terminal_groups = [['A', 'C']]  # Must connect A to C
    
    problem = PrizeCollectingProblem(
        graph=G,
        terminal_groups=terminal_groups,
        node_prizes=node_prizes,
        penalty_cost=100,  # High penalty to encourage connection
        preprocess=False
    )
    
    # Verify problem setup
    assert set(problem.nodes) == {'A', 'B', 'C'}
    assert len(problem.edges) == 3
    assert problem.steiner_points == {'B'}
    
    # Test solution (would need actual solver implementation)
    # For now, just test that get_solution method exists and has correct signature
    assert hasattr(problem, 'get_solution')


def test_prize_collecting_multiple_terminal_groups():
    """Test Prize Collecting with multiple terminal groups (forest)."""
    G = nx.Graph()
    # Create two disconnected components
    G.add_edge('A1', 'B1', weight=2)
    G.add_edge('A2', 'B2', weight=3)
    
    node_prizes = {'A1': 5, 'B1': 8, 'A2': 6, 'B2': 7}
    terminal_groups = [['A1', 'B1'], ['A2', 'B2']]  # Two separate groups
    
    problem = PrizeCollectingProblem(
        graph=G,
        terminal_groups=terminal_groups,
        node_prizes=node_prizes,
        penalty_cost=15,
        preprocess=False
    )
    
    assert len(problem.terminal_groups) == 2
    assert problem.roots == ['A1', 'A2']
    assert problem.steiner_points == set()  # All nodes are terminals


def test_prize_collecting_with_budget_constraint():
    """Test Prize Collecting with penalty budget constraint."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=10)  # Expensive edge
    
    node_prizes = {'A': 2, 'B': 3, 'C': 15}  # C has high prize but expensive to reach
    terminal_groups = [['A', 'C']]
    
    problem = PrizeCollectingProblem(
        graph=G,
        terminal_groups=terminal_groups,
        node_prizes=node_prizes,
        penalty_cost=5,
        penalty_budget=3,  # Low budget - may need to pay penalties
        preprocess=False
    )
    
    assert problem.penalty_budget == 3


def test_prize_collecting_solution_repr():
    """Test string representation of PrizeCollectingSolution."""
    solution = PrizeCollectingSolution(
        gap=0.1,
        runtime=2.0,
        objective=15.5,
        selected_edges=[('A', 'B')],
        selected_nodes=['A', 'B', 'C'],
        penalties={},
        total_prize=25.0
    )
    
    repr_str = repr(solution)
    assert "PrizeCollectingSolution" in repr_str
    assert "objective=15.50" in repr_str
    assert "prizes=25.00" in repr_str
    assert "nodes=3" in repr_str
    assert "edges=1" in repr_str


def test_prize_collecting_inheritance():
    """Test that PrizeCollectingProblem properly inherits from SteinerProblem."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    
    node_prizes = {'A': 5, 'B': 8}
    terminal_groups = [['A', 'B']]
    
    problem = PrizeCollectingProblem(G, terminal_groups, node_prizes)
    
    # Should inherit all SteinerProblem attributes
    assert hasattr(problem, 'graph')
    assert hasattr(problem, 'terminal_groups')
    assert hasattr(problem, 'weight')
    assert hasattr(problem, 'edges')
    assert hasattr(problem, 'arcs')
    assert hasattr(problem, 'nodes')
    assert hasattr(problem, 'steiner_points')
    assert hasattr(problem, 'roots')
    
    # Should have prize collecting specific attributes
    assert hasattr(problem, 'node_prizes')
    assert hasattr(problem, 'penalty_cost')
    assert hasattr(problem, 'penalty_budget')


def test_prize_collecting_empty_prizes():
    """Test Prize Collecting with empty or no prizes."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=2)
    
    # Test with empty prizes dict
    problem1 = PrizeCollectingProblem(G, [['A', 'B']], {})
    assert problem1.node_prizes == {}
    
    # Test with None prizes for some nodes
    problem2 = PrizeCollectingProblem(G, [['A', 'B']], {'A': 5})
    assert problem2.node_prizes.get('B', 0) == 0


def test_prize_collecting_custom_weight_attribute():
    """Test Prize Collecting with custom weight attribute."""
    G = nx.Graph()
    G.add_edge('A', 'B', cost=3, weight=1)  # Different weight attributes
    
    node_prizes = {'A': 4, 'B': 6}
    
    problem = PrizeCollectingProblem(
        graph=G,
        terminal_groups=[['A', 'B']],
        node_prizes=node_prizes,
        weight="cost"  # Use 'cost' instead of 'weight'
    )
    
    assert problem.weight == "cost"
    # Edge weight should be 3 (cost attribute), not 1 (weight attribute)
    assert G.edges[('A', 'B')][problem.weight] == 3

def test_prize_collecting_get_solution_integration():
    """Integration test: run get_solution end-to-end and verify objective, selected nodes, and penalties."""
    # Chain graph: A --1-- B --1-- C
    # Terminals: [['A', 'C']] — A is root, C is the non-root terminal
    # B is a Steiner point with prize 5
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)

    problem = PrizeCollectingProblem(
        graph=G,
        terminal_groups=[['A', 'C']],
        node_prizes={'A': 0, 'B': 5, 'C': 0},
        penalty_cost=100,
        preprocess=False,
    )

    solution = problem.get_solution(time_limit=30)

    assert isinstance(solution, PrizeCollectingSolution)

    # Both edges must be selected to connect A to C through B
    selected_undirected = {tuple(sorted(e)) for e in solution.selected_edges}
    assert selected_undirected == {('A', 'B'), ('B', 'C')}

    # B is a Steiner point with a positive prize — should be selected
    assert 'B' in solution.selected_nodes

    # Total prize collected from B
    assert solution.total_prize == 5.0

    # In a correct Prize Collecting formulation:
    # - Root terminal A should NOT incur a penalty (it's the source)
    # - Only non-root terminals incur penalties if not connected
    # - Since both A and C are connected via the path A-B-C, no penalties
    assert len(solution.penalties) == 0, f"Expected no penalties, got {solution.penalties}"

    # Objective = edge costs (2) - prize (5) + penalties (0) = -3
    # This means we have a net profit of 3 (prize exceeds edge costs)
    assert abs(solution.objective - (-3.0)) < 1e-4, f"Expected objective -3.0, got {solution.objective}"

    # Solution is solved to optimality on this small instance
    assert solution.gap < 1e-4


def test_prize_collecting_get_solution_with_budget():
    """Integration test: verify penalty_budget constraint limits the number of penalties."""
    # Chain graph: A --1-- B --1-- C
    # With penalty_budget=1, at most 1 terminal penalty variable can be active
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)

    problem = PrizeCollectingProblem(
        graph=G,
        terminal_groups=[['A', 'C']],
        node_prizes={'A': 0, 'B': 5, 'C': 0},
        penalty_cost=100,
        penalty_budget=1,
        preprocess=False,
    )

    solution = problem.get_solution(time_limit=30)

    assert isinstance(solution, PrizeCollectingSolution)

    # Budget of 1 allows at most 1 terminal to be penalized
    assert len(solution.penalties) <= 1

    # Same edges and prize as the unconstrained case
    selected_undirected = {tuple(sorted(e)) for e in solution.selected_edges}
    assert selected_undirected == {('A', 'B'), ('B', 'C')}
    assert solution.total_prize == 5.0


# ---------------------------------------------------------------------------
# Node-Weighted Steiner Tree (NWST) tests
# ---------------------------------------------------------------------------

def test_node_weighted_steiner_initialization():
    """Test NodeWeightedSteinerProblem initialization."""
    G = nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')

    node_weights = {'A': 1, 'B': 5, 'C': 1}
    problem = NodeWeightedSteinerProblem(G, [['A', 'C']], node_weights)

    assert problem.node_weights == node_weights
    assert problem.original_terminal_groups_nw == [['A', 'C']]
    # Transformed graph is a DiGraph with split nodes
    assert isinstance(problem.graph, nx.DiGraph)
    # Each original node becomes v_in and v_out → 2 * n nodes
    assert problem.graph.number_of_nodes() == 2 * G.number_of_nodes()


def test_node_weighted_steiner_repr():
    """Test NodeWeightedSteinerProblem and NodeWeightedSolution repr."""
    G = nx.Graph()
    G.add_edge('A', 'B')
    problem = NodeWeightedSteinerProblem(G, [['A', 'B']], {'A': 1, 'B': 2})
    assert 'nodes' in repr(problem)

    sol = NodeWeightedSolution(
        gap=0.0, runtime=0.1, objective=3.0,
        selected_edges=[('A', 'B')], selected_nodes=['A', 'B']
    )
    assert 'NodeWeightedSolution' in repr(sol)
    assert 'objective=3.00' in repr(sol)


def test_node_weighted_steiner_direct_path():
    """Triangle graph: direct cheap path preferred over expensive node."""
    G = nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    G.add_edge('A', 'C')

    # B is expensive (10); optimal solution skips B and goes direct A-C.
    solution = NodeWeightedSteinerProblem(
        G, [['A', 'C']], {'A': 1, 'B': 10, 'C': 1}
    ).get_solution(time_limit=30)

    assert isinstance(solution, NodeWeightedSolution)
    # Objective = node_A + edge_AC(=0) + node_C = 1 + 0 + 1 = 2
    assert abs(solution.objective - 2.0) < 1e-4
    # Only edge A-C should appear
    edges_undirected = {tuple(sorted(e)) for e in solution.selected_edges}
    assert edges_undirected == {('A', 'C')}
    # Selected nodes should be exactly A and C (both terminals in the solution)
    assert set(solution.selected_nodes) == {'A', 'C'}


def test_node_weighted_steiner_chain():
    """Chain graph: must include intermediate node."""
    G = nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')

    solution = NodeWeightedSteinerProblem(
        G, [['A', 'C']], {'A': 1, 'B': 2, 'C': 1}
    ).get_solution(time_limit=30)

    # Must go through B: total = node_A + edge_AB(0) + node_B + edge_BC(0) + node_C = 1+2+1 = 4
    assert abs(solution.objective - 4.0) < 1e-4
    edges_undirected = {tuple(sorted(e)) for e in solution.selected_edges}
    assert edges_undirected == {('A', 'B'), ('B', 'C')}
    assert set(solution.selected_nodes) == {'A', 'B', 'C'}


def test_node_weighted_steiner_with_edge_weights():
    """Node and edge weights together determine optimal path."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=3)  # Expensive edge
    G.add_edge('A', 'C', weight=1)  # Cheap edge

    # Via A-B edge (cost 3 + node_B cost 5): node_A(1) + edge_AB(3) + node_B(5) + node_C(1) = 10
    # Via A-C edge (cost 1): node_A(1) + edge_AC(1) + node_C(1) = 3  ← optimal
    solution = NodeWeightedSteinerProblem(
        G, [['A', 'C']], {'A': 1, 'B': 5, 'C': 1}
    ).get_solution(time_limit=30)

    assert abs(solution.objective - 3.0) < 1e-4


def test_node_weighted_steiner_custom_weight():
    """Test with a non-default edge weight attribute."""
    G = nx.Graph()
    G.add_edge('A', 'B', cost=2)

    problem = NodeWeightedSteinerProblem(G, [['A', 'B']], {'A': 1, 'B': 1}, weight='cost')
    assert problem.weight == 'cost'


# ---------------------------------------------------------------------------
# Maximum-Weight Connected Subgraph (MWCS) tests
# ---------------------------------------------------------------------------

def test_mwcs_initialization():
    """Test MaxWeightConnectedSubgraph initialization."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=0)
    G.add_edge('B', 'C', weight=0)

    node_weights = {'A': 3, 'B': 5, 'C': -2}
    problem = MaxWeightConnectedSubgraph(G, node_weights)

    # Root should be the highest-weight node (B)
    assert problem.mwcs_root == 'B'
    # Positive-weight nodes become terminals
    assert set(problem.terminal_groups[0]) >= {'A', 'B'}


def test_mwcs_explicit_root():
    """Test MaxWeightConnectedSubgraph with explicit root."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=0)

    problem = MaxWeightConnectedSubgraph(G, {'A': 1, 'B': 2}, root='A')
    assert problem.mwcs_root == 'A'


def test_mwcs_get_solution():
    """MWCS finds subgraph maximising total node weight."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=0)
    G.add_edge('B', 'C', weight=0)
    G.add_edge('C', 'D', weight=0)

    # D has strongly negative weight → should be excluded
    node_weights = {'A': 2, 'B': 3, 'C': 1, 'D': -10}
    solution = MaxWeightConnectedSubgraph(G, node_weights).get_solution(time_limit=30)

    assert isinstance(solution, PrizeCollectingSolution)
    # Nodes A, B, C should be selected; D should not appear
    assert 'D' not in solution.selected_nodes
    assert solution.total_prize >= 6.0  # At least A+B+C


# ---------------------------------------------------------------------------
# Degree-Constrained Steiner Tree tests
# ---------------------------------------------------------------------------

def test_degree_constrained_initialization():
    """Test DegreeConstrainedSteinerProblem initialization."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)

    problem = DegreeConstrainedSteinerProblem(G, [['A', 'C']], max_degree=2)
    assert problem.max_degree == 2
    assert problem.terminal_groups == [['A', 'C']]


def test_degree_constrained_repr():
    """Test DegreeConstrainedSteinerProblem repr (inherited)."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    problem = DegreeConstrainedSteinerProblem(G, [['A', 'B']], max_degree=1)
    assert 'nodes' in repr(problem)


def test_degree_constrained_solution_respects_degree():
    """Solution must not violate the degree constraint."""
    G = nx.Graph()
    for u, v, w in [('A', 'B', 1), ('B', 'C', 1), ('C', 'D', 1),
                    ('A', 'C', 1), ('B', 'D', 1)]:
        G.add_edge(u, v, weight=w)

    solution = DegreeConstrainedSteinerProblem(
        G, [['A', 'D']], max_degree=2
    ).get_solution(time_limit=30)

    assert isinstance(solution, Solution)
    assert solution.objective is not None
    # Verify every node in the solution has degree <= max_degree
    for node in ['A', 'B', 'C', 'D']:
        degree = sum(1 for e in solution.selected_edges if node in e)
        assert degree <= 2, f"Node {node} has degree {degree} > 2"


def test_degree_constrained_inherits_steiner():
    """DegreeConstrainedSteinerProblem is a SteinerProblem."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    problem = DegreeConstrainedSteinerProblem(G, [['A', 'B']], max_degree=3)
    assert isinstance(problem, SteinerProblem)
    assert hasattr(problem, 'max_degree')


# ---------------------------------------------------------------------------
# Budget-Constrained Steiner Tree tests
# ---------------------------------------------------------------------------

def test_budget_constrained_initialization():
    """Test BudgetConstrainedSteinerProblem initialization."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)

    problem = BudgetConstrainedSteinerProblem(G, [['A', 'B', 'C']], budget=1.5)
    assert problem.budget == 1.5
    assert problem.terminal_groups == [['A', 'B', 'C']]


def test_budget_solution_repr():
    """Test BudgetSolution repr."""
    sol = BudgetSolution(
        gap=0.0, runtime=0.5, objective=3,
        selected_edges=[('A', 'B'), ('B', 'C')],
        connected_terminals=3, total_terminals=4,
        penalties={'group_0_D': 1},
    )
    assert 'BudgetSolution' in repr(sol)
    assert '3/4' in repr(sol)


def test_budget_constrained_connects_within_budget():
    """With sufficient budget, all terminals should be connected."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)

    solution = BudgetConstrainedSteinerProblem(
        G, [['A', 'B', 'C']], budget=10.0  # Generous budget
    ).get_solution(time_limit=30)

    assert isinstance(solution, BudgetSolution)
    assert solution.connected_terminals == 3
    assert solution.total_terminals == 3
    assert len(solution.penalties) == 0


def test_budget_constrained_limited_budget():
    """With a tight budget, not all terminals can be connected."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)
    G.add_edge('C', 'D', weight=1)

    # Budget 2 can connect at most 3 out of 4 terminals in a chain
    solution = BudgetConstrainedSteinerProblem(
        G, [['A', 'B', 'C', 'D']], budget=2.0
    ).get_solution(time_limit=30)

    assert isinstance(solution, BudgetSolution)
    assert solution.connected_terminals == 3
    assert solution.total_terminals == 4
    # Total edge cost must not exceed budget
    total_cost = sum(G.edges[e]['weight'] for e in solution.selected_edges)
    assert total_cost <= 2.0 + 1e-6


def test_budget_constrained_zero_budget():
    """With zero budget no edges can be selected; only root is connected."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)

    solution = BudgetConstrainedSteinerProblem(
        G, [['A', 'B', 'C']], budget=0.0
    ).get_solution(time_limit=30)

    assert len(solution.selected_edges) == 0
    assert solution.connected_terminals == 1  # only root A is connected


def test_budget_constrained_inherits_steiner():
    """BudgetConstrainedSteinerProblem is a SteinerProblem."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    problem = BudgetConstrainedSteinerProblem(G, [['A', 'B']], budget=5.0)
    assert isinstance(problem, SteinerProblem)
    assert hasattr(problem, 'budget')


# ---------------------------------------------------------------------------
# Directed Steiner Tree (Arborescence) tests
# ---------------------------------------------------------------------------

def test_directed_steiner_initialization():
    """Test DirectedSteinerProblem initialization."""
    DG = nx.DiGraph()
    DG.add_edge('A', 'B', weight=1)
    DG.add_edge('B', 'C', weight=1)

    problem = DirectedSteinerProblem(DG, root='A', terminals=['B', 'C'])
    assert problem.terminal_groups == [['A', 'B', 'C']]
    assert problem.roots == ['A']
    # Arcs are one-directional for DiGraph
    assert set(problem.arcs) == {('A', 'B'), ('B', 'C')}
    assert problem.preprocess is False


def test_directed_steiner_requires_digraph():
    """DirectedSteinerProblem must reject undirected graphs."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    with pytest.raises(ValueError, match="directed"):
        DirectedSteinerProblem(G, root='A', terminals=['B'])


def test_directed_steiner_repr():
    """Test DirectedSteinerProblem repr (inherited)."""
    DG = nx.DiGraph()
    DG.add_edge('A', 'B', weight=1)
    problem = DirectedSteinerProblem(DG, root='A', terminals=['B'])
    assert 'nodes' in repr(problem)


def test_directed_steiner_prefers_directed_path():
    """Directed model picks the cheapest directed path, not the undirected shortest."""
    DG = nx.DiGraph()
    DG.add_edge('A', 'B', weight=1)
    DG.add_edge('B', 'C', weight=1)
    DG.add_edge('A', 'C', weight=10)  # Direct but expensive

    solution = DirectedSteinerProblem(
        DG, root='A', terminals=['B', 'C']
    ).get_solution(time_limit=30)

    assert isinstance(solution, Solution)
    # Optimal: A→B→C costs 2; direct A→C costs 10
    assert abs(solution.objective - 2.0) < 1e-4
    edges = {tuple(sorted(e)) for e in solution.selected_edges}
    assert edges == {('A', 'B'), ('B', 'C')}


def test_directed_steiner_no_reverse_path():
    """Directed model cannot use reverse arcs that are not in the DiGraph."""
    DG = nx.DiGraph()
    DG.add_edge('A', 'B', weight=1)  # Only A→B, not B→A
    DG.add_edge('A', 'C', weight=2)

    # From root A, we can reach B and C; solution must use only directed arcs
    solution = DirectedSteinerProblem(
        DG, root='A', terminals=['B', 'C']
    ).get_solution(time_limit=30)

    assert abs(solution.objective - 3.0) < 1e-4  # A→B (1) + A→C (2)
    for u, v in solution.selected_edges:
        assert DG.has_edge(u, v), f"Arc ({u},{v}) not in DiGraph"


def test_directed_steiner_inherits_base():
    """DirectedSteinerProblem is a BaseSteinerProblem."""
    from steinerpy.objects import BaseSteinerProblem
    DG = nx.DiGraph()
    DG.add_edge('A', 'B', weight=1)
    problem = DirectedSteinerProblem(DG, root='A', terminals=['B'])
    assert isinstance(problem, BaseSteinerProblem)


# ---------------------------------------------------------------------------
# Mix-and-match constraint modifier tests
# ---------------------------------------------------------------------------

def test_steiner_problem_with_max_degree_solution():
    """SteinerProblem accepts max_degree kwarg and enforces the degree constraint in the solution."""
    G = nx.Graph()
    for u, v, w in [('A', 'B', 1), ('B', 'C', 1), ('C', 'D', 1), ('A', 'C', 1), ('B', 'D', 1)]:
        G.add_edge(u, v, weight=w)

    problem = SteinerProblem(G, [['A', 'D']], max_degree=2, preprocess=False)
    assert problem.max_degree == 2

    solution = problem.get_solution(time_limit=30)

    assert isinstance(solution, Solution)
    for node in G.nodes():
        deg = sum(1 for e in solution.selected_edges if node in e)
        assert deg <= 2, f"Node {node} has degree {deg} > 2"


def test_steiner_problem_with_budget_kwarg():
    """SteinerProblem accepts budget as a kwarg modifier and returns BudgetSolution."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)
    G.add_edge('C', 'D', weight=1)

    problem = SteinerProblem(G, [['A', 'B', 'C', 'D']], budget=2.0, preprocess=False)
    assert problem.budget == 2.0

    solution = problem.get_solution(time_limit=30)
    assert isinstance(solution, BudgetSolution)
    # Total edge cost must not exceed budget
    total_cost = sum(G.edges[e]['weight'] for e in solution.selected_edges)
    assert total_cost <= 2.0 + 1e-6


def test_steiner_problem_with_both_modifiers():
    """SteinerProblem supports combining max_degree and budget simultaneously."""
    G = nx.Graph()
    for u, v, w in [('A', 'B', 1), ('B', 'C', 1), ('C', 'D', 1), ('A', 'C', 2), ('B', 'D', 2)]:
        G.add_edge(u, v, weight=w)

    # Budget of 2 and max_degree of 2
    solution = SteinerProblem(
        G, [['A', 'B', 'C', 'D']], max_degree=2, budget=2.0, preprocess=False
    ).get_solution(time_limit=30)

    assert isinstance(solution, BudgetSolution)
    total_cost = sum(G.edges[e]['weight'] for e in solution.selected_edges)
    assert total_cost <= 2.0 + 1e-6
    for node in G.nodes():
        deg = sum(1 for e in solution.selected_edges if node in e)
        assert deg <= 2, f"Node {node} has degree {deg} > 2"


def test_deprecated_degree_constrained_still_works():
    """DegreeConstrainedSteinerProblem still works but emits DeprecationWarning."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)

    with pytest.warns(DeprecationWarning):
        problem = DegreeConstrainedSteinerProblem(G, [['A', 'B']], max_degree=2)

    assert problem.max_degree == 2
    assert isinstance(problem, SteinerProblem)


def test_deprecated_budget_constrained_still_works():
    """BudgetConstrainedSteinerProblem still works but emits DeprecationWarning."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)

    with pytest.warns(DeprecationWarning):
        problem = BudgetConstrainedSteinerProblem(G, [['A', 'B']], budget=5.0)

    assert problem.budget == 5.0
    assert isinstance(problem, SteinerProblem)


def test_prize_collecting_with_max_degree_modifier():
    """PrizeCollectingProblem accepts max_degree as a kwarg modifier."""
    G = nx.Graph()
    for u, v, w in [('A', 'B', 1), ('B', 'C', 1), ('A', 'C', 1)]:
        G.add_edge(u, v, weight=w)

    problem = PrizeCollectingProblem(
        G, [['A', 'C']], node_prizes={'A': 5, 'B': 5, 'C': 5},
        max_degree=2, preprocess=False
    )
    assert problem.max_degree == 2

    solution = problem.get_solution(time_limit=30)
    assert isinstance(solution, PrizeCollectingSolution)
    for node in G.nodes():
        deg = sum(1 for e in solution.selected_edges if node in e)
        assert deg <= 2, f"Node {node} has degree {deg} > 2"


# ---------------------------------------------------------------------------
# Solver parameter tests
# ---------------------------------------------------------------------------

def test_invalid_solver_raises_value_error():
    """get_solution raises ValueError for an unknown solver name."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    problem = SteinerProblem(G, [['A', 'B']], preprocess=False)
    with pytest.raises(ValueError, match="solver"):
        problem.get_solution(solver="cplex")


def test_highs_solver_explicit():
    """Explicitly passing solver='highs' produces the same result as the default."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=2)
    G.add_edge('C', 'D', weight=1)

    problem = SteinerProblem(G, [['A', 'D']], preprocess=False)
    solution = problem.get_solution(time_limit=30, solver="highs")

    assert isinstance(solution, Solution)
    assert abs(solution.objective - 4.0) < 1e-4


def test_gurobi_solver_raises_import_error_when_not_installed():
    """get_solution raises ImportError when gurobipy is absent."""
    import sys
    import importlib

    gurobi_available = importlib.util.find_spec("gurobipy") is not None
    if gurobi_available:
        pytest.skip("gurobipy is installed; skipping import-error test.")

    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    problem = SteinerProblem(G, [['A', 'B']], preprocess=False)
    with pytest.raises(ImportError, match="gurobipy"):
        problem.get_solution(solver="gurobi")


def test_gurobi_solver_matches_highs():
    """When gurobipy is available and licensed, Gurobi and HiGHS give the same objective."""
    import importlib

    gurobi_available = importlib.util.find_spec("gurobipy") is not None
    if not gurobi_available:
        pytest.skip("gurobipy is not installed.")

    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        gp.Model(env=env).dispose()
        env.dispose()
    except Exception:
        pytest.skip("Gurobi license not available.")

    # Build a small graph
    G = nx.Graph()
    edges = [("A", "C"), ("A", "D"), ("B", "C"), ("C", "D")]
    weights = [1, 10, 1, 1]
    for edge, w in zip(edges, weights):
        G.add_edge(edge[0], edge[1], weight=w)

    problem = SteinerProblem(G, [["A", "B", "D"]], preprocess=False)

    sol_highs = problem.get_solution(time_limit=30, solver="highs")
    sol_gurobi = problem.get_solution(time_limit=30, solver="gurobi")

    assert abs(sol_highs.objective - sol_gurobi.objective) < 1e-4, (
        f"HiGHS objective {sol_highs.objective} != Gurobi objective {sol_gurobi.objective}"
    )


def test_gurobi_solver_node_weighted():
    """NodeWeightedSteinerProblem accepts solver parameter without error."""
    import importlib

    gurobi_available = importlib.util.find_spec("gurobipy") is not None
    if not gurobi_available:
        pytest.skip("gurobipy is not installed.")

    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        gp.Model(env=env).dispose()
        env.dispose()
    except Exception:
        pytest.skip("Gurobi license not available.")

    from steinerpy import NodeWeightedSteinerProblem, NodeWeightedSolution

    G = nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    node_weights = {'A': 1, 'B': 2, 'C': 1}

    problem = NodeWeightedSteinerProblem(G, [['A', 'C']], node_weights)
    solution = problem.get_solution(time_limit=30, solver="gurobi")
    assert isinstance(solution, NodeWeightedSolution)


if __name__ == "__main__":
    pytest.main([__file__])
