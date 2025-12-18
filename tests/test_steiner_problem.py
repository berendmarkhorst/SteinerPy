"""Basic tests for SteinerProblem class."""

import networkx as nx
import pytest
from src import SteinerProblem, Solution


def test_steiner_problem_initialization():
    """Test SteinerProblem initialization."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=2)
    G.add_edge('C', 'D', weight=1)
    
    terminal_groups = [['A', 'D']]
    
    problem = SteinerProblem(G, terminal_groups)
    
    assert problem.graph.number_of_nodes() == 4
    assert problem.graph.number_of_edges() == 3
    assert problem.terminal_groups == terminal_groups
    assert problem.weight == "weight"
    assert len(problem.edges) == 3
    assert len(problem.arcs) == 6  # bidirectional
    assert len(problem.nodes) == 4
    assert problem.steiner_points == {'B', 'C'}
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
    problem = SteinerProblem(G, terminal_groups)
    
    assert len(problem.terminal_groups) == 2
    assert problem.roots == ['A', 'D']
    assert problem.steiner_points == {'C'}


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


if __name__ == "__main__":
    pytest.main([__file__])
