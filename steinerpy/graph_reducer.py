import networkx as nx
from typing import Set, Union, List, Tuple, Dict


class ReductionTracker:
    """Track graph reductions to enable solution mapping back to original graph."""
    
    def __init__(self):
        self.degree_two_contractions = []  # List of (removed_node, u, w, weight_uv, weight_vw)
        self.degree_one_removals = []      # List of removed nodes (just for statistics)
    
    def add_degree_two_contraction(self, node: str, u: str, w: str, weight_uv: float, weight_vw: float):
        """Record a degree-2 node contraction."""
        self.degree_two_contractions.append((node, u, w, weight_uv, weight_vw))
    
    def add_degree_one_removal(self, node: str):
        """Record a degree-1 node removal."""
        self.degree_one_removals.append(node)


def degree_one_reduction(G: nx.Graph, terminals: Set[str], weight: str = "weight", 
                        tracker: ReductionTracker = None) -> nx.Graph:
    """
    Iteratively remove degree-1 nodes that are not terminals.
    """
    reduced_graph = G.copy()
    
    changed = True
    while changed:
        changed = False
        nodes_to_remove = []
        
        for node in list(reduced_graph.nodes()):  # Convert to list to avoid iteration issues
            if reduced_graph.degree(node) == 1 and node not in terminals:
                nodes_to_remove.append(node)
        
        if nodes_to_remove:
            changed = True
            for node in nodes_to_remove:
                if tracker:
                    tracker.add_degree_one_removal(node)
                reduced_graph.remove_node(node)
    
    return reduced_graph


def degree_two_reduction(G: nx.Graph, terminals: Set[str], weight: str = "weight",
                        tracker: ReductionTracker = None) -> nx.Graph:
    """
    Replace degree-2 non-terminal nodes with direct edges between their neighbors.
    """
    reduced_graph = G.copy()
    
    changed = True
    while changed:
        changed = False
        nodes_to_contract = []
        
        # Find all degree-2 non-terminals in current graph state
        for node in list(reduced_graph.nodes()):  # Convert to list to avoid iteration issues
            if (reduced_graph.degree(node) == 2 and 
                node not in terminals and 
                reduced_graph.has_node(node)):  # Double-check node still exists
                
                neighbors = list(reduced_graph.neighbors(node))
                if len(neighbors) == 2:  # Safety check
                    nodes_to_contract.append((node, neighbors[0], neighbors[1]))
        
        # Contract all identified nodes
        for node, u, w in nodes_to_contract:
            # Check if node and neighbors still exist (might have been removed in previous contractions)
            if (reduced_graph.has_node(node) and 
                reduced_graph.has_node(u) and 
                reduced_graph.has_node(w) and
                reduced_graph.degree(node) == 2):  # Re-verify degree
                
                changed = True
                
                # Get weights of the two edges
                weight_uv = reduced_graph[u][node].get(weight, 1)
                weight_vw = reduced_graph[node][w].get(weight, 1)
                
                # Track the contraction
                if tracker:
                    tracker.add_degree_two_contraction(node, u, w, weight_uv, weight_vw)
                
                # Remove the degree-2 node
                reduced_graph.remove_node(node)
                
                # Add/update direct edge between neighbors
                if reduced_graph.has_edge(u, w):
                    # If edge already exists, keep the minimum weight
                    existing_weight = reduced_graph[u][w].get(weight, float('inf'))
                    new_weight = weight_uv + weight_vw
                    if new_weight < existing_weight:
                        reduced_graph[u][w][weight] = new_weight
                else:
                    # Add new edge
                    reduced_graph.add_edge(u, w, **{weight: weight_uv + weight_vw})
    
    return reduced_graph


def preprocess_graph(G: nx.Graph, terminal_groups: List[List[str]], weight: str = "weight") -> Tuple[nx.Graph, ReductionTracker]:
    """
    Apply both reductions and return the reduced graph and tracker.
    """
    # Flatten terminal groups to get all terminals
    all_terminals = set()
    for group in terminal_groups:
        all_terminals.update(group)
    
    reduced_graph = G.copy()
    tracker = ReductionTracker()
    
    # Keep applying reductions until no more changes occur
    max_iterations = len(G.nodes())  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        initial_nodes = reduced_graph.number_of_nodes()
        initial_edges = reduced_graph.number_of_edges()
        
        # Apply degree-1 reduction first
        reduced_graph = degree_one_reduction(reduced_graph, all_terminals, weight, tracker)
        
        # Apply degree-2 reduction
        reduced_graph = degree_two_reduction(reduced_graph, all_terminals, weight, tracker)
        
        # Check if any changes occurred
        if (reduced_graph.number_of_nodes() == initial_nodes and 
            reduced_graph.number_of_edges() == initial_edges):
            break
            
        iteration += 1
    
    return reduced_graph, tracker


def map_solution_to_original(reduced_solution_edges: List[Tuple[str, str]], 
                           tracker: ReductionTracker) -> List[Tuple[str, str]]:
    """
    Map a solution from the reduced graph back to the original graph.
    
    Args:
        reduced_solution_edges: List of edges in the reduced graph solution
        tracker: ReductionTracker that recorded the reductions
        
    Returns:
        List of edges in the original graph that correspond to the solution
    """
    original_edges = reduced_solution_edges.copy()
    
    # Process degree-2 contractions in reverse order
    for node, u, w, weight_uv, weight_vw in reversed(tracker.degree_two_contractions):
        # Check if the contracted edge (u,w) is in the solution
        edge_uw = None
        for edge in original_edges:
            if (edge == (u, w)) or (edge == (w, u)):
                edge_uw = edge
                break
        
        if edge_uw:
            # Remove the contracted edge and add the original path
            original_edges.remove(edge_uw)
            original_edges.extend([(u, node), (node, w)])
    
    return original_edges


def reduction_stats(original: nx.Graph, reduced: nx.Graph) -> dict:
    """Calculate statistics about the graph reduction."""
    return {
        "original_nodes": original.number_of_nodes(),
        "original_edges": original.number_of_edges(),
        "reduced_nodes": reduced.number_of_nodes(),
        "reduced_edges": reduced.number_of_edges(),
        "nodes_removed": original.number_of_nodes() - reduced.number_of_nodes(),
        "edges_removed": original.number_of_edges() - reduced.number_of_edges(),
        "node_reduction_percent": (1 - reduced.number_of_nodes() / max(1, original.number_of_nodes())) * 100,
        "edge_reduction_percent": (1 - reduced.number_of_edges() / max(1, original.number_of_edges())) * 100
    }