import networkx as nx
from typing import Set, Union, List, Tuple, Dict


class ReductionTracker:
    """Track graph reductions to enable solution mapping back to original graph."""
    
    def __init__(self):
        # Store (removed_node, u, w, weight_uv, weight_vw, edge_id)
        # edge_id helps distinguish between original and contracted edges
        self.degree_two_contractions = []  
        self.degree_one_removals = []      # List of removed nodes (just for statistics)
        self._edge_counter = 0  # Unique ID for contracted edges
    
    def add_degree_two_contraction(self, node: str, u: str, w: str, weight_uv: float, weight_vw: float, edge_id: str):
        """Record a degree-2 node contraction that actually created/modified an edge."""
        self.degree_two_contractions.append((node, u, w, weight_uv, weight_vw, edge_id))
    
    def add_degree_one_removal(self, node: str):
        """Record a degree-1 node removal."""
        self.degree_one_removals.append(node)
    
    def get_next_edge_id(self) -> str:
        """Generate unique ID for contracted edges."""
        self._edge_counter += 1
        return f"contracted_edge_{self._edge_counter}"


def degree_one_reduction(G: nx.Graph, terminals: Set[any], 
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


def degree_two_reduction(G: nx.Graph, terminals: Set[any], weight: str = "weight",
                        tracker: ReductionTracker = None) -> nx.Graph:
    """
    Replace degree-2 non-terminal nodes with direct edges between their neighbors.
    Only record contractions that actually create or modify edges.
    """
    reduced_graph = G.copy()
    
    # Add metadata to track which edges are contracted vs original
    for u, v in reduced_graph.edges():
        if 'edge_type' not in reduced_graph[u][v]:
            reduced_graph[u][v]['edge_type'] = 'original'
    
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
                new_weight = weight_uv + weight_vw
                
                # Remove the degree-2 node
                reduced_graph.remove_node(node)
                
                # Check if edge (u,w) already exists
                if reduced_graph.has_edge(u, w):
                    # Edge already exists - only record contraction if we're improving the weight
                    existing_weight = reduced_graph[u][w].get(weight, float('inf'))
                    if new_weight < existing_weight:
                        # We're replacing the edge weight - record the contraction
                        edge_id = tracker.get_next_edge_id() if tracker else None
                        reduced_graph[u][w][weight] = new_weight
                        reduced_graph[u][w]['edge_type'] = 'contracted'
                        reduced_graph[u][w]['edge_id'] = edge_id
                        
                        if tracker:
                            tracker.add_degree_two_contraction(node, u, w, weight_uv, weight_vw, edge_id)
                    # If existing weight is better, don't record contraction since we're not using this path
                else:
                    # No existing edge - we're creating a new contracted edge
                    edge_id = tracker.get_next_edge_id() if tracker else None
                    reduced_graph.add_edge(u, w, **{
                        weight: new_weight,
                        'edge_type': 'contracted',
                        'edge_id': edge_id
                    })
                    
                    if tracker:
                        tracker.add_degree_two_contraction(node, u, w, weight_uv, weight_vw, edge_id)
    
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
        reduced_graph = degree_one_reduction(reduced_graph, all_terminals, tracker)
        
        # Apply degree-2 reduction
        reduced_graph = degree_two_reduction(reduced_graph, all_terminals, weight, tracker)
        
        # Check if any changes occurred
        if (reduced_graph.number_of_nodes() == initial_nodes and 
            reduced_graph.number_of_edges() == initial_edges):
            break
            
        iteration += 1
    
    return reduced_graph, tracker


def map_solution_to_original(reduced_solution_edges: List[Tuple[str, str]], 
                           tracker: ReductionTracker,
                           reduced_graph: nx.Graph) -> List[Tuple[str, str]]:
    """
    Map a solution from the reduced graph back to the original graph.
    Only expand edges that were actually created by contractions.
    
    Args:
        reduced_solution_edges: List of edges in the reduced graph solution
        tracker: ReductionTracker that recorded the reductions
        reduced_graph: The reduced graph (to check edge metadata)
        
    Returns:
        List of edges in the original graph that correspond to the solution
    """
    original_edges = []
    
    for edge in reduced_solution_edges:
        u, v = edge
        
        # Normalize edge direction to match graph storage
        if reduced_graph.has_edge(u, v):
            edge_data = reduced_graph[u][v]
        elif reduced_graph.has_edge(v, u):
            edge_data = reduced_graph[v][u]
            u, v = v, u  # Swap to match graph
        else:
            # Edge not found in reduced graph - this shouldn't happen
            original_edges.append(edge)
            continue
        
        # Check if this edge was created by contraction
        if edge_data.get('edge_type') == 'contracted':
            edge_id = edge_data.get('edge_id')
            
            # Find the corresponding contraction
            expanded = False
            for node, cu, cw, weight_uv, weight_vw, cid in tracker.degree_two_contractions:
                if cid == edge_id and ((cu == u and cw == v) or (cu == v and cw == u)):
                    # This edge was created by contracting 'node' - expand it
                    original_edges.extend([(u, node), (node, v)])
                    expanded = True
                    break
            
            if not expanded:
                # Edge ID not found in contractions - treat as original edge
                original_edges.append(edge)
        else:
            # Original edge - keep as is
            original_edges.append(edge)
    
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