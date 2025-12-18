import networkx as nx
import highspy as hp
from typing import List, Tuple
from .mathematical_model import build_model, run_model

class SteinerProblem:
    def __init__(self, graph: nx.Graph, terminal_groups: List[List], weight="weight"):
        """
        Initialize the SteinerProblem (can be tree or forest).
        :param graph: networkx graph.
        :param terminal_groups: nested list of terminals.
        :param weight: edge attribute specified by this string as the edge weight.
        """
        self.graph = graph
        self.terminal_groups = terminal_groups
        self.weight = weight
        self.edges = list(self.graph.edges())
        self.arcs = self.edges + [(v, u) for (u, v) in self.edges]
        self.nodes = list(self.graph.nodes())
        self.steiner_points = set(self.nodes) - set([t for group in terminal_groups for t in group])
        self.roots = [group[0] for group in self.terminal_groups]

    def __repr__(self):
        return f"Problem with a graph of {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges and {self.terminal_groups} as terminal groups."

    def get_solution(self, time_limit: float = 300, log_file: str = "") -> 'Solution':
        """
        Get the solution of the Steiner Problem using HighsPy.
        :param time_limit: time limit in seconds.
        :param log_file: path to the log file.
        :return: Solution object.
        """

        model, x, y1, y2, z, f = build_model(self, time_limit=time_limit, logfile=log_file)

        gap, runtime, objective, selected_edges = run_model(model, self, x)

        solution = Solution(gap, runtime, objective, selected_edges)

        return solution

class Solution:
    def __init__(self, gap: float, runtime: float, objective: float, selected_edges: List[Tuple]):
        self.gap = gap
        self.runtime = runtime
        self.objective = objective
        self.selected_edges = selected_edges



