"""Normalize public connectivity data without choosing a solver class."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import networkx as nx


def terminal_sets(value, name="terminals"):
    """Normalize flat terminals or nested lists; tuples remain node labels."""
    values = list(value)
    if not values:
        raise ValueError("{} must not be empty.".format(name))
    nested = [isinstance(item, list) for item in values]
    if any(nested) and not all(nested):
        raise ValueError("{} cannot mix nodes and nested lists.".format(name))
    sets = values if all(nested) else [values]
    if any(not group for group in sets):
        raise ValueError("{} must not contain empty lists.".format(name))
    return tuple(tuple(dict.fromkeys(group)) for group in sets)


def choice_sets(value):
    """Normalize at-least-one groups, which always use a nested structure."""
    groups = []
    for group in value:
        if isinstance(group, (str, bytes)):
            raise TypeError("Groups must be nested collections of nodes, not strings.")
        groups.append(tuple(dict.fromkeys(group)))
    groups = tuple(groups)
    if not groups or any(not group for group in groups):
        raise ValueError("Each group must contain at least one vertex.")
    return groups


@dataclass(frozen=True)
class ProblemSpec:
    """Original-space connectivity and costs, independent of implementations."""

    graph: nx.Graph
    terminal_groups: Tuple[Tuple, ...] = ()
    groups: Tuple[Tuple, ...] = ()
    node_costs: Optional[Dict] = None
    weight: str = "weight"

    def validate_nodes(self):
        """Check every connectivity requirement before any transformation."""
        for group in self.terminal_groups + self.groups:
            for node in group:
                if node not in self.graph:
                    raise ValueError(
                        "Required vertex {!r} is not in the graph.".format(node)
                    )
