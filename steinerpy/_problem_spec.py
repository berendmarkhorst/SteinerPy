"""Normalize public connectivity data without choosing a solver class."""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple

import networkx as nx

_FEATURE_OPTIONS = (
    "root",
    "weight",
    "node_costs",
    "node_weights",
    "node_prizes",
    "penalty_cost",
    "penalty_budget",
    "max_degree",
    "budget",
    "hop_limit",
    "node_budget",
    "partial_terminals",
    "points",
)


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
    """Original-space features, independent of a solver class or formulation.

    ``supplied_features`` preserves omitted versus explicitly supplied values,
    including ``None``. ``options`` contains preprocessing and other constructor
    options, including unknown names retained for dispatch-time validation.
    Graphs and feature values retain the legacy constructors' ownership rules;
    freezing this record does not freeze the caller's graph or cost mappings.
    """

    graph: Optional[nx.Graph]
    terminal_groups: Tuple[Tuple, ...] = ()
    groups: Tuple[Tuple, ...] = ()
    node_costs: Optional[Dict] = None
    weight: str = "weight"
    root: Any = None
    node_weights: Optional[Dict] = None
    node_prizes: Optional[Dict] = None
    penalty_cost: Optional[float] = None
    penalty_budget: Optional[float] = None
    max_degree: Optional[int] = None
    budget: Optional[float] = None
    hop_limit: Optional[int] = None
    node_budget: Optional[float] = None
    partial_terminals: Any = None
    points: Any = None
    requested_variant: Optional[str] = None
    supplied_features: FrozenSet[str] = frozenset()
    options: Dict[str, Any] = field(default_factory=dict)

    def implementation_options(self):
        """Recover supplied options without introducing constructor defaults."""
        options = dict(self.options)
        options.update((name, getattr(self, name)) for name in self.supplied_features)
        return options

    def validate_nodes(self):
        """Check connectivity requirements and roots before any transformation."""
        if self.graph is None:
            return
        for group in self.terminal_groups + self.groups:
            for node in group:
                if node not in self.graph:
                    raise ValueError(
                        "Required vertex {!r} is not in the graph.".format(node)
                    )
        if self.root is not None and self.root not in self.graph:
            raise ValueError("root {!r} is not in the graph.".format(self.root))


def normalize_problem(graph, terminal_groups, terminals, variant, options, variants):
    """Normalize public input without selecting or constructing an implementation."""
    if variant is not None and (
        not isinstance(variant, str) or variant not in variants
    ):
        raise ValueError(
            "Unknown variant {!r}; choose one of {}.".format(
                variant, ", ".join(variants)
            )
        )
    if terminals is not None and terminal_groups is not None:
        raise TypeError("Supply terminals or terminal_groups, not both.")
    if graph is not None and not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph.")
    if graph is None and variant != "rectilinear":
        raise TypeError("{} requires graph.".format(variant or "Problem"))

    normalized = terminal_sets(terminals) if terminals is not None else ()
    if terminal_groups is not None:
        # Legacy forest syntax accepts any iterable of terminal iterables.
        normalized = choice_sets(terminal_groups)
    options = dict(options)
    groups = choice_sets(options.pop("groups")) if "groups" in options else ()
    supplied = frozenset(options).intersection(_FEATURE_OPTIONS)
    features = {name: options.pop(name) for name in supplied}
    spec = ProblemSpec(
        graph=graph,
        terminal_groups=normalized,
        groups=groups,
        requested_variant=variant,
        supplied_features=supplied,
        options=options,
        **features,
    )
    spec.validate_nodes()
    return spec
