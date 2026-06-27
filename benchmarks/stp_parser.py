"""Parser for SteinLib / OR-Library ``.stp`` Steiner tree instances (STP 1.0).

The SteinLib format (https://steinlib.zib.de) looks like::

    33D32945 STP File, STP Format Version 1.0
    SECTION Graph
    Nodes 5
    Edges 6
    E 1 2 1
    E 2 3 2
    ...
    END
    SECTION Terminals
    Terminals 2
    T 1
    T 4
    END
    EOF

``E u v w`` are undirected edges; ``A u v w`` (in directed instances) are arcs.
A single terminal group is returned (SteinLib SPG instances are Steiner *trees*).
"""

from typing import Dict, List, Tuple

import networkx as nx


def read_stp(path: str, weight: str = "weight") -> Tuple[nx.Graph, List[List]]:
    """Parse a ``.stp`` file into a graph and terminal groups.

    :param path: path to the ``.stp`` instance.
    :param weight: edge-attribute name to store weights under.
    :return: ``(graph, terminal_groups)`` where ``graph`` is an ``nx.Graph``
        (or ``nx.DiGraph`` if the instance uses ``A`` arcs) and
        ``terminal_groups`` is ``[[t1, t2, ...]]`` (single group, first = root).
    """
    directed = False
    edges: List[Tuple[int, int, float]] = []
    terminals: List[int] = []
    section = None

    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            head = line.split()[0].upper()

            if head == "SECTION":
                section = line.split()[1].lower() if len(line.split()) > 1 else None
                continue
            if head == "END":
                section = None
                continue
            if head == "EOF":
                break

            if section == "graph":
                if head == "E":  # undirected edge
                    _, u, v, *rest = line.split()
                    w = float(rest[0]) if rest else 1.0
                    edges.append((int(u), int(v), w))
                elif head == "A":  # directed arc
                    directed = True
                    _, u, v, *rest = line.split()
                    w = float(rest[0]) if rest else 1.0
                    edges.append((int(u), int(v), w))
                # 'Nodes'/'Edges'/'Arcs' counters are ignored (derived from data)
            elif section == "terminals":
                if head == "T":
                    terminals.append(int(line.split()[1]))
                # SteinLib also has 'TP'/'Root' variants; 'Root r' sets the root
                elif head in ("ROOT", "ROOTP") and len(line.split()) > 1:
                    r = int(line.split()[1])
                    if r in terminals:
                        terminals.remove(r)
                    terminals.insert(0, r)

    graph = nx.DiGraph() if directed else nx.Graph()
    for u, v, w in edges:
        graph.add_edge(u, v, **{weight: w})

    if not terminals:
        raise ValueError(f"No terminals found in {path}")

    return graph, [terminals]


def read_pcspg(path: str, weight: str = "weight") -> Tuple[nx.Graph, Dict]:
    """Parse a PCSPG ``.stp`` prize-collecting Steiner instance (DIMACS 2014).

    The PCSPG format extends SteinLib: edges ``E u v c`` carry costs, and the
    ``SECTION Terminals`` lists node prizes as ``TP v p`` (vertices not listed
    have prize 0). The classic forgo-prize objective is
    ``sum_{e in T} c(e) + sum_{v not in T} p(v)``.

    :return: ``(graph, node_prizes)`` — an ``nx.Graph`` with edge costs under
        ``weight`` and a ``{node: prize}`` dict (prize nodes are guaranteed to be
        present in the graph even if they have no incident edges).
    """
    edges: List[Tuple[int, int, float]] = []
    prizes: Dict[int, float] = {}
    section = None
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            parts = raw.split()
            if not parts or parts[0].startswith("#"):
                continue
            head = parts[0].upper()
            if head == "SECTION":
                section = parts[1].lower() if len(parts) > 1 else None
                continue
            if head == "END":
                section = None
                continue
            if head == "EOF":
                break
            if section == "graph" and head == "E":
                w = float(parts[3]) if len(parts) > 3 else 1.0
                edges.append((int(parts[1]), int(parts[2]), w))
            elif section == "terminals" and head == "TP" and len(parts) > 2:
                prizes[int(parts[1])] = float(parts[2])

    graph = nx.Graph()
    for u, v, w in edges:
        graph.add_edge(u, v, **{weight: w})
    for v in prizes:  # keep isolated prize nodes
        graph.add_node(v)
    if not prizes:
        raise ValueError(f"No node prizes (TP lines) found in {path}")
    return graph, prizes


def instance_stats(graph: nx.Graph, terminal_groups: List[List]) -> dict:
    """Basic size statistics for reporting."""
    n_terms = sum(len(g) for g in terminal_groups)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "terminals": n_terms,
        "directed": isinstance(graph, nx.DiGraph),
    }
