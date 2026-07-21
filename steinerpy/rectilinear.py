"""
Rectilinear Steiner minimum tree helpers (Rehfeldt 2021, Ch. 5.4).

The rectilinear Steiner minimum tree (RSMT) of a set of points in the plane is
reduced *exactly* to a Steiner tree problem on the Hanan grid (Hanan 1966): the grid
nodes are all ``(x, y)`` where ``x`` is the x-coordinate of some input point and ``y``
is the y-coordinate of some input point, adjacent grid nodes are joined by edges
weighted with their L1 (Manhattan / rectilinear) length, and the input points are the
terminals.  A minimum Steiner tree of these terminals on the grid is an RSMT of the
points.
"""

from typing import List, Tuple

import networkx as nx


def hanan_grid(points: List[Tuple[float, float]], weight: str = "weight"):
    """
    Build the Hanan grid for a set of 2-D points.

    :param points: list of ``(x, y)`` coordinate tuples (duplicates are ignored).
    :param weight: edge-attribute name for edge weights.
    :return: ``(graph, terminal_nodes)`` where ``graph`` is an undirected
        :class:`networkx.Graph` whose nodes are ``(x, y)`` tuples and
        ``terminal_nodes`` is the list of input points (each a grid node).
    :raises ValueError: if no points are given.
    """
    # Preserve order, drop duplicates, normalise to float tuples.
    pts = list(dict.fromkeys((float(x), float(y)) for (x, y) in points))
    if not pts:
        raise ValueError("at least one point is required.")

    xs = sorted({x for (x, _) in pts})
    ys = sorted({y for (_, y) in pts})

    G = nx.Graph()
    for x in xs:
        for y in ys:
            G.add_node((x, y))

    # Horizontal edges between consecutive x-coordinates on each grid line y.
    for y in ys:
        for xa, xb in zip(xs, xs[1:]):
            G.add_edge((xa, y), (xb, y), **{weight: xb - xa})

    # Vertical edges between consecutive y-coordinates on each grid line x.
    for x in xs:
        for ya, yb in zip(ys, ys[1:]):
            G.add_edge((x, ya), (x, yb), **{weight: yb - ya})

    terminals = [(x, y) for (x, y) in pts]
    return G, terminals
