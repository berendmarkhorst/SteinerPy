"""PCSTP / MWCSP -> Steiner Arborescence (SAP) transformation for SteinerPy.

Both Rehfeldt & Koch papers solve the prize-collecting Steiner tree problem
(PCSTP) and the maximum-weight connected subgraph problem (MWCSP) by the same
recipe: *change the problem class* — transform the problem to a rooted Steiner
Arborescence Problem (SAP) and solve the directed-cut formulation with dual
ascent.  SteinerPy already has a dual-ascent + directed-cut solver for the SAP
(``steinerpy.dual_ascent`` and ``steinerpy.mathematical_model``), so the
adaptation is mostly a *transformation* that turns a prize-collecting problem
into that SAP, plus a back-mapping from the arborescence to the original.

This module implements

* :func:`transform_pcstp_to_sap` — the (classic, forgo-prize) PCSTP-to-SAP
  transformation of Rehfeldt & Koch, "On the exact solution of prize-collecting
  Steiner tree problems" (ZIB 20-11, 2020), Transformation 2, *with the
  cost-shifting on non-proper potential terminals*;
* :func:`transform_mwcsp_to_pcstp` — the MWCSP-to-PCSTP reduction of the same
  report, Section 2.2 (``c(e):=-w0``, ``p(v):=w(v)-w0``);
* :func:`map_sap_solution_to_pcstp` — the back-mapping (report eq. 26-29) that
  recovers the original edges, the prize-collected vertices, and the exact
  PCSTP objective from an arborescence.

**Classic PCSTP objective.** ``C(S) = sum_{e in S} c(e) + sum_{v not in S} p(v)``
with ``c >= 0`` and ``p >= 0`` (forgo the prize of any vertex left out of the
tree ``S``).  This is the objective both papers solve; it is *not* the same as
SteinerPy's penalty/Big-M ``PrizeCollectingProblem`` model unless that model is
used with ``penalty_cost == 0`` (see ``objects._pc_eligible``).
"""

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Set, Tuple

import networkx as nx

Arc = Tuple[Hashable, Hashable]

# Sentinel node labels for the transformed graph.  Tuples with a reserved string
# head are practically collision-free with original node labels and format
# cleanly into the solver's variable-name strings.
ROOT = ("__pc_root__",)
HUB = ("__pc_hub__",)          # v0' in the report: the shared "forgo" hub


def _root_label():
    return ROOT


def _hub_label():
    return HUB


def _term_label(t) -> Arc:
    """Auxiliary terminal t'_i for the (proper) potential terminal ``t``."""
    return ("__pc_term__", t)


# ---------------------------------------------------------------------------
# Transform context
# ---------------------------------------------------------------------------

@dataclass
class PCTransform:
    """Everything needed to solve a PCSTP via its SAP and map the result back."""
    sap_graph: nx.DiGraph
    root: Hashable
    terminals: List[Hashable]          # the auxiliary t'_i (SAP terminals minus root)
    weight: str
    node_prizes: Dict
    proper_terminals: Set
    aux_arc_kind: Dict[Arc, str]       # 'orig' | 'root' | 'gadget' | 'collect'
    offset: float                      # pcstp_obj = sap_obj - offset
    big_m: float
    # Constant added to recover an MWCSP weight from a PCSTP objective:
    # mwcsp_weight = mwcsp_const - pcstp_obj.  None for a native PCSTP.
    mwcsp_const: Optional[float] = None
    # Original graph kept for direct (offset-independent) objective recomputation.
    _orig_graph: Optional[nx.Graph] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Proper / non-proper potential terminals
# ---------------------------------------------------------------------------

def _min_incident_cost(graph, v, weight: str) -> float:
    costs = [data.get(weight, 1) for _, _, data in graph.edges(v, data=True)]
    return min(costs) if costs else float("inf")


def is_proper_potential_terminal(graph, v, node_prizes: Dict, weight: str = "weight") -> bool:
    """``t`` is a *proper* potential terminal iff ``p(t) > min`` incident edge cost.

    (Rehfeldt & Koch 2020, Sec. 2.)  A non-proper potential terminal (``0 <
    p(t) <= min`` incident cost) is handled by *cost-shifting* its incoming arcs
    rather than by a terminal gadget, which keeps every shifted arc cost ``>= 0``.
    """
    p = node_prizes.get(v, 0)
    if p <= 0:
        return False
    return p > _min_incident_cost(graph, v, weight)


# ---------------------------------------------------------------------------
# PCSTP -> SAP
# ---------------------------------------------------------------------------

def transform_pcstp_to_sap(
    graph: nx.Graph,
    node_prizes: Dict,
    weight: str = "weight",
    big_m: Optional[float] = None,
) -> PCTransform:
    """Classic forgo-prize PCSTP ``(graph, node_prizes)`` -> equivalent SAP.

    Builds the directed graph ``G' = (V', A')`` of Rehfeldt & Koch (2020),
    Transformation 2:

    * every original undirected edge ``{v, w}`` becomes the two arcs ``(v, w)``
      and ``(w, v)``; the head's prize is *cost-shifted* into the arc when the
      head is a **non-proper** potential terminal:
      ``c'(v, w) = c({v, w}) - p(w)`` if ``w`` non-proper, else ``c({v, w})``
      (this keeps all arc costs ``>= 0``);
    * an artificial root ``r'`` and a shared forgo-hub ``v0'`` are added; for
      each **proper** potential terminal ``t`` an auxiliary terminal ``t'`` is
      added together with arcs ``(r', t)=M``, ``(t, v0')=0``, ``(t, t')=0`` and
      ``(v0', t')=p(t)``;
    * the SAP terminals are ``{t'} U {r'}``.

    The big-M ``M`` anchors the tree to ``r'`` through exactly one arc; the
    objective is recovered exactly via ``pcstp_obj = sap_obj - offset`` with
    ``offset = M - sum p(non-proper potential terminals)``.

    :returns: a :class:`PCTransform`.
    """
    proper: Set = set()
    nonproper: Set = set()
    for v in graph.nodes():
        if node_prizes.get(v, 0) > 0:
            if is_proper_potential_terminal(graph, v, node_prizes, weight):
                proper.add(v)
            else:
                nonproper.add(v)

    sum_proper = sum(node_prizes[t] for t in proper)
    sum_nonproper = sum(node_prizes[t] for t in nonproper)
    # M must strictly exceed any single forgo so the optimum uses exactly one
    # (r', t) anchor arc (never a second one in place of forgoing a terminal).
    if big_m is None:
        big_m = sum_proper + max((node_prizes[t] for t in proper), default=0.0) + 1.0

    D = nx.DiGraph()
    D.add_nodes_from(graph.nodes())
    aux_kind: Dict[Arc, str] = {}

    # Original edges -> two cost-shifted arcs.
    for u, v, data in graph.edges(data=True):
        c = data.get(weight, 1)
        for (a, b) in ((u, v), (v, u)):
            shifted = c - node_prizes[b] if b in nonproper else c
            # Guard against tiny negative round-off.
            D.add_edge(a, b, **{weight: max(0.0, shifted)})
            aux_kind[(a, b)] = "orig"

    # Gadgets for proper potential terminals.
    root, hub = _root_label(), _hub_label()
    D.add_node(root)
    terminals: List = []
    if proper:
        D.add_node(hub)
    for t in proper:
        tp = _term_label(t)
        D.add_node(tp)
        terminals.append(tp)
        D.add_edge(root, t, **{weight: big_m});       aux_kind[(root, t)] = "root"
        D.add_edge(t, hub, **{weight: 0.0});          aux_kind[(t, hub)] = "gadget"
        D.add_edge(t, tp, **{weight: 0.0});           aux_kind[(t, tp)] = "gadget"
        D.add_edge(hub, tp, **{weight: node_prizes[t]}); aux_kind[(hub, tp)] = "collect"

    offset = big_m - sum_nonproper

    return PCTransform(
        sap_graph=D, root=root, terminals=terminals, weight=weight,
        node_prizes=dict(node_prizes), proper_terminals=proper,
        aux_arc_kind=aux_kind, offset=offset, big_m=big_m,
        _orig_graph=graph,
    )


# ---------------------------------------------------------------------------
# MWCSP -> PCSTP (then compose with transform_pcstp_to_sap)
# ---------------------------------------------------------------------------

def transform_mwcsp_to_pcstp(
    graph: nx.Graph,
    node_weights: Dict,
    weight: str = "weight",
) -> Tuple[nx.Graph, Dict, float]:
    """MWCSP ``(graph, node_weights)`` -> equivalent classic PCSTP.

    Rehfeldt & Koch (2020), Sec. 2.2: let ``w0 = min_v w(v)``.  Define edge costs
    ``c(e) := -w0`` (``> 0`` when some weight is negative) and prizes
    ``p(v) := w(v) - w0 >= 0``.  Maximising ``sum_{v in S} w(v)`` over connected
    ``S`` is then equivalent to minimising the PCSTP objective, and the original
    weight is recovered by ``mwcsp_weight = mwcsp_const - pcstp_obj`` with
    ``mwcsp_const = (1 - n) * w0 + sum_v w(v)``.

    :returns: ``(pcstp_graph, pcstp_prizes, mwcsp_const)``.  The returned graph
        reuses the original topology with uniform edge cost ``-w0``.
    """
    if not node_weights:
        raise ValueError("node_weights must be non-empty for MWCSP.")
    w0 = min(node_weights.get(v, 0) for v in graph.nodes())
    n = graph.number_of_nodes()

    pc_graph = nx.Graph()
    pc_graph.add_nodes_from(graph.nodes())
    edge_cost = -w0  # >= 0; 0 only in the degenerate all-nonnegative case
    for u, v in graph.edges():
        pc_graph.add_edge(u, v, **{weight: edge_cost})

    prizes = {v: node_weights.get(v, 0) - w0 for v in graph.nodes()}
    mwcsp_const = (1 - n) * w0 + sum(node_weights.get(v, 0) for v in graph.nodes())
    return pc_graph, prizes, mwcsp_const


# ---------------------------------------------------------------------------
# SAP arborescence -> PCSTP solution
# ---------------------------------------------------------------------------

def map_sap_solution_to_pcstp(
    ctx: PCTransform,
    sap_arcs: List[Arc],
) -> Tuple[List[Tuple], List, float]:
    """Map an arborescence (list of SAP arcs) back to a *valid* PCSTP tree.

    Strips every auxiliary arc (root / gadget / collect) to recover the original
    undirected edges used.  A classic PCSTP solution must be a **single connected
    tree**, but an SAP arborescence may legitimately branch from the artificial
    root ``r'`` to several potential terminals (each branch a separate subtree),
    which maps to a *forest*.  We therefore evaluate every connected component of
    the recovered original-edge subgraph (each isolated gadget-collected terminal
    is its own component, plus the empty tree) as a candidate single-tree PCSTP
    solution and return the cheapest:

        ``pcstp_obj(C) = sum_{e in C} c(e) + (sum_all p - sum_{v in C} p(v))``.

    For an exact (one-anchor) arborescence there is a single component, so this is
    a no-op; for a heuristic primal that branched, it extracts the best valid
    sub-tree.

    :returns: ``(original_edges, collected_nodes, pcstp_objective)``.
    """
    arc_set = set(sap_arcs)
    kind = ctx.aux_arc_kind

    seen: Set[frozenset] = set()
    H = nx.Graph()
    for a in sap_arcs:
        if kind.get(a) == "orig":
            key = frozenset(a)
            if key not in seen:
                seen.add(key)
                w = ctx._orig_graph.get_edge_data(a[0], a[1]).get(ctx.weight, 1)
                H.add_edge(a[0], a[1], _w=w)
    # Proper terminals reached through their own (t, t') gadget arc: include even
    # if they have no original edge (isolated single-terminal tree).
    for t in ctx.proper_terminals:
        if (t, _term_label(t)) in arc_set:
            H.add_node(t)

    total_prize = sum(p for p in ctx.node_prizes.values() if p > 0)
    best_edges: List[Tuple] = []
    best_nodes: List = []
    best_obj = total_prize  # the empty tree (forgo everything)

    for comp in nx.connected_components(H):
        sub = H.subgraph(comp)
        edge_cost = sum(d.get("_w", 1) for _, _, d in sub.edges(data=True))
        comp_prize = sum(ctx.node_prizes.get(v, 0) for v in comp)
        obj = edge_cost + (total_prize - comp_prize)
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_edges = [(u, v) for u, v in sub.edges()]
            best_nodes = sorted(comp, key=lambda n: str(n))

    return best_edges, best_nodes, best_obj


def best_trivial_pcstp(node_prizes: Dict) -> Tuple[List, float]:
    """Best single-vertex / empty PCSTP solution.

    The SAP can represent neither the empty tree nor a single non-proper /
    Steiner vertex, so the caller must compare its result with this baseline.
    Returns ``([best_node] or [], objective)`` where the single-vertex objective
    is ``sum_all p - max_v p(v)`` and the empty-tree objective is ``sum_all p``.
    """
    total = sum(p for p in node_prizes.values() if p > 0)
    if not node_prizes:
        return [], 0.0
    best_node = max(node_prizes, key=lambda v: node_prizes.get(v, 0))
    best_p = node_prizes.get(best_node, 0)
    if best_p <= 0:
        return [], total  # empty tree is best
    return [best_node], total - best_p
