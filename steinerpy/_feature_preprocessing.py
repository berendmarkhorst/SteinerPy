"""Resource bounds that preserve every feasible original selection."""


def feasible_edges(spec):
    """Drop only edges individually forbidden by active original-space limits.

    No contractions or cost rewriting: node costs, hops, leaves and all tied
    optima retain their meaning. Stronger reductions need feature-specific proofs.
    """
    for u, v in spec.graph.edges:
        if spec.hop_limit == 0 or spec.max_degree == 0:
            continue
        if (
            spec.budget is not None
            and spec.graph.edges[u, v].get(spec.weight, 1) > spec.budget
        ):
            continue
        if (
            spec.node_budget is not None
            and sum((spec.node_costs or {}).get(n, 0) for n in (u, v))
            > spec.node_budget
        ):
            continue
        yield u, v
