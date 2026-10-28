# Getting started

A Steiner tree connects required nodes, called *terminals*, at minimum cost.
It can pass through other nodes when that makes the connection cheaper.

The {doc}`../examples/quickstart` notebook builds a four-node NetworkX graph,
solves it with HiGHS, and plots the selected edges. It includes all the code
needed to run the example locally.

## Trees and forests

Use `terminals=["A", "B", "C"]` to connect all three nodes. Nested lists such as
`terminals=[["A", "B"], ["C", "D"]]` specify forest requirements: each list must
connect internally, and the paths may share edges. For alternatives, use
`groups=[["A", "B"], ["C", "D"]]` to select at least one member of each group
in one connected component.

## Next steps

- {doc}`../guide/variants`: choose connectivity, costs, and constraints.
- {doc}`../guide/solvers`: choose a solver and interpret its result.
- {doc}`../examples/connectivity`: compare connectivity requirements visually.
- {doc}`../guide/performance`: understand defaults and optional accelerators.
