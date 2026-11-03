---
title: 'SteinerPy: a modular Python package for solving Steiner tree and forest problems'
tags:
  - Python
  - operations research
  - combinatorial optimization
  - network optimization
  - Steiner tree
  - Steiner forest
authors:
  - name: Berend Markhorst
    orcid: 0000-0002-3383-6862
    affiliation: '1, 2, 3'
affiliations:
  - name: NHH Norwegian School of Economics, Bergen, Norway
    index: 1
  - name: Centrum Wiskunde & Informatica (CWI), Amsterdam, The Netherlands
    index: 2
  - name: Vrije Universiteit Amsterdam, Amsterdam, The Netherlands
    index: 3
date: 3 November 2026
bibliography: paper.bib
---

# Summary

Designing networks, e.g. for telecommunication, computer chips, or even road planning, often involves connecting a set of locations at minimum costs [@ljubic2021]. Such a network can usually be constructed by allowing connections to share routes and by adding extra junction points where useful. Formally, given a weighted graph $G=(V,E)$ with nodes $V$, edges $E$ and a set of required terminal nodes $T \subseteq V$, the Steiner tree problem looks for a minimum-cost connected subgraph containing all nodes in $T$. The Steiner forest problem generalizes this by considering multiple sets of terminals, where the nodes within each set must be connected, but different sets need not be connected to each other [@schmidt2021].

[SteinerPy](https://github.com/berendmarkhorst/SteinerPy) is an open-source Python package for solving Steiner tree and forest problems and related network design variants. It builds upon the established [NetworkX](https://github.com/networkx/networkx) package [@hagberg2008], provides a modular framework that allows users to model specific variants of the Steiner tree problem, and solve them with exact and heuristic methods. The result includes an optimality gap, helping users assess the quality of the solution. `SteinerPy` is distributed through [PyPI](https://pypi.org/project/steinerpy/) under the MIT licence, has extensive [documentation](https://steinerpy.readthedocs.io/), and can use both the open-source [HiGHS](https://highs.dev) [@huangfu2018] solver and the commercial [Gurobi](https://www.gurobi.com) solver [@gurobi2026].

# Statement of need

Steiner tree and forest problems occur in many network design applications, but translating such an application into an optimization model requires considerable implementation effort. Furthermore, non-standard combinations of specific connectivity requirements, costs, or additional constraints can lead to even more development work, making it difficult to reuse code across related problems.

`SteinerPy` addresses this challenge through a modelling framework in which users can combine these problem features without implementing a new optimization model for each variant. This supports both methodological research, where comparable implementations are needed to evaluate algorithms, and applied research, where network optimization is one component of a larger study.

The package was motivated by the author's research on ship pipe routing, which used mixed-integer programming to model stochastic Steiner forests [@markhorst2025]. That work highlighted the value of reusable software for related problems and motivated the development of an open-source package.

# State of the field

Several software packages address problems related to Steiner trees using approximation algorithms, heuristics, or exact optimization methods. [`GeoSteiner`](https://github.com/Cui-Git-Hub/geosteiner-windows-python) provides exact algorithms for geometric Steiner tree problems, including Euclidean and rectilinear variants. `NetworkX` provides approximation algorithms for the classical Steiner tree problem on an undirected graph using algorithms by Kou et al. and Mehlhorn et al. [@kou1981fast; @mehlhorn1988; @hagberg2008]. [`OGDF`](https://github.com/ogdf/ogdf) provides exact, approximation, and heuristic Steiner tree methods in a `C++` graph library, which can also be used in Python using a separate package [@chimani2014ogdf]. The [`pcst_fast`](https://github.com/fraenkel-lab/pcst_fast) package provides approximations for prize-collecting Steiner trees [@hegde2015]. [`rustworkx`](https://github.com/Qiskit/rustworkx) is a Python graph library implemented in Rust that includes a Steiner tree approximation based on the algorithm by Mehlhorn et al. [@mehlhorn1988; @Treinish2022] and [`SageMath`](https://github.com/sagemath/sage) uses mixed-integer programming to solve Steiner tree problems [@sagemath]. [`SCIP-Jack`](https://scipjack.zib.de) combines graph reductions and branch-and-cut to solve Steiner tree problems and many related variants, and provides a Python interface with `NetworkX` support [@rehfeldt2020scip; @rehfeldt2023]. A few years after the original R package `SteinerNet` was published, the Python equivalent [`steinernetpy`](https://github.com/afshinsadeghi/steinernetpy) was released. It contains Steiner tree heuristics and an exact method based on exhaustive enumeration [@sadeghi2013steinernet]. Compared with these packages, `SteinerPy` focuses on combining multiple Steiner tree and forest variants within one Python modelling framework while supporting both exact and heuristic solution methods.

The table below shows an overview of the aforementioned packages. The second column indicates whether the package consists of an approximation algorithm (A), an exact methodology (E), or a heuristic (H). The third column addresses the scope of the package and the last column indicates whether the package qualifies as open source under the [OSI definition](https://opensource.org/osd). As the licenses of both `GeoSteiner` and `SCIP-Jack` do not allow commercial use, they cannot be labeled as open source. The package `pcst_fast` includes a forest option. Indeed, users can demand multiple trees, but they cannot specify which nodes belong in those trees. Therefore, this functionality is not entirely comparable with the Steiner forest functionality provided by `SteinerPy`. Finally, `steinernetpy`'s exact method is based on exhaustive enumeration, which quickly becomes intractable when the number of nodes and edges in the network grows.

| **Software** | **Algorithm type** | **Main scope**                                  | **Open source** |
|--------------|--------------------|--------------------------------------------|-----------------|
| `GeoSteiner`   | E                  | Geometric Steiner trees                    | No              |
| `NetworkX`     | A                  | Steiner tree                               | Yes             |
| `OGDF`         | A + E + H          | Steiner tree                               | Yes             |
| `pcst_fast`    | A                  | Prize-collecting Steiner tree/forest     | Yes             |
| `rustworkx`    | A                  | Steiner tree                               | Yes             |
| `SageMath`     | E                  | Steiner tree                               | Yes             |
| `SCIP-Jack`   | E + H              | Steiner tree + many related variants       | No              |
| `steinernetpy` | E + H          | Steiner tree                               | Yes             |
| `SteinerPy`    | E + H              | Steiner tree/forest + many related variants | Yes             |

# Software design

`SteinerPy` has a single modelling interface for a range of Steiner tree and forest problems. Users can specify a graph, connectivity requirements, costs, rewards, and optional constraints, after which the package selects a compatible solution strategy. Not every feature can be combined with every algorithm, so `SteinerPy` checks whether a requested combination is supported and selects a suitable solution strategy.

The package builds on `NetworkX` [@hagberg2008]. This allows users to work with familiar Python graph objects and makes `SteinerPy` easy to combine with other packages. The main trade-off is that this flexibility comes with some Python and memory overhead compared with faster, more specialised graph libraries.

For exact optimization, `SteinerPy` uses a mixed-integer programming formulation [@schmidt2021] and can solve it with either HiGHS or Gurobi. HiGHS is the default open-source solver [@huangfu2018], while Gurobi can be used when a licence is available [@gurobi2026]. For models that use connectivity cuts, `SteinerPy` uses the solver in different ways. With HiGHS, it repeatedly solves the model, adds any missing cuts, and solves again. With Gurobi, these cuts can be added during the solve using callbacks.

Where appropriate, `SteinerPy` also uses problem-specific algorithms and preprocessing. These include graph reductions [@rehfeldt2023], dual-ascent for obtaining bounds and candidate solutions [@wong1984], and dynamic programming for instances with few terminals [@dreyfus1971; @erickson1987]. The package automatically checks when such methods are applicable.

Finally, `SteinerPy` makes a clear distinction between finding a feasible solution and proving that it is optimal. Results therefore include both the best known solution and, where available, an optimality gap. A zero gap indicates a proven optimum, while a positive gap shows that a difference remains between the solution and the best known bound. This distinction is especially useful for larger instances where solution methods may be stopped before optimality is proven.

# Research impact statement

`SteinerPy` has been benchmarked on public instances from the PACE 2018 Steiner Tree Challenge [@bonnet2019] and the corresponding results can be found on GitHub. This setup makes it easier to test new algorithms. Beyond standalone use, `SteinerPy` can be used as a building block in larger software packages that require network optimization. This can lower the implementation effort and allow researchers to focus on the application itself rather than developing Steiner tree models from scratch.

# AI usage disclosure

Codex and Claude were used as a sparring partner for software design choices, assistance with code writing and reviewing, development of tests and documentation, and editing this paper. The author reviewed these suggestions through manual code review and testing against known optimal solutions and takes full responsibility for the software, documentation, and paper.

# Acknowledgements

This work was supported by the projects COCONUT (project number 18938) and READINESS (project number TWM.BL.019.002), both financed by the Dutch Research Council (NWO). The author thanks [SURF](www.surf.nl) for the support in using the National Supercomputer Snellius.

# References