# References

SteinerPy builds on a large body of prior work.
The algorithms below are credited inline in the relevant source modules; this page collects them in one place.

## Formulation & exact solving

- R. T. Wong (1984), *A dual ascent approach for Steiner tree problems on a directed graph*, Mathematical Programming 28, 271–287, [doi:10.1007/BF02612335](https://doi.org/10.1007/BF02612335) — dual ascent for the directed-cut (arborescence) dual (`steinerpy/dual_ascent.py`).
- M. Leitner, I. Ljubić, M. Luipersbeck, M. Sinnl (2018), *A dual ascent-based branch-and-bound framework for the prize-collecting Steiner tree and related problems*, INFORMS Journal on Computing 30(2), 402–420, [doi:10.1287/ijoc.2017.0788](https://doi.org/10.1287/ijoc.2017.0788) — reduced-cost variable fixing (`steinerpy/dual_ascent.py`).
- D. Schmidt, B. Zey, F. Margot (2021), *Stronger MIP formulations for the Steiner forest problem*, Mathematical Programming 186, 373–407, [doi:10.1007/s10107-019-01460-6](https://doi.org/10.1007/s10107-019-01460-6) — branch-and-cut acceleration via creep flows and back cuts, used in the directed-cut separation (`steinerpy/mathematical_model.py`).

## Graph reductions

- D. Rehfeldt, T. Koch (2023), *Implications, conflicts, and reductions for Steiner trees*, Mathematical Programming 197(2), 903–966, [doi:10.1007/s10107-021-01757-5](https://doi.org/10.1007/s10107-021-01757-5) — Special Distance / bottleneck Steiner distance edge-deletion tests (`steinerpy/graph_reducer.py`).
- D. Rehfeldt, T. Koch (2020), *On the exact solution of prize-collecting Steiner tree problems*, ZIB-Report 20-11 ([PDF](https://optimization-online.org/wp-content/uploads/2020/04/7749.pdf); published in INFORMS Journal on Computing, [doi:10.1287/ijoc.2021.1087](https://doi.org/10.1287/ijoc.2021.1087)) — PCSTP/MWCSP transformations and the prize-constrained distance (PCD) reductions (`steinerpy/pc_transform.py`, `steinerpy/pc_reductions.py`).
- C. W. Duin (1993), *Steiner's problem in graphs*, PhD thesis, University of Amsterdam, and T. Polzin & S. Vahdati Daneshmand (2001), *Improved algorithms for the Steiner problem in networks*, Discrete Applied Mathematics 112(1–3), 263–300, [doi:10.1016/S0166-218X(00)00319-X](https://doi.org/10.1016/S0166-218X(00)00319-X) — alternative-based reduction tests (`steinerpy/graph_reducer.py`).
- I. Ljubić (2021), *Solving Steiner trees: Recent advances, challenges, and perspectives*, Networks 77(2), 177–204, [doi:10.1002/net.22005](https://doi.org/10.1002/net.22005) — survey informing the reduction and dual-ascent implementations.

## Heuristics & classic constructions

- L. Kou, G. Markowsky, L. Berman (1981), *A fast algorithm for Steiner trees*, Acta Informatica 15(2), 141–145, [doi:10.1007/BF00288961](https://doi.org/10.1007/BF00288961) — shortest-path Steiner tree heuristic and tree cleanup (`steinerpy/objects.py`, `steinerpy/dual_ascent.py`).
- K. Mehlhorn (1988), *A faster approximation algorithm for the Steiner problem in graphs*, Information Processing Letters 27(3), 125–128, [doi:10.1016/0020-0190(88)90066-X](https://doi.org/10.1016/0020-0190(88)90066-X) — terminal Voronoi / boundary-MST construction (`steinerpy/graph_reducer.py`, `steinerpy/objects.py`).

## Problem transformations

- M. Hanan (1966), *On Steiner's problem with rectilinear distance*, SIAM Journal on Applied Mathematics 14(2), 255–265, [doi:10.1137/0114025](https://doi.org/10.1137/0114025) — the Hanan grid reduction for rectilinear Steiner minimum trees (`steinerpy/rectilinear.py`).
- S. Voß (1999), *The Steiner tree problem with hop constraints*, Annals of Operations Research 86, 321–345, [doi:10.1023/A:1018967121276](https://doi.org/10.1023/A:1018967121276) — super-terminal transformation for the group Steiner tree problem (`steinerpy/objects.py`).
- D. Rehfeldt (2021), *Faster algorithms for Steiner tree and related problems: From theory to practice*, PhD thesis, TU Berlin — the Chapter 5 transformations behind the terminal-leaf, group, hop-constrained, rectilinear, and budgeted variants.
