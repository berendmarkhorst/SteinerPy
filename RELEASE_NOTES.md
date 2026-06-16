# SteinerPy v1.0.1 - Preprocessing Bug Fixes

## Bug Fixes

- **Recursive solution backmapping ([#20])**: When graph preprocessing collapsed
  a chain of degree-2 nodes in several steps, the solution was only expanded one
  level, so `SteinerProblem` could return edges that don't exist in the original
  graph (e.g. `[(0, 2), (2, 3)]` for a `0-1-2-3` chain). Backmapping now walks the
  full contraction chain and returns the correct original edges
  (`[(0, 1), (1, 2), (2, 3)]`).

- **Prize-Collecting preprocessing ([#19])**: Graph reduction is incompatible with
  the Prize-Collecting objective — degree-1 removal and degree-2 contraction
  discard non-terminal nodes together with their prizes. `PrizeCollectingProblem`
  and `MaxWeightConnectedSubgraph` now force `preprocess=False`, and passing
  `preprocess=True` explicitly raises a warning explaining why.

- **Version reporting**: `steinerpy.__version__` is now sourced from the installed
  package metadata, so it always matches the released version.

## Testing

- Added regression tests for multi-hop contraction backmapping (`tests/test_graph_reducer.py`).
- Added Prize-Collecting regression and warning tests.
- Full suite passes on Python 3.8–3.12.

## Install

```bash
pip install --upgrade steinerpy
```

## Academic Citation

If you use this software in your research, please cite:

> Markhorst, B., Berkhout, J., Zocca, A., Pruyn, J., & van der Mei, R. (2025). Future-proof ship pipe routing: Navigating the energy transition. *Ocean Engineering*, 319, 120113.

---

**Full Changelog**: https://github.com/berendmarkhorst/SteinerPy/compare/v1.0.0...v1.0.1

[#19]: https://github.com/berendmarkhorst/SteinerPy/issues/19
[#20]: https://github.com/berendmarkhorst/SteinerPy/issues/20
