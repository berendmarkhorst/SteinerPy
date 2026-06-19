# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Dual-ascent accelerator** (opt-in `dual_ascent=True`): a Wong (1984)
  dual-ascent procedure computes a lower bound, a primal heuristic, and reduced
  costs, then applies reduced-cost variable fixing (Leitner et al. 2018) to
  shrink the ILP before solving — and solves directly (no ILP) when the bound is
  tight. Supported for Steiner tree, forest (multi-root) and directed problems;
  off by default and returns the same optimum as the baseline. New module
  `steinerpy.dual_ascent`.
- `benchmarks/` harness: SteinLib `.stp` parser, known-optima validation, and an
  HPC-friendly (resumable, parallel) runner comparing baseline vs accelerator.

## [1.0.1] - 2026-06-16

### Fixed
- Backmapping of solutions from preprocessed graphs now expands chains of
  degree-2 contractions recursively, so `SteinerProblem` no longer returns edges
  that don't exist in the original graph ([#20]).
- `PrizeCollectingProblem` (and its subclass `MaxWeightConnectedSubgraph`) no
  longer run graph preprocessing: degree-1/degree-2 reductions discarded
  non-terminal node prizes and corrupted the objective. Preprocessing is forced
  off, and explicitly passing `preprocess=True` now raises a warning ([#19]).
- `steinerpy.__version__` is now read from the installed package metadata, so it
  always matches the released version instead of drifting out of sync.

[#19]: https://github.com/berendmarkhorst/SteinerPy/issues/19
[#20]: https://github.com/berendmarkhorst/SteinerPy/issues/20

## [0.1.3] - 2025-12-18

### Fixed
- Python 3.8 compatibility by replacing union type syntax (`|`) with `typing.Union`
- Updated PyPI badge links to point to the official PyPI project page

## [0.1.2] - 2025-12-18

### Fixed
- Python 3.8 compatibility by replacing modern type annotations (`list[type]`) with `typing.List[Type]`
- Type annotations throughout the codebase now compatible with Python 3.8+
- Pytest configuration updated to use correct package name for coverage

## [0.1.1] - 2025-12-18

### Fixed
- Package structure corrected for proper PyPI installation
- Import statements updated to use `steinerpy` package name

## [0.1.0] - 2025-12-18

### Added
- Initial release of SteinerPy
- `SteinerProblem` class for defining Steiner Tree and Forest problems
- `Solution` class for handling optimization results
- Support for NetworkX graphs with custom edge weights
- HiGHS solver integration for optimization
- Basic test coverage
- Documentation and examples

### Features
- Solve Steiner Tree problems (single terminal group)
- Solve Steiner Forest problems (multiple terminal groups)
- Configurable time limits and logging
- MIT license for open source usage
