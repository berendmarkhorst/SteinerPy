# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
