# SteinerPy v0.1.2 - Python 3.8 Compatibility & PyPI Release

## 🚀 Major Milestone: Now Available on PyPI!

Install with:
```bash
pip install steinerpy
```

## 🐛 Bug Fixes

- **Python 3.8 Compatibility**: Fixed type annotations throughout the codebase
  - Replaced modern `list[type]` syntax with `typing.List[Type]`
  - Added proper imports for `List`, `Set`, `Tuple`, `Dict` from `typing`
  - Now fully compatible with Python 3.8+ as advertised

- **Package Configuration**: 
  - Fixed pytest coverage configuration
  - Updated CI/CD pipeline for cross-platform testing

## 🧪 Testing

- Comprehensive test suite with **94% code coverage**
- Integration tests that verify actual solver functionality
- Tests pass on Python 3.8, 3.9, 3.10, 3.11, and 3.12

## 📦 Package Features

- **Steiner Tree Problems**: Connect a single set of terminals optimally
- **Steiner Forest Problems**: Connect multiple sets of terminals
- **HiGHS Solver Integration**: High-performance optimization
- **NetworkX Compatibility**: Works seamlessly with NetworkX graphs
- **Flexible Edge Weights**: Support for custom edge weight attributes

## 📖 Documentation

- Updated README with real PyPI badges and download statistics
- Example Jupyter notebook included
- Comprehensive API documentation
- Citation information for academic use

## 🔧 Development

- Modern Python packaging with `pyproject.toml`
- Automated CI/CD with GitHub Actions
- Cross-platform testing (Linux, macOS, Windows)
- Automated PyPI publishing on releases

## 🙏 Academic Citation

If you use this software in your research, please cite:

> Markhorst, B., Berkhout, J., Zocca, A., Pruyn, J., & van der Mei, R. (2025). Future-proof ship pipe routing: Navigating the energy transition. *Ocean Engineering*, 319, 120113.

---

**Full Changelog**: https://github.com/berendmarkhorst/SteinerPy/compare/v0.1.1...v0.1.2
