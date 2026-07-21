# Contributing to SteinerPy

Thank you for considering contributing to SteinerPy! We welcome bug reports,
feature requests, documentation improvements, and pull requests.

## Before you start

Conversations about new features and non-trivial changes belong in the
[issue tracker](https://github.com/berendmarkhorst/SteinerPy/issues). Before
you invest significant effort in a change, please discuss it first in a GitHub
issue so we can agree on the approach together. This avoids wasted work and
usually results in a much smoother review.

One branch, one feature. Branches are cheap, and keeping each pull request
focused on a single change makes it far easier to review and merge.

## Setting up a development environment

SteinerPy uses [uv](https://docs.astral.sh/uv/) to manage dependencies and
environments. After forking and cloning the repository:

```bash
uv sync --dev
```

This installs SteinerPy in editable mode together with the development
dependencies. Then check that everything works by running the test suite:

```bash
uv run pytest
```

The test suite runs with coverage enabled by default (configured in
`pyproject.toml`).

## Code style

- Format code with **black** (line length 88): `uv run black steinerpy tests`
- Lint with **flake8**: `uv run flake8 steinerpy tests`
- Type-check with **mypy** where practical: `uv run mypy steinerpy`
- Support Python 3.8–3.12 — avoid syntax or standard-library features newer
  than 3.8, since CI tests the full matrix.

Every public function, class, and method should have a docstring. When a class
or algorithm implements a technique from the literature, cite the source paper
in the docstring — see the existing classes in `steinerpy/objects.py` for the
established style.

## Tests

All new code should come with tests. A few conventions used in this project:

- Tests live in `tests/`, one `test_*.py` file per feature.
- Where feasible, validate solver output against an **independent brute-force
  oracle** on small instances (see `tests/test_pc_transform.py` for the
  pattern), rather than against hard-coded expected values.
- Solutions returned by `get_solution()` carry a certified optimality `gap`;
  exact solve paths should assert `gap == 0.0`.
- CI uploads coverage to Codecov and fails if the upload fails, so make sure
  new code paths are exercised by the tests.

Run the suite locally with `uv run pytest` before opening a pull request.

## Documentation

The documentation lives in `docs/` and is built with Sphinx (MyST Markdown)
and published on [Read the Docs](https://steinerpy.readthedocs.io). If your
change adds or alters user-facing behaviour:

- Update the relevant guide pages under `docs/source/guide/` (for example,
  new problem variants belong in `guide/variants.md`).
- Update the API reference in `docs/source/api/steinerpy.rst`.
- Build the docs locally to check they parse (requires Python ≥ 3.10):

  ```bash
  uv sync --extra docs
  uv run sphinx-build -b html docs/source docs/build/html
  ```

## Adding a new problem variant

New problem variants are a common and very welcome type of contribution. There
is no plugin system — variants are plain classes. The checklist:

1. **Add the problem class** in `steinerpy/objects.py`, subclassing the
   closest existing variant (`BaseSteinerProblem`, `SteinerProblem`,
   `DirectedSteinerProblem`, `PrizeCollectingProblem`, ...). Preserve the
   public contract: the constructor takes a NetworkX graph plus
   variant-specific arguments, and `get_solution()` returns a `Solution`
   object with `.objective`, `.selected_edges`/`.edges`, `.gap`, and
   `.runtime`.
2. **Export it** from `steinerpy/__init__.py` (both the import block and
   `__all__`).
3. **Add tests** in a new `tests/test_<variant>.py`, ideally with a
   brute-force oracle on small instances.
4. **Document it** in `docs/source/guide/variants.md` and
   `docs/source/api/steinerpy.rst`, and cite the relevant literature in the
   class docstring.
5. If standard benchmark instances exist for the variant, consider extending
   the parsers in `benchmarks/stp_parser.py`.

## Commits and pull requests

- Write clear, concise commit messages: a subject line under 80 characters,
  with an explanatory body when the change needs context.
- Open pull requests against `main`.
- Note that **every merge to `main` automatically bumps the patch version and
  publishes a release to PyPI**, so pull requests must be complete and
  self-contained — no "part 1 of 3" merges that leave `main` in a broken
  state.
- If you used LLMs / generative AI to produce part of the contribution,
  please disclose this in the pull request description.

## Licensing

SteinerPy is licensed under the MIT License. By contributing you agree that
your contributions are licensed under the same terms. If your contribution
incorporates existing code, please make sure its license is compatible with
MIT and credit the original source.
