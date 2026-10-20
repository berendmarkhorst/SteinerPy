# Contributing to SteinerPy

Thank you for considering contributing to SteinerPy! We welcome bug reports,
feature requests, documentation improvements, and pull requests.

## Code of Conduct

Participation in this project is governed by our
[Code of Conduct](.github/CODE_OF_CONDUCT.md). Please read it before opening an
issue or pull request.

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

The public `steinerpy/problem.py` normalizes input through `_problem_spec.py`,
then delegates implementation selection and construction to `_problem_dispatch.py`.
`ProblemSpec` records connectivity, roots, costs/prizes, constraints, geometry,
and an optional explicit variant. Preprocessing and other constructor options
remain separate. `supplied_features` distinguishes omitted arguments from explicit
values (including `None`), preserving the legacy constructors' defaults.

`ProblemSpec.objective(resolved_variant, pc_transform=...)` builds a private
`ObjectiveSpec` in `_objective.py`: original edge/node coefficients, an optional
per-requirement penalty, and the reporting sign/offset. Use `None` as the resolved
variant for group composition; specify an active solve-time transform override
when needed (heuristic prize/MWCS solves use the transform convention). Its
evaluation assumes a feasible selection and does not validate constraints or
convert solver gaps. Keep objective coefficients separate from budget costs.
The composition decoder uses it to check original-space values. Standard HiGHS
and Gurobi edge objectives use `ObjectiveSpec.from_graph` on the current working
graph, so reduced and transformed costs are preserved. Use `edge_cost(edge)`
to respect directed arc orientation. Do not substitute the original-space
specification for working-graph coefficients. Native prize objectives and
their normalized specifications use `ObjectiveSpec.from_prizes`, including
the legacy maximum-weight penalty path. This conversion rewards positive
prizes only and retains an explicit penalty per omitted terminal requirement.
Signed node-weight objectives use `ObjectiveSpec.from_node_weights`: negate
the node weights, give topology edges zero cost, and report the opposite sign
with `value(nodes=...)`. Both node-budgeted MWCS solvers and the normalized
signed-weight specification use this conversion. Keep the node-cost budget
mapping separate from these signed objective coefficients. Classic prize
offsets, budget objectives, and transformation corrections remain separate.
Check both the objective algebra and unchanged reported values against tiny
independent oracles; the variant guide lists the reporting differences and special cases.
Use `value(...)` on original selections for reporting: it cancels prize offsets
before adding small edge costs. Converting an already rounded canonical scalar
with `reported_value(...)` cannot recover precision lost before that conversion.

For feature combinations, first extend the normalized specification and the
central `validate_problem` entry point in `_problem_validation.py`. Dispatch
calls it before constructing either a legacy class or the composition strategy;
it checks required arguments, compatible features, original-space costs, and
preprocessing restrictions. Add a regression that forbids constructor calls to
prove that unsupported combinations fail before transformations run. Keep the
legacy classes' own checks for callers that use those classes directly. Normalization
must not construct solver objects or reinterpret costs as a new objective. The private
`steinerpy/_composition.py` compiles group choices and node costs into the
existing directed-cut solver. Preserve original-space semantics and explicit
backmapping, and check new combinations against an independent tiny-instance
oracle. A new public subclass is not required for each combination.

Node-cost splitting lives in `steinerpy/_transformations.py` and is shared by
the legacy node-weighted adapter and group/node-cost composition. Its
`SplitNodes` result records the working graph, original-node input/output ports,
paid node arcs, and cross-arcs mapped to original edges. Later stages attach
group sinks to output ports, so group membership includes the member's node
cost. Preserve these mappings when composing stages; never recover original
identities by parsing artificial labels. Source graphs/cost mappings stay owned
by their callers. Missing edge costs default to zero in legacy node weighting
and one in composition. Existing terminal-cost corrections remain in the legacy
adapter. Group/node-cost composition sets `_hop_arcs` to the split stage's
original-edge cross-arcs. Non-negative costs admit a tree whose orientation
uses one cross-arc per original edge. Paid-node arcs, source arcs and group
connectors do not consume hops. Decoding checks the original-edge count.
Composition supplies `_degree_arc_groups`, mapping each original node to all
incident original-edge cross-arcs across both split ports. Both backends and
budget incumbent checks use this incidence mapping. Decoding independently
checks original degrees. `_budget_edge_costs` contains only original-edge
cross-arcs. Budget constraints and incumbent validation use these coefficients,
so paid-node/source/group arcs consume no edge budget. Composition first proves
maximum group coverage, fixes that count, then minimizes original edge/node
costs within the remaining time. Mandatory sinks cannot be penalized. Source
choices and component filtering must depend on mandatory requirements, not all
optional groups. Decoding validates original constraints and both objectives.

`attach_group_terminals` is the shared next stage for legacy undirected/directed
group solvers and composition. It returns a copied working graph, artificial
terminals, and connector provenance `(group index, original member)`. Directed
connectors enter sinks. Supply original-member output ports when attaching
groups after node splitting, so group membership cannot bypass node costs.
Artificial connector costs stay zero; their presence must be accounted for
explicitly when mapping constraints. Directed groups derive `_hop_arcs` by
excluding connector provenance, and use that same original-arc domain for
`_degree_edges`. Both HiGHS and Gurobi count original arcs, including zero-cost
ones. Degree is incoming plus outgoing selected original arcs at each original
node. Decoded results independently check both limits; the two constraints may
be combined. Plain models retain their full arc/edge domains. Directed-group
budgets reuse the optional-flow budget model: original costs are charged once,
connectors stay free, and the objective counts reached groups. Decoding removes
the working model's extra root count, validates original costs and coverage,
and reports missed original group indices. Budget incumbent validation must
use the same original degree/hop domains as the model. Plain undirected-group
and legacy node-weighted modifiers remain guarded until they use safe original
accounting. Composition provides the tested path for these combinations.

The undirected `GroupSteinerProblem` delegates solves to shared composition,
preserving its public `Solution` result and private augmentation helper. Never
solve its undirected augmented graph directly: connector nodes can join group
members for free. A three-group path regression checks that this cannot yield
disconnected original edges. Legacy constructors reject unmapped modifiers
before transformation, and modular validation rejects the same unsupported
requests before constructing adapters. Keep legacy missing-cost conventions
and explicit `preprocess=False` behavior when maintaining compatibility.

New problem variants are a common and very welcome type of contribution. There
is no plugin system — variants are plain classes. The checklist:

1. **Add the problem class** in `steinerpy/objects.py`, subclassing the
   closest existing variant (`BaseSteinerProblem`, `SteinerProblem`,
   `DirectedSteinerProblem`, `PrizeCollectingProblem`, ...). Preserve the
   public contract: the constructor takes a NetworkX graph plus
   variant-specific arguments, and `get_solution()` returns a `Solution`
   object with `.objective`, `.selected_edges`/`.edges`, `.gap`, and
   `.runtime`.
2. **Register it** in `steinerpy/_problem_dispatch.py` so it is available through the
   unified `Problem` interface, and add a case to `tests/test_problem.py`.
   **Export it** from `steinerpy/__init__.py` (both the import block and
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
- Pull requests and pushes to `main` run tests automatically without publishing
  a release. Keep pull requests complete and self-contained so `main` stays usable.
- To publish a release, open **Actions → CI/CD Pipeline → Run workflow**, select
  **main**, and run it. After all tests pass, the workflow bumps the patch version,
  creates a tag, and publishes to PyPI. Manual runs on other branches only run tests.
- If you used LLMs / generative AI to produce part of the contribution,
  please disclose this in the pull request description.

## Licensing

SteinerPy is licensed under the MIT License. By contributing you agree that
your contributions are licensed under the same terms. If your contribution
incorporates existing code, please make sure its license is compatible with
MIT and credit the original source.
