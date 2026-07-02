# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
from pathlib import Path

import steinerpy

# The example notebook lives at the repository root (it is shipped in the
# sdist); copy it into the examples section so myst-nb can render it.
_root = Path(__file__).resolve().parents[2]
_examples = Path(__file__).parent / "examples"
_examples.mkdir(exist_ok=True)
shutil.copyfile(_root / "example.ipynb", _examples / "quickstart.ipynb")

# -- Project information -----------------------------------------------------

project = "SteinerPy"
author = "Berend Markhorst"
copyright = "2025, SteinerPy contributors"
version = steinerpy.__version__
release = steinerpy.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
]

exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
}

# -- Options for autodoc -----------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- Options for myst-nb -----------------------------------------------------

# Notebooks are committed with their outputs; never execute them on build.
nb_execution_mode = "off"

myst_enable_extensions = ["colon_fence", "dollarmath"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = f"SteinerPy {version}"
html_static_path = []

html_theme_options = {
    "source_repository": "https://github.com/berendmarkhorst/SteinerPy",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
