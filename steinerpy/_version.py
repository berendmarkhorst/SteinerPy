"""Version information for SteinerPy.

The version is read from the installed package metadata so it always matches the
version declared in ``pyproject.toml`` (the single source of truth) and can never
drift out of sync.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("steinerpy")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
