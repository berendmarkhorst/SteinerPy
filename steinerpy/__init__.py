"""
SteinerPy: A Python package for solving Steiner Tree and Steiner Forest Problems.

This package provides tools to solve Steiner Tree and Steiner Forest problems
using the HiGHS solver with NetworkX graphs.
"""

from .objects import SteinerProblem, Solution
from .mathematical_model import build_model, run_model
from ._version import __version__
__author__ = "Berend Markhorst, Joost Berkhout, Alessandro Zocca, Jeroen Pruyn, Rob van der Mei"
__email__ = "berend.markhorst@cwi.nl" 

__all__ = ["SteinerProblem", "Solution", "build_model", "run_model"]
