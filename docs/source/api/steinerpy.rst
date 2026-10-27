API reference
=============

Use :class:`~steinerpy.Problem` to define connectivity, costs, and constraints.
It selects a compatible implementation from the supplied features:

.. code-block:: python

   from steinerpy import Problem

   problem = Problem(graph, terminals=["A", "D"], max_degree=3)
   solution = problem.get_solution(solver="highs")

The named problem classes below remain importable from :mod:`steinerpy` and
supported for existing code. See :ref:`problem-compatibility` for migration
examples, cost conventions, and the two existing wrapper deprecations.

Unified interface
-----------------

.. autoclass:: steinerpy.Problem
   :members:
   :special-members: __init__

Core problems
-------------

.. autoclass:: steinerpy.SteinerProblem
   :members:
   :inherited-members:
   :special-members: __init__

.. autoclass:: steinerpy.DirectedSteinerProblem
   :members:
   :show-inheritance:
   :special-members: __init__

Prize-collecting and node-weighted problems
-------------------------------------------

.. autoclass:: steinerpy.PrizeCollectingProblem
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: steinerpy.DirectedPrizeCollectingProblem
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: steinerpy.NodeWeightedSteinerProblem
   :members:
   :special-members: __init__

.. autoclass:: steinerpy.MaxWeightConnectedSubgraph
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: steinerpy.BudgetedMaxWeightConnectedSubgraph
   :members:
   :show-inheritance:
   :special-members: __init__

Further variants
----------------

.. autoclass:: steinerpy.PartialTerminalSteinerProblem
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: steinerpy.FullTerminalSteinerProblem
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: steinerpy.GroupSteinerProblem
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: steinerpy.DirectedGroupSteinerProblem
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: steinerpy.HopConstrainedSteinerProblem
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: steinerpy.RectilinearSteinerProblem
   :members:
   :show-inheritance:
   :special-members: __init__

Solutions
---------

.. autoclass:: steinerpy.Solution
   :members:

.. autoclass:: steinerpy.PrizeCollectingSolution
   :members:
   :show-inheritance:

.. autoclass:: steinerpy.NodeWeightedSolution
   :members:
   :show-inheritance:

.. autoclass:: steinerpy.BudgetSolution
   :members:
   :show-inheritance:

.. autoclass:: steinerpy.RectilinearSolution
   :members:
   :show-inheritance:

.. autoclass:: steinerpy.OptimalSolutionPool
   :members:

Deprecated
----------

These two wrappers were already deprecated before the modular interface.
They remain available for backward compatibility; pass ``max_degree=`` or
``budget=`` to :class:`~steinerpy.Problem` instead. Other named problem classes
are not deprecated by this migration (see :ref:`problem-compatibility`).

.. autoclass:: steinerpy.DegreeConstrainedSteinerProblem
   :show-inheritance:

.. autoclass:: steinerpy.BudgetConstrainedSteinerProblem
   :show-inheritance:
