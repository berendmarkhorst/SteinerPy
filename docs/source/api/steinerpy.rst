API reference
=============

All public classes are importable directly from the top-level :mod:`steinerpy`
package:

.. code-block:: python

   from steinerpy import SteinerProblem, PrizeCollectingProblem, ...

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

Deprecated
----------

These classes remain for backward compatibility; pass ``max_degree=`` or
``budget=`` to the base problem class instead (see :doc:`../guide/variants`).

.. autoclass:: steinerpy.DegreeConstrainedSteinerProblem
   :show-inheritance:

.. autoclass:: steinerpy.BudgetConstrainedSteinerProblem
   :show-inheritance:
