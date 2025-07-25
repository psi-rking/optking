Algorithms
==========

Methods Implemented
~~~~~~~~~~~~~~~~~~~

Optking performs Newton-Raphson (NR) and Qausi-NR optimizations for small to medium-sized molecules.

Algorithms include:

* Minimization
    * Steepest descent
    * NR
    * Rational function optimization (RFO)
    * Restricted Step RFO
    * Conjugate Gradient

* TS
    * Image RFO
    * Partioned RFO

* Reaction Path
    * Intrinsic Reaction Coordinate (IRC)

The type of optimization is controlled by the ``step_type`` and ``opt_type`` keywords. ``step_type`` chooses optimization algorithm (SD, NR, etc.)
``opt_type`` selects the kind of optimization (min, TS, or IRC) and unless overriden chooses the appropriate (or default) ``step_type``.

Classes and Functions
~~~~~~~~~~~~~~~~~~~~~

For more information on the OptHelper classes see :ref:`OptHelper <interfaces>` To simply run an optimization
the ``OptHelper`` classes or the interfaces through ``QCEngine`` and ``Psi4`` are recommended.

For more information on interacting directly with the algorithms see the API documentation below. In short,
the OptimizationAlgorithm class provides a basic "interface" for running optimizations through the ``take_step()``
method. ``The OptimizationManager`` class extends this interface to encompass the addition of linesearching
to any of the basic algorithms and ``IRCFollowing``.

.. Sphinx keeps trying to show imported variables and classes which it can't. Easier to include
.. than to skip

.. automodapi:: optking.stepAlgorithms
.. automodapi:: optking.linesearch
.. automodapi:: optking.IRCfollowing
