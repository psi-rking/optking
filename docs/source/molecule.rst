Molecules
=========

Overview
--------

Opkting's molecular system, at its core, is a list of intramolecular fragments ``frag.Frag`` and
intermolecular pseudo-fragments ``dimerfrag.DimerFrag``. The fragments in turn are lists of masses, atomic numbers and
coordinates (cartesian or internal) along with a numpy array for the cartesian geometry.
NIST values for masses, and atomic numbers can be easily retrieved through ``qcelemental`` e.g.
``qcelemental.periodictable.to_Z('O')`` to aid in creating a molecule. To utilize a custom molecule
in an optimzation see :ref:`CustomHelper` and :ref:`Interfaces`.

A fragment for water:

.. code-block:: python-console

    >>> import optking
    >>> zs = [1, 8, 1]
    >>> masses = [1.007825032230, 15.994914619570, 1.007825032230]
    >>> geometry = [[-0.028413670411,     0.928922556351,     0.000000000000],
                    [-0.053670056908,    -0.039737675589,     0.000000000000],
                    [ 0.880196420813,    -0.298256807934,     0.000000000000]]
    >>> fragment = optking.Frag(zs, geometry, masses)

The optimization coordinates can be added manually.

.. code-block:: python-console

    >>> intcos = [optking.Stre(1, 2), optking.Stre(2, 3), optking.Bend(1, 2, 3)]
    >>> fragment.intcos = intcos  # intcos can also be added at instantiation

More typically, the coordinate system is automatically generated once the full molecular system is built and then edited if nessecary.

.. code-block:: python-console

    >>> molsys = optking.Molsys([fragment])
    >>> optking.make_internal_coords(molsys)
