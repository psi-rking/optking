General Recommendations
=======================

Redundant internal coordinates are the default coordinate set for optking. All possible simple stretches, bends,
and dihedral angles are constructed. Automatic supplementation of cartesian coordinates is in development.
If difficulty is encountered with linear coordinates, switching to `opt_coordinates = 'cartesian'` may be
nessecary. In some cases a full set of internal and cartesian coordinates has been found to work well this
corresonds to `opt_coordinates = 'both'`

Internal Coordinates
--------------------

.. automodapi:: optking.stre
.. automodapi:: optking.bend
.. automodapi:: optking.tors
