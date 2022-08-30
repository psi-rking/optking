General Recommendations
=======================

Redundant internal coordinates are the default coordinate set for optking. All possible simple stretches, bends,
and dihedral angles are constructed. Automatic supplementation of cartesian coordinates is in development.
If difficulty is encountered with linear coordinates, switching to `opt_coordinates = 'cartesian'` may be
necessary. In some cases a full set of internal and cartesian coordinates has been found to work well this
corresonds to `opt_coordinates = 'both'`

It should be noted that optking does NOT accept zmatrix coordinates from any of it's supported interfaces.
Communication between qcengine and psi4 and optking is purely in cartesian coordinates.
Simple internal coordinates can be created through the class constructors if desired; however,
ghost atoms are not supported.

Adding Constraints to the Optimization
======================================

Three types of constraints are supported. The built in coordinates (stre, bend, cart, etc) can be frozen with a
`frozen_<coord>` keyword. The coordinate will be held constant with the initial provided value throughout the
optimization.

For users of the old c++ optking, the `fixed_<coord>` options have been replaced with more general 
`ranged_<coord>` counterparts. This keyword adds forces to the optimization to keep the coordinate within a desired
min and max value (these can be set arbitrarily close to each other to recover the `fixed_<coord>` behavior).

Where applicable input should be specified in Angstroms and Degrees (even if the geomtry is provided in atomic units through QCEngine).

Frozen Coordinates
~~~~~~~~~~~~~~~~~~

Input for the `frozen_<coord>` keywords is expected as a list of the indices to constrain.
Parentheses can be added by the user for clarity and are avoided.
The string parsing is more robust and forgiving than in the c++ version and multiline strings and strange spacing
is allowed.

For hydrogen peroxide the following coordinates could be frozen::

    hooh = psi4.geometry(
        """
      H
      O 1 0.90
      O 2 1.40 1 100.0
      H 3 0.90 2 100.0 1 115.0
    """
    )

    params = {
        "frozen_distance": "1 2 3 4"
    }

freezing bond angles::

    "frozen_bend": "1 2 3 2 3 4"

freezing torsions::

    "frozen_dihedral": "1 2 3 4"

The following strings are all acceptable to freeze the cartesian coordinates of the hydrogens::

    """ 1 Xyz 4 xYz """
    """ 2 xyz 3 xyz """
    """
     1 x
     1 y
     1 Z
     4 x
     4 Y
     4 z """

Ranged Coordinates
~~~~~~~~~~~~~~~~~~

The `ranged_<coord>` follows the basic format. `1 2 ... min max`. Multiple coordinates can be assigned
ranges by specifying them sequentially in the string. Using the above example for `HOOH`::
    
    params = {
        "ranged_distance": "2 3 1.38 1.42"
    }

or using parentheses for clarity::

    params = {
        "ranged_bend": "(1 2 3 99.0 110.0) (2 3 4 99.0 110.0)"
    }


Adding Forces
~~~~~~~~~~~~~

Custom forces may be added to specific coordinates by adding a potential function to the coordinate as a function of x.

For instance a simple linear potential function can be added to push the OH bond lengths towards 0.95 angstroms::

    "ext_force_distance": "1 2 '-8.0*(x-0.950)' 3 4 '-8.0*(x-0.950)'"

Currently supported functions are `"sin", "cos", "log", "ln", "log10", "exp", "pow", "abs"`


Multifragment Optimizations
===========================

For multifragment systems, dimer coordinates are recommended. These may be configured automatically
(simply set `"frag_mode"="MULTI"`) or manually through the `interfrag_coords` keyword. 
The `interfrag_coords` keyword expects a dictionary and has a number of fields to allow full
specification of these coordinates

The DimerFrag creates internal coordinates between pairs of molecules. i.e. water trimer would 
consist internal coordinates for A, B, and C as well as dimer coordinates for AB, AC, and BC

The important keys are
    * `"Natoms per frag"` is a list of ints
    * `X Frag` specifies the index of the Xth fragment in the molecular system.
    * `A Ref Atoms` list of the atoms to use as the three reference points. In the below example we choose the Oxygen atom,
      third hydrogen atom, and the center of the two hydrogens.
      Lists of multiple indices denote the center of mass between the specified atoms.

::

    h2oA = psi4.geometry(
        """
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """
    )
    Axyz = h2oA.geometry().np

    h2oB = psi4.geometry(
        """
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """
    )

    water_dimer = { 
        "Natoms per frag": [3, 3], 
        "A Frag": 1,  # Index of fragment in Molsys. 1 Based indexing
        "A Ref Atoms": [[1], [2, 3], [3]],
        "A Label": "Water-A",  # optional
        "B Frag": 2,
        "B Ref Atoms": [[4], [5, 6], [6]],
        "B Label": "Water-B",  # optional
    }   


.. automodapi:: optking.stre
.. automodapi:: optking.bend
.. automodapi:: optking.tors
