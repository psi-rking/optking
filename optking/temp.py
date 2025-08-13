diff --git a/optking/addIntcos.py b/optking/addIntcos.py
index 43cba52..dfa0f0b 100644
--- a/optking/addIntcos.py
+++ b/optking/addIntcos.py
@@ -1,6 +1,5 @@
 import json
 import logging
-import re
 from copy import deepcopy
 from itertools import combinations, permutations
 
@@ -13,7 +12,6 @@ from . import stre, tors, v3d
 from .exceptions import AlgError, OptError
 from .v3d import are_collinear
 from . import log_name
-from . misc import grouper
 
 # Functions related to freezing, fixing, determining, and
 #    adding coordinates.
@@ -1105,302 +1103,3 @@ def improper_torsion_around_oofp(center, a, b, c, geom):
         return tors.Tors(b, a, center, c)
     else:
         return tors.Tors(a, b, center, c)
-
-
-def add_coordinates(type, spec, molsys, **kwargs):
-    """ Primary interface to create coordinates of `type` from a list of atom indices.
-    (assume 1-based indexing at this point)
-
-    Parameters
-    ----------
-    type: str
-        one of 'stre', 'bend', 'tors', 'oofp'
-    spec: iterable[int],
-        a list of atomic indices (must be overall indices) to define the coordinates. Zero based
-        indices expected.
-
-        If 'ranged' is set in kwargs, then allow floats among coordinate indices to specify range:
-        `0 1 1.080 1.095 0 2 1.080 1.095...`
-    'frozen': bool (optional):
-        whether these coordinates should be added
-    'ranged': bool (optional):
-         Whether there are ranges in the provided spec
-
-    Raises
-    ------
-    ValueError
-        If `type` is not one of the allowed values or if `atoms` list does not have the correct
-        number of entries (this error comes from `grouper`)
-
-    """
-
-    # classes. can be called to create the desired coordinate can access type and natom
-    # info
-    coords = {
-        "stre": stre.Stre,
-        "bend": bend.Bend,
-        "tors": tors.Tors,
-        "oofp": oofp.Oofp,
-    }
-
-    if type not in coords.keys():
-        raise ValueError(
-            f" Cannot create the coordinate: {type}. Valid values include: {coords.keys()}"
-        )
-
-    coord = coords[type]
-
-    if kwargs.get("constraint") == "ranged":
-        parse_ranged_intco(coord, spec, molsys, **kwargs)
-    elif kwargs.get("constraint") == "ext_force":
-        parse_ext_intco(coord, spec, molsys, **kwargs)
-    else:
-        # frozen can be allowed straight through in kwargs
-        parse_std_intco(coord, spec, molsys, **kwargs)
-
-
-def add_coordinates_from_dict(user_coords: str, molsys):
-    """ add coordinates with a dictionary
-
-    Examples
-    --------
-    {
-        'stretches': [(1, 2), (3, 4), (5, 6)],
-        'bends': [[1, 2, 3], [4, 5, 6]],
-        'torsion': '1 2 3 4 5 6 7 8, 9 10 11 12'
-        'frozen streches': [(7, 8)]
-        'ranged torsions': [(10 11 12 13 1.0 3.16)]
-        'carts': [1 X 1 Y 1 Z 2 XY 3 XYZ]
-        'ext_force torsion: [1 2 3 4 Sin(x) + Cos(x)]
-    }
-
-    """
-    # Regexes to allow users to slightly vary how they denote incoming coordinates
-    # a few words are allowed, plurals allowed, arbitrary capitalization will be allowed
-    S_REGEX = r"(stretch|bond|stre)s?(es)?"
-    B_REGEX = r"(bend|angle)s?"
-    T_REGEX = r"(tors|torsion|dihedral)s?"
-    O_REGEX = r"(out[\s_-]of[\s_-]plane|oofp)s?"
-    C_REGEX = r"(cart|cartesian)s?"
-
-    # extra regexes to detect a prefix indicating that the coordinate is constrained in some way 
-    F_REGEX = r"(frozen|freeze)\s?\-?\_?"
-    R_REGEX = r"ranged?\s?\-?\_?"
-    E_REGEX = r"(external\s?\-?\_?]force|ext\s?\-?\_?force)"
-
-    def check_constrained(key):
-        """Create an initial kwargs to send to add_coordinates to signify constraints"""
-        if re.match(F_REGEX, key):
-            # This is all thats needed for frozen
-            return {"constraint": "frozen"}
-        elif re.match(R_REGEX, key):
-            # Prep we'll need to fill in 'range_[min/max]' later.
-            return {"constraint": "ranged"}
-        elif re.match(E_REGEX, key):
-            return {"constraint": "ext_force"}
-        else:
-            return {}
-
-    # load dictionary so it can be accessed
-    user_dict: dict = json.loads(user_coords)
-
-    for key, value in user_dict.items():
-        if not isinstance(value, str):
-            # if loads turned values into arrays or similar turn back to string
-            value = str(value)
-
-        key_l = key.lower()
-        if re.search(S_REGEX, key_l):
-            add_coordinates("stre", value, molsys, **check_constrained(key_l))
-        elif re.search(B_REGEX, key_l):
-            add_coordinates("bend", value, molsys, **check_constrained(key_l))
-        elif re.search(T_REGEX, key_l):
-            add_coordinates("tors", value, molsys, **check_constrained(key_l))
-        elif re.search(O_REGEX, key_l):
-            add_coordinates("oofp", value, molsys, **check_constrained(key_l))
-        elif re.search(C_REGEX, key_l):
-            raise NotImplementedError(
-                "Cartesian coordinates are not supported in `CUSTOM_COORDINATES`"
-            )
-        else:
-            raise ValueError(
-                f"An unexpected key {key_l} was found. Valid keywords include:"
-                "stretches: stretch(es), bond(s), r, and s; bends: bend(s), angle(s), b, and a;"
-                "torsions: torsions(t), tors, dihedral(s), t and d; oofps: oofp(s) and o."
-                "Any capitalization allowed. Singular, or plural allowed"
-            )
-
-def parse_ext_intco(coord, spec, molsys, **kwargs):
-    """Create an internal coordinate with user supplied forces. Adds to the proper fragment in the
-    molecular system"""
-
-    regex = create_coord_regex(coord, coord_type='standard')
-    coords = re.findall(regex, spec)
-    forces = re.split(regex, spec)
-
-    tmp_kwargs = kwargs.copy()
-
-    for coord_spec, force in zip(coords, forces[1:]):
-        # values should look like [atoms..., complex string] 
-        values = split_and_remove_symbols(coord_spec)
-        tmp_kwargs.update({"ext_force": force})
-
-        if isinstance(coord, cart.Cart):
-            tmp_kwargs.update({"xyz_spec": values[1]})
-            _create_all_cartesians(coord, values[0], molsys, **tmp_kwargs)
-        else:
-            _create_coord(coord, values, molsys, **tmp_kwargs)
-
-def parse_ranged_intco(coord, spec, molsys, **kwargs):
-    """Create a ranged coordinate. Adds to the proper fragment in the molecular system """
-
-    natoms = coord.natoms()
-    matches = parse_input(spec, coord, coord_type="ranged")
-
-    # Remove all parenthesis, commans, etc... and convert to list. the above regex check should
-    # guarantee than a parsing error is impossible
-    for match in matches:
-        # values should look like [atoms..., float, float] 
-        values = split_and_remove_symbols(match)  # the function used is decided above
-
-        # grab range and throw in kwargs to pass to coord constructor
-        limits = values[-2:]  # last two values are the range
-        tmp_kwargs = kwargs.copy()
-        tmp_kwargs.update({
-            "range_min": float(limits[0]) / coord.q_show_factor,
-            "range_max": float(limits[1]) / coord.q_show_factor
-        })
-        if isinstance(coord, cart.Cart):
-            tmp_kwargs.update({"xyz_spec": values[1]})  # add "X", "Y", "Z" individually
-            _create_all_cartesians(coord, values[:natoms], molsys, **tmp_kwargs)
-        else:
-            _create_coord(coord, values[:natoms], molsys, **tmp_kwargs)
-
-def parse_std_intco(coord, spec, molsys, **kwargs):
-    """Create a frozen internal coordinate or a regular (unconstrained coordinate). Adds to proper
-    fragment in the molecular system"""
-
-    natoms = coord.natoms()
-    type = "frozen" if kwargs.get("constraint") else ""
-    matches = parse_input(spec, coord, coord_type=type)
-
-    for match in matches:
-        # values should simply be a list[int] 
-        values = split_and_remove_symbols(match)
-
-        if isinstance(coord, cart.Cart):
-            tmp_kwargs = kwargs.copy()
-            tmp_kwargs.update({"xyz_spec": values[1]})
-            _create_all_cartesians(coord, values[:natoms], molsys, **tmp_kwargs)
-        else:
-            _create_coord(coord, values[:natoms], molsys, **kwargs)
-
-def split_and_remove_symbols(value, type='intco'):
-    """ Remove symbols based on type. If type is 'cart', remove any characters that are not
-    numerical one or x,y, or z. Otherwise remove all non alphanumerical characters and replace with
-    ' '. split by whitespace into a list. This cannot be called on a string including a
-    mathematical formula
-    
-    list
-        list[int] for frozen or standard coordinates
-        list[int|float] for ranged internal coordinates
-        list[int|char] for frozen or standard cartesian coordinates
-        list[int|char|float] for ranged cartesian coordinates
-    """
-    if type == 'cart':
-        # replace anything that is NOT in the set 0-9xyz .
-        value = re.sub(r"[^0-9xyz\.]", " ", value)
-    else:
-        value = re.sub(r"[^0-9\.]", " ", value)
-
-    value = value.split()
-    return value
-
-def _create_all_cartesians(coord, coord_atoms, molsys, **kwargs):
-    """ Call _create_coord for x, y, and/or z as needed """
-    for val in kwargs.pop("xyz_spec"):
-        tmp_kwargs = kwargs.copy()
-        tmp_kwargs.update({"xyz_in": val})
-        _create_coord(coord, coord_atoms, molsys, **tmp_kwargs)
-
-def _create_coord(coord, coord_atoms, molsys, **kwargs):
-    """ Creates an internal coordinate of type `coord` checked for a valid group of atoms and adds
-    it to a frag's intcos """
-
-    # convert atoms to zero based indexing here
-    atoms = [int(atom) - 1 for atom in coord_atoms]
-    try:
-        frag, frag_atoms = molsys.get_fragment_atom_indexes(atoms)
-    except ValueError as e:
-        raise OptError(
-            f"Can't create {coord.__name__} between atoms {coord_atoms} Atoms are not in the"
-            "same fragment. Please use DimerFrag coordinates to create custom interfragment"
-            "coordinates."
-        ) from e
-    new_coord = coord(*frag_atoms, **kwargs)
-    checked_coord_append(frag, new_coord)
-
-def checked_coord_append(frag, new_coord):
-    """ Check whether the coordinate is already present in the selected frag. If it is
-    replace it so that any updates to the coord are obeyed.
-    
-    Notes
-    -----
-    This works because the __eq__ for the internal coordindates does not compare constraints """
-
-    if new_coord not in frag.intcos:
-        frag.intcos.append(new_coord)
-    else:
-        # replace coordinate so that constraints are obeyed if desired
-        i = frag.intcos.index(new_coord)
-        frag.intcos[i] = new_coord
-
-def parse_input(spec, coord, coord_type):
-    """ Parses input with the regex cooresponding to coord and coord_type. Raises an error if
-    the string is not parsable or only partly parsable (parse_ext_force does not call this due to
-    issue in verifying a potentially arbitary math function)
-    """
-    print(f"Called parse_input with args {spec} {coord} {coord_type}")
-    regex = create_coord_regex(coord, coord_type)
-    tmp_spec = spec.lower()
-    matches = re.findall(regex, tmp_spec)
-    if not re.fullmatch(rf"(?:{regex})+", tmp_spec):
-        logger.critical("cannot fully parse %s%s for coord %s", coord_type, spec, coord.__name__)
-        logger.critical(
-            "Found the following valid substrings: %s", matches
-        )
-        raise OptError(f"cannot fully parse {coord_type}{spec} for coord: {coord.__name__}")
-    return matches
-
-# constants for creating the regexes below
-FLOAT = r"\d+\.\d+"
-SEP = r"\W+\s*?"
-SEP2 = r"\W*?\s*?"  # non greedy matching can be important in ext_force for equation parenthesis
-SEP3 = r"\W*\s*"
-INT = r"\d"
-CART_STR = r"(?:xyz|xy|xz|yz|x|y|z)"
-LABEL = r"(?:[srabtdo]|stre|stretch|bond|bend|angle|tors|torsion|dihedral|oofp)"
-
-def create_coord_regex(coord, coord_type):
-    """ Select the appropriate regex to use for matching string coordinate specification.
-    Each regex allows a range of non-alphanumeric characters to seperate the indices and floats
-
-    Notes
-    -----
-    SEP requires there be some kind of seperator. SEP2 allows the string to terminate
-    """
-
-    natoms = coord.natoms()
-    if coord_type == 'ranged':
-        if isinstance(coord, cart.Cart):
-            regex = rf"(?:{SEP})?{INT}{SEP}{CART_STR}{SEP}(?:{FLOAT}{SEP3}){{2}}\s*"
-        else:
-            regex = rf"{LABEL}?(?:{SEP})?(?:{INT}{SEP}){{{natoms}}}{SEP}(?:{FLOAT}{SEP3}){{2}}\s*"
-    else:
-        if isinstance(coord, cart.Cart):
-            regex = rf"(?:{SEP})?{INT}{SEP}{CART_STR}{SEP2}"
-        else:
-            regex = rf"{LABEL}?(?:{SEP})?(?:{INT}{SEP2}){{{natoms}}}"
-
-    print(regex)
-    return regex
diff --git a/optking/bend.py b/optking/bend.py
index 7e63974..120d682 100644
--- a/optking/bend.py
+++ b/optking/bend.py
@@ -61,10 +61,6 @@ class Bend(Simple):
 
         Simple.__init__(self, atoms, constraint, range_min, range_max, ext_force)
 
-    @classmethod
-    def natoms(cls):
-        return 3
-
     def __str__(self):
         if self.frozen:
             s = "*"
diff --git a/optking/cart.py b/optking/cart.py
index cdf55fd..1a22965 100644
--- a/optking/cart.py
+++ b/optking/cart.py
@@ -30,10 +30,6 @@ class Cart(Simple):
         atoms = (a,)
         Simple.__init__(self, atoms, constraint, range_min, range_max, ext_force)
 
-    @classmethod
-    def natoms(cls):
-        return 1
-
     def __str__(self):
         if self.frozen:
             s = "*"
diff --git a/optking/misc.py b/optking/misc.py
index ee56f7c..af68f24 100644
--- a/optking/misc.py
+++ b/optking/misc.py
@@ -1,6 +1,5 @@
 import logging
 import math
-from itertools import zip_longest
 
 import numpy as np
 import qcelemental as qcel
@@ -289,20 +288,3 @@ def int_xyz_fx_string(inString, Nint=1):
         logger.debug(entry)
 
     return outList
-
-def grouper(iterable, n, *, incomplete="fill", fillvalue=None):
-    """Collect data into non-overlapping fixed-length chunks or blocks.
-    Itertools recipe until batched becomes standard"""
-    # grouper('ABCDEFG', 3, fillvalue='x') → ABC DEF Gxx
-    # grouper('ABCDEFG', 3, incomplete='strict') → ABC DEF ValueError
-    # grouper('ABCDEFG', 3, incomplete='ignore') → ABC DEF
-    iterators = [iter(iterable)] * n
-    match incomplete:
-        case "fill":
-            return zip_longest(*iterators, fillvalue=fillvalue)
-        case "strict":
-            return zip(*iterators, strict=True)
-        case "ignore":
-            return zip(*iterators)
-        case _:
-            raise ValueError("Expected fill, strict, or ignore")
diff --git a/optking/molsys.py b/optking/molsys.py
index 5732597..a2e1024 100644
--- a/optking/molsys.py
+++ b/optking/molsys.py
@@ -214,61 +214,13 @@ class Molsys(object):
         start = self.frag_1st_atom(iF)
         return slice(start, start + self._fragments[iF].natom)
 
+    # accepts absolute atom index, returns fragment index
     def atom2frag_index(self, atom_index):
-        """ Returns the index of the fragment that contains atom `atom_index` 
-
-        Parameters
-        ----------
-        atom_index: int
-         the absolute atom index
-        
-        Returns
-        -------
-        int: fragment index
-
-        Notes
-        -----
-        fragments: [a, b, c, d], [e, f, g, h]
-        atom 6 == g -> 1
-
-        """
-
         for iF in range(self.nfragments):
             if atom_index in self.frag_atom_range(iF):
                 return iF
         raise OptError("atom2frag_index: atom_index impossibly large")
 
-    def get_fragment_atom_indexes(self, atom_indexes):
-        """ Translates overall atomic indexing to fragment based indexing 
-        
-        Notes
-        -----
-        atoms: [a, b, c, d], [e, f, g, h]
-        overall: [0, 1, 2, 3], [4, 5, 6, 7]
-        frag: [0, 1, 2, 3], [0, 1, 2, 3]
-        atom 6 == g -> 2
-
-        Returns
-        -------
-        frag.Frag:
-            the fragment the atoms lie in.
-        list[int]:
-            a list of indices for the atoms within a fragment
-
-        Raises
-        ------
-        ValueError:
-            If the atom indices span 2 or more fragments (can't find all atoms in a single
-            fragments range)
-
-        """
-
-        for f in range(self.nfragments):
-            current_range = self.frag_atom_range(f)
-            if atom_indexes[0] in current_range:
-               # translate overall indexing to index within the fragments range
-               return self.fragments[f], [current_range.index(i) for i in atom_indexes]
-
     # Given a list of atoms, return all the fragments to which they belong
     def atom_list2unique_frag_list(self, atomList):
         fragList = []
diff --git a/optking/oofp.py b/optking/oofp.py
index 3033acf..8887c0f 100644
--- a/optking/oofp.py
+++ b/optking/oofp.py
@@ -69,10 +69,6 @@ class Oofp(Simple):
         # else:
         #    self.symbolic_coord = coordinates.OutOfPlane(atoms)
 
-    @classmethod
-    def natoms(cls):
-        return 4
-
     def __str__(self):
         if self.frozen:
             s = "*"
diff --git a/optking/optimize.py b/optking/optimize.py
index 05ae300..9d6f805 100644
--- a/optking/optimize.py
+++ b/optking/optimize.py
@@ -651,11 +651,6 @@ def make_internal_coords(o_molsys: Molsys, params: op.OptParams):
     connectivity = addIntcos.connectivity_from_distances(o_molsys.geom, o_molsys.Z)
     logger.debug("Connectivity Matrix\n" + print_mat_string(connectivity))
 
-    if params.opt_coordinates == 'CUSTOM':
-        if params.custom_coordinates is None:
-            raise ValueError("Custom coordinates were requested but none were provided")
-        addIntcos.add_coordinates_from_dict(params.custom_coordinates, o_molsys)
-
     if params.frag_mode == "SINGLE":
         try:
             # Make a single, supermolecule.
@@ -665,24 +660,17 @@ def make_internal_coords(o_molsys: Molsys, params: op.OptParams):
             o_molsys.augment_connectivity_to_single_fragment(connectivity)
             o_molsys.consolidate_fragments()  # collapse into one frag
 
-            # single frag - add coordinates directly
-            frag = o_molsys.fragments[0]
-
             if params.opt_coordinates in ["INTERNAL", "REDUNDANT", "BOTH"]:
-                frag.add_intcos_from_connectivity(connectivity)
+                o_molsys.fragments[0].add_intcos_from_connectivity(connectivity)
                 if params.add_auxiliary_bonds:
-                    frag.add_auxiliary_bonds(connectivity)
+                    o_molsys.fragments[0].add_auxiliary_bonds(connectivity)
         except AlgError as error:
-            frag._intcos = []
+            o_molsys.fragments[0]._intcos = []
             if error.oofp_failures or error.linear_bends or error.linear_torsions:
                 params.opt_coordinates = "CARTESIAN"
 
         if params.opt_coordinates in ["CARTESIAN", "BOTH"]:
-            frag.add_cartesian_intcos()
-
-        if params.custom_coordinates:
-            customs = addIntcos.add_coordinates_from_dict(params.custom_coordinates)
-            frag.add_custom_coordinates(customs)
+            o_molsys.fragments[0].add_cartesian_intcos()
 
     elif params.frag_mode == "MULTI":
 
diff --git a/optking/optparams.py b/optking/optparams.py
index 665cc67..965ab23 100644
--- a/optking/optparams.py
+++ b/optking/optparams.py
@@ -37,7 +37,6 @@ allowedStringOptions = {
         "NATURAL",
         "CARTESIAN",
         "BOTH",
-        "CUSTOM",
     ),
     "irc_direction": ("FORWARD", "BACKWARD"),
     "g_convergence": (
@@ -126,11 +125,6 @@ class OptParams(object):
         # CARTESIAN uses only cartesian coordinates.
         # BOTH uses both redundant and cartesian coordinates.
         self.opt_coordinates = uod.get("OPT_COORDINATES", "REDUNDANT")
-        # use dictionary of custom coordinates
-        # if dictionary is provided while OPT_COORDINATES != 'CUSTOM'
-        # append coordinates but do not override
-        self.custom_coordinates = uod.get("CUSTOM_COORDINATES", {})
-
         # Do follow the initial RFO vector after the first step?
         self.rfo_follow_root = uod.get("RFO_FOLLOW_ROOT", False)
         # Root for RFO to follow, 0 being lowest (typical for a minimum)
diff --git a/optking/simple.py b/optking/simple.py
index 90478d6..dcbe702 100644
--- a/optking/simple.py
+++ b/optking/simple.py
@@ -2,7 +2,7 @@ from abc import ABC, abstractmethod
 
 from .exceptions import AlgError, OptError
 
-supported_constraint_types = ("free", "frozen", "ranged", "ext_force")
+supported_constraint_types = ("free", "frozen", "ranged")
 
 
 class Simple(ABC):
@@ -16,10 +16,6 @@ class Simple(ABC):
         self._range_max = range_max
         self._ext_force = ext_force
 
-    @classmethod
-    def natoms(cls):
-        raise NotImplementedError("Coordinate needs to implement this method")
-
     @property
     def atoms(self):
         return self._atoms
@@ -68,7 +64,7 @@ class Simple(ABC):
 
     @property
     def has_ext_force(self):
-        return self.constraint == 'ext_force'
+        return bool(self._ext_force is not None)
 
     @property
     def ext_force(self):
@@ -77,17 +73,7 @@ class Simple(ABC):
     # could add checking here later
     @ext_force.setter
     def ext_force(self, eqn):
-        try:
-            import sympy
-        except ImportError:
-            logger.critical("To use custom forces with coordinates, please install sympy")
-            logger.critical(
-                "Please see sympy's documentation for further information on installation and for"
-                "details on parsing."
-                "https://docs.sympy.org/latest/install.html"
-                "https://docs.sympy.org/latest/modules/parsing.html"
-            )
-        self._ext_force = sympy.parsing.sympy_parser.parse_expr(eqn, evaluate=False)
+        self._ext_force = eqn
 
     def ext_force_val(self, geom):
         val = self.q_show(geom)  # angstroms or degrees
diff --git a/optking/stre.py b/optking/stre.py
index d432107..abbae9a 100644
--- a/optking/stre.py
+++ b/optking/stre.py
@@ -50,10 +50,6 @@ class Stre(Simple):
 
         Simple.__init__(self, atoms, constraint, range_min, range_max, ext_force)
 
-    @classmethod
-    def natoms(cls):
-        return 2
-
     def __str__(self):
         if self.frozen:
             s = "*"
diff --git a/optking/tests/test_dimers_h2o.py b/optking/tests/test_dimers_h2o.py
index 21897e5..862f2ba 100644
--- a/optking/tests/test_dimers_h2o.py
+++ b/optking/tests/test_dimers_h2o.py
@@ -102,41 +102,3 @@ def test_dimers_h2o_auto(check_iter, option, iter):  # auto reference pt. creati
     assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from afar, auto")
 
     utils.compare_iterations(json_output, iter, check_iter)
-
-
-@pytest.mark.dimers
-@pytest.mark.parametrize("option, iter", [("gau_tight", 13), ("interfrag_tight", 11)])
-def test_dimers_h2o_freeze(check_iter, option, iter):  # auto reference pt. creation
-    h2oD = psi4.geometry(
-        """
-      0 1
-      H   0.280638    -1.591779    -0.021801
-      O   0.351675    -1.701049     0.952490
-      H  -0.464013    -1.272980     1.251761
-      --
-      0 1
-      H  -0.397819    -1.918411    -2.373012
-      O  -0.105182    -1.256691    -1.722965
-      H   0.334700    -0.589454    -2.277374
-      nocom
-      noreorient
-    """
-    )
-
-    psi4.core.clean_options()
-    psi4_options = {
-        "basis": "aug-cc-pvdz",
-        "geom_maxiter": 40,
-        "frag_mode": "MULTI",
-        "g_convergence": f"{option}",
-        "frozen_bend": "1 2 3 4 5 6" # Freeze both bends
-    }
-    psi4.set_options(psi4_options)
-
-    newOptParams = {"interfrag_collinear_tol": 0.2}  # increase to prevent too colinear reference points
-    json_output = optking.optimize_psi4("mp2", **newOptParams)
-
-    E = json_output["energies"][-1]
-    assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from afar, auto")
-
-    utils.compare_iterations(json_output, iter, check_iter)
diff --git a/optking/tors.py b/optking/tors.py
index e2b3e35..fd8be62 100644
--- a/optking/tors.py
+++ b/optking/tors.py
@@ -58,10 +58,6 @@ class Tors(Simple):
 
         Simple.__init__(self, atoms, constraint, range_min, range_max, ext_force)
 
-    @classmethod
-    def natoms(cls):
-        return 4
-
     def __str__(self):
         if self.frozen:
             s = "*"
