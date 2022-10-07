import logging
from itertools import combinations, permutations
from typing import List

import numpy as np
import qcelemental as qcel

from . import dimerfrag, frag, v3d
from .addIntcos import add_cartesian_intcos, connectivity_from_distances
from .exceptions import OptError
from .linearAlgebra import symm_mat_inv
from .printTools import print_array_string, print_mat_string
from . import optparams as op
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")


class Molsys(object):
    def __init__(self, fragments, dimer_intcos=None):
        """The molecular system consisting of a collection of fragments

        Parameters
        ----------
        fragments : list[Frag]
        """
        # def __init__(self, fragments, fb_fragments=None, intcos=None, multiplicity=1):
        # ordinary fragments with internal structure

        if fragments:
            self._fragments = fragments
        else:
            self._fragments = []

        if dimer_intcos:
            self._dimer_intcos = dimer_intcos
        else:
            self._dimer_intcos = []

        # fixed body fragments defined by Euler/rotation angles
        # self._fb_fragments = []
        # if fb_fragments:
        #    self._fb_fragments = fb_fragments

    def __str__(self):
        s = ""
        for i, frag in enumerate(self._fragments):
            s += f"\n\t {'===> Fragment':>40} {i + 1} <== \n"
            s += str(frag)
        self.update_dimer_intco_reference_points()
        if self._dimer_intcos:
            s += f"\n\t{'==> Dimer Coordinates <==':^80}\n"
        for dimer in self._dimer_intcos:
            s += str(dimer)
        return s

    @classmethod
    def from_schema(cls, qc_molecule):
        """Creates optking molecular system from JSON input.

        Parameters
        ----------
        qc_molecule: dict
            molecule key in MOLSSI QCSchema
            see http://molssi-qc-schema.readthedocs.io/en/latest/auto_topology.html

        Returns
        -------
        cls:
            molsys cls consists of list of Frags
        """
        logger.debug("\tGenerating molecular system for optimization from QC Schema.\n")

        geom = np.asarray(qc_molecule["geometry"])
        geom = geom.reshape(-1, 3)

        z_list = [qcel.periodictable.to_Z(atom) for atom in qc_molecule["symbols"]]

        masses_list = qc_molecule.get("masses")
        if masses_list is None:
            masses_list = [qcel.periodictable.to_mass(atom) for atom in qc_molecule["symbols"]]

        frags = []
        if "fragments" in qc_molecule:
            for fr in qc_molecule["fragments"]:
                frags.append(frag.Frag(np.array(z_list)[fr], geom[fr], np.array(masses_list)[fr]))
        else:
            frags.append(frag.Frag(z_list, geom, masses_list))

        return cls(frags)

    @staticmethod
    def from_psi4(mol):
        """Creates a optking molecular system from psi4 mol. Note that not all information
        is preserved.
        Parameters
        ----------
        mol: object
            psi4 mol
        Returns
        -------
        cls :
            optking molecular system: list of fragments
        """

        import psi4

        logger.debug("\tConverting psi4 molecular system to schema")

        if not isinstance(mol, psi4.core.Molecule):
            logger.critical("from_psi4 cannot handle a non psi4 molecule")
            raise OptError("Cannot make molecular system from this molecule")

        qc_mol = mol.to_schema(dtype=2)
        qc_mol.update({"fix_com": True, "fix_orientation": True})
        opt_mol = Molsys.from_schema(qc_mol)
        return opt_mol, qc_mol

    def to_schema(self):
        mol_dict = {"symbols": self.atom_symbols,
                    "geometry": self.geom.flat,
                    "atomic_numbers": self.Z,
                    "mass_numbers": self.masses,
                    "fix_com": True,
                    "fix_orientation": True}
        return qcel.models.Molecule(**mol_dict)

    def to_dict(self):
        d = {
            "fragments": [f.to_dict() for f in self._fragments],
            "dimer_intcos": [di.to_dict() for di in self._dimer_intcos],
        }
        return d

    @classmethod
    def from_dict(cls, d):
        if "fragments" not in d:
            raise OptError("'fragments' key missing from input dict")
        frags = [frag.Frag.from_dict(F) for F in d["fragments"]]

        if "dimer_intcos" in d:
            dimers = [dimerfrag.DimerFrag.from_dict(DF) for DF in d["dimer_intcos"]]
        else:
            dimers = None

        return cls(frags, dimers)

    @property
    def natom(self) -> int:
        return sum(fragment.natom for fragment in self._fragments)

    @property
    def nfragments(self) -> int:
        return len(self._fragments)
        # return len(self._fragments) + len(self._fb_fragments)

    @property
    def frag_natoms(self) -> List[int]:
        return [fragment.natom for fragment in self._fragments]

    @property
    def fragments(self) -> List[frag.Frag]:
        return self._fragments

    @property
    def dimer_intcos(self) -> List[dimerfrag.DimerFrag]:
        return self._dimer_intcos

    @property
    def dimer_psuedo_frags(self) -> List[frag.Frag]:
        return [dimer.pseudo_frag for dimer in self._dimer_intcos]

    @property
    def all_fragments(self):
        return self._fragments + self.dimer_psuedo_frags

    # @property
    # def intcos(self):
    #    """ Collect intcos for all fragments. Add dimer coords to end.
    #    Returns
    #    -------
    #    """
    #    logger.warning("""This method is currently implemented as a last resort used as a last
    #                   resort. Should be safe assuming no dimer coordinates, otherwise unknown.""")
    #    coords = [coord for f in self._fragments for coord in f.intcos]
    #    for d_coord in self.dimer_intcos:
    #        coords.append(d_coord)
    #    return coords

    @property
    def intco_lbls(self):
        lbls = [str(coord) for f in self._fragments for coord in f.intcos]
        for DI in self.dimer_intcos:
            for coord in DI.pseudo_frag.intcos:
                lbls.append("Dimer({:d},{:d})".format(DI.A_idx + 1, DI.B_idx + 1) + str(coord))
        return lbls

    # Return overall index of first atom in fragment, beginning 0,1,...
    # For last fragment returns one past the end.
    def frag_1st_atom(self, iF):
        if iF > len(self._fragments):
            return ValueError()
        start = 0
        for i in range(0, iF):
            start += self._fragments[i].natom
        return start

    def frag_atom_range(self, iF):
        start = self.frag_1st_atom(iF)
        return range(start, start + self._fragments[iF].natom)

    def frag_atom_slice(self, iF):
        start = self.frag_1st_atom(iF)
        return slice(start, start + self._fragments[iF].natom)

    # accepts absolute atom index, returns fragment index
    def atom2frag_index(self, atom_index):
        for iF in range(self.nfragments):
            if atom_index in self.frag_atom_range(iF):
                return iF
        raise OptError("atom2frag_index: atom_index impossibly large")

    # Given a list of atoms, return all the fragments to which they belong
    def atom_list2unique_frag_list(self, atomList):
        fragList = []
        for a in atomList:
            f = self.atom2frag_index(a)
            if f not in fragList:
                fragList.append(f)
        return fragList

    @property
    def geom(self):
        """cartesian geometry [a0]"""
        geom = np.zeros((self.natom, 3))
        for iF, F in enumerate(self._fragments):
            row = self.frag_1st_atom(iF)
            geom[row : (row + F.natom), :] = F.geom
        return geom

    @geom.setter
    def geom(self, newgeom):
        """ setter for geometry"""
        for iF, F in enumerate(self._fragments):
            row = self.frag_1st_atom(iF)
            F.geom[:] = newgeom[row : (row + F.natom), :]

    def frag_geom(self, iF):
        """cartesian geometry for fragment i"""
        return self._fragments[iF].geom
        # return copy instead?
        # using in displace_molsys

    @property
    def masses(self):
        m = np.zeros(self.natom)
        for iF, F in enumerate(self._fragments):
            m[self.frag_atom_slice(iF)] = F.masses
        return m

    @property
    def Z(self):
        z = [0 for i in range(self.natom)]
        for iF, F in enumerate(self._fragments):
            first = self.frag_1st_atom(iF)
            z[first : (first + F.natom)] = F.Z
        return z

    # Needed?  may make more sense to loop over fragments
    # @property
    # def intcos(self):
    #    _intcos = []
    #    for F in self._fragments:
    #        _intcos += F.intcos
    #    return _intcos

    @property
    def num_intcos(self):

        nintco_list = [f.num_intcos for f in self.all_fragments]
        return sum(nintco_list)

    @property
    def num_intrafrag_intcos(self):

        nintco_list = [f.num_intcos for f in self.fragments]
        return sum(nintco_list)

    @property
    def intcos_present(self):
        for fragment in self.all_fragments:
            if fragment.intcos:
                return True

        return False

    @property
    def frozen_intco_list(self):
        """Determine vector with 1 for any frozen internal coordinate"""
        frozen = np.zeros(self.num_intcos, dtype=bool)
        cnt = 0

        for f in self.all_fragments:
            for intco in f.intcos:
                if intco.frozen:
                    frozen[cnt] = True
                cnt += 1
        return frozen

    # Used to zero out forces.  For any ranged intco, indicate frozen if
    # within 0.1% of boundary and its corresponding force is in that direction.
    def ranged_frozen_intco_list(self, fq):
        """Determine vector with 1 for any ranged intco that is at its limit"""
        qvals = self.q()
        frozen = np.zeros(self.num_intcos, dtype=bool)
        cnt = 0

        for f in self.all_fragments:
            for intco in f.intcos:
                if intco.ranged:
                    tol = 0.001 * (intco.range_max - intco.range_min)
                    if np.fabs(qvals[cnt] - intco.range_max) < tol and fq[cnt] > 0:
                        frozen[cnt] = True
                    elif np.fabs(qvals[cnt] - intco.range_min) < tol and fq[cnt] < 0:
                        frozen[cnt] = True
                cnt += 1

        return frozen

    def constraint_matrix(self, fq=None):
        """Returns constraint matrix with 1 on diagonal for frozen coordinates"""
        frozen = self.frozen_intco_list

        if fq is not None:
            range_frozen = self.ranged_frozen_intco_list(fq)
            frozen = np.logical_or(frozen, range_frozen)

        if np.any(frozen):
            return np.diagflat(frozen)
        else:
            return None

    # returns the index of the first internal coordinate belonging to fragment
    def frag_1st_intco(self, iF):
        if iF >= len(self._fragments):
            return ValueError()
        start = 0
        for i in range(0, iF):
            start += self._fragments[i].num_intcos
        return start

    def frag_intco_range(self, iF):
        start = self.frag_1st_intco(iF)
        return range(start, start + self._fragments[iF].num_intcos)

    def frag_intco_slice(self, iF):
        start = self.frag_1st_intco(iF)
        return slice(start, start + self._fragments[iF].num_intcos)

    # Given the index i looping through the list of dimer coordinate (sets),
    # returns the total intco row number for the first/start of the dimer coordinate.
    def dimerfrag_1st_intco(self, iDI):
        # we assume the intrafragment coordinates come first
        N = sum(F.num_intcos for F in self._fragments)
        for i in range(0, iDI):
            N += self._dimer_intcos[i].num_intcos
        return N

    def dimerfrag_intco_range(self, iDI):
        start = self.dimerfrag_1st_intco(iDI)
        return range(start, start + self._dimer_intcos[iDI].num_intcos)

    def dimerfrag_intco_slice(self, iDI):
        start = self.dimerfrag_1st_intco(iDI)
        return slice(start, start + self._dimer_intcos[iDI].num_intcos)

    def print_intcos(self):
        for frag_index, frag in enumerate(self.all_fragments):
            logger.info("Fragment %d\n", frag_index + 1)
            frag.print_intcos()

    # If connectivity is provided, only intrafragment connections
    # are used.  Interfragment connections are ignored here.
    # def add_intcos_from_connectivity(self, C=None):
    #    for F in self._fragments:
    #        if C is None:
    #            C = F.connectivity_from_distances()
    #        F.add_intcos_from_connectivity(C)

    def add_cartesian_intcos(self):
        for F in self._fragments:
            add_cartesian_intcos(F._intcos, F._geom)

    def print_geom(self):
        """Returns a string of the geometry for logging in [a0]"""
        for iF, F in enumerate(self._fragments):
            logger.info("\tFragment %d\n" % (iF + 1))
            F.print_geom()

    def show_geom(self):
        """Return a string of the geometry in [A]"""
        molsys_geometry = ""
        for iF, F in enumerate(self._fragments):
            molsys_geometry += "\tFragment {:d} (Ang)\n\n".format(iF + 1)
            molsys_geometry += F.show_geom()
        return molsys_geometry

    @property
    def atom_symbols(self):
        symbol_list = []
        for F in self._fragments:
            symbol_list += F.get_atom_symbol_list()
        return symbol_list

    @property
    def fragments_atom_list(self):
        l = []
        for iF in range(self.nfragments):
            fl = [i for i in self.frag_atom_range(iF)]
            l.append(fl)
        return l

    def q(self):
        """Returns internal coordinate values in au as list."""
        vals = []
        for F in self._fragments:
            vals += F.q()
        self.update_dimer_intco_reference_points()
        for DI in self._dimer_intcos:
            vals += DI.q()
        return vals

    def q_array(self):
        """Returns internal coordinate values in au as array."""
        return np.asarray(self.q())

    def q_show(self):
        """returns internal coordinates values in Angstroms/degrees as list."""
        vals = []
        for F in self._fragments:
            vals += F.q_show()

        for DI in self._dimer_intcos:
            vals += DI.q_show()
        return vals

    def q_show_array(self):
        """returns internal coordinates values in Angstroms/degrees as array."""
        return np.asarray(self.q_show())

    def consolidate_fragments(self):
        if self.nfragments == 1:
            return
        logger.info("\tConsolidating multiple fragments into one for optimization.")
        Z = self._fragments[0].Z
        g = self._fragments[0].geom
        m = self._fragments[0].masses
        for i in range(1, self.nfragments):
            Z = np.concatenate((Z, self._fragments[i].Z))
            g = np.concatenate((g, self._fragments[i].geom))
            m = np.concatenate((m, self._fragments[i].masses))
        # self._fragments.append(consolidatedFrag)
        del self._fragments[:]
        consolidatedFrag = frag.Frag(Z, g, m)
        self._fragments.append(consolidatedFrag)

    def split_fragments_by_connectivity(self):
        """ Split any fragment not connected by bond connectivity."""
        tempZ = np.copy(self.Z)
        tempGeom = np.copy(self.geom)
        tempMasses = np.copy(self.masses)

        newFragments = []
        for F in self._fragments:
            C = connectivity_from_distances(F.geom, F.Z)
            atomsToAllocate = list(reversed(range(F.natom)))
            while atomsToAllocate:
                frag_atoms = [atomsToAllocate.pop()]

                more_found = True
                while more_found:
                    more_found = False
                    addAtoms = []
                    for A in frag_atoms:
                        for B in atomsToAllocate:
                            if C[A, B]:
                                if B not in addAtoms:
                                    addAtoms.append(B)
                                more_found = True
                    for a in addAtoms:
                        frag_atoms.append(a)
                        atomsToAllocate.remove(a)

                frag_atoms.sort()
                subNatom = len(frag_atoms)
                subZ = [0] * subNatom
                subGeom = np.zeros((subNatom, 3))
                subMasses = [0] * subNatom
                for i, I in enumerate(frag_atoms):
                    subZ[i] = tempZ[I]
                    subGeom[i, 0:3] = tempGeom[I, 0:3]
                    subMasses[i] = tempMasses[I]
                newFragments.append(frag.Frag(subZ, subGeom, subMasses))

        del self._fragments[:]
        self._fragments = newFragments

    def purge_interfragment_connectivity(self, C):
        for f1, f2 in permutations([i for i in range(self.nfragments)], 2):
            for a in self.frag_atom_range(f1):
                for b in self.frag_atom_range(f2):
                    C[a, b] = 0.0
        return

    # Supplements a connectivity matrix to connect all fragments.  Assumes the
    # definition of the fragments has ALREADY been determined before function called.
    def augment_connectivity_to_single_fragment(self, C):
        logger.debug("\tAugmenting connectivity matrix to join fragments.")
        fragAtoms = []
        geom = self.geom
        for iF, F in enumerate(self._fragments):
            fragAtoms.append(range(self.frag_1st_atom(iF), self.frag_1st_atom(iF) + F.natom))

        # Which fragments are connected?
        nF = self.nfragments
        if self.nfragments == 1:
            return

        frag_connectivity = np.zeros((nF, nF))
        for iF in range(nF):
            frag_connectivity[iF, iF] = 1

        Z = self.Z

        scale_dist = 1.3
        all_connected = False
        while not all_connected:
            for f2 in range(nF):
                for f1 in range(f2):
                    if frag_connectivity[f1][f2]:
                        continue  # already connected
                    minVal = 1.0e12

                    # Find closest 2 atoms between fragments.
                    for f1_atom in fragAtoms[f1]:
                        for f2_atom in fragAtoms[f2]:
                            tval = v3d.dist(geom[f1_atom], geom[f2_atom])
                            if tval < minVal:
                                minVal = tval
                                i = f1_atom
                                j = f2_atom

                    Rij = v3d.dist(geom[i], geom[j])
                    R_i = qcel.covalentradii.get(Z[i], missing=4.0)
                    R_j = qcel.covalentradii.get(Z[j], missing=4.0)
                    if Rij > scale_dist * (R_i + R_j):
                        # ignore this as too far - for starters.  may have A-B-C situation.
                        continue

                    logger.info("\tConnecting fragments with atoms %d and %d" % (i + 1, j + 1))
                    C[i][j] = C[j][i] = True
                    frag_connectivity[f1][f2] = frag_connectivity[f2][f1] = True

                    # Now check for possibly symmetry-related atoms which are just as close
                    # We need them all to avoid symmetry breaking.
                    for f1_atom in fragAtoms[f1]:
                        for f2_atom in fragAtoms[f2]:
                            if f1_atom == i and f2_atom == j:  # already have this one
                                continue
                            tval = v3d.dist(geom[f1_atom], geom[f2_atom])
                            if np.fabs(tval - minVal) < 1.0e-10:
                                i = f1_atom
                                j = f2_atom
                                logger.info("\tAlso, with atoms %d and %d\n" % (i + 1, j + 1))
                                C[i][j] = C[j][i] = True

            # Test whether all frags are connected using current distance threshold
            if np.sum(frag_connectivity[0]) == nF:
                logger.info("\tAll fragments are connected in connectivity matrix.")
                all_connected = True
            else:
                scale_dist += 0.2
                logger.info("\tIncreasing scaling to %6.3f to connect fragments." % scale_dist)
        return

    def distance_matrix(self):
        xyz = self.geom
        R = np.zeros((self.natom, self.natom))
        for i, j in combinations(range(self.natom), r=2):
            R[i, j] = R[j, i] = v3d.dist(xyz[i], xyz[j])
        return R

    # Given fragment numbers A and B, determine the closest two atoms between
    # the fragments; return the local/fragment index for both.
    def closest_atoms_between_2_frags(self, A, B):
        self.distance_matrix()
        fragAtoms = self.fragments_atom_list
        closestR = 1e10
        R = self.distance_matrix()
        for f1_atom in fragAtoms[A]:
            for f2_atom in fragAtoms[B]:
                if R[f1_atom, f2_atom] < closestR:
                    closestR = R[f1_atom, f2_atom]
                    save = (f1_atom, f2_atom)
        return save[0] - self.frag_1st_atom(A), save[1] - self.frag_1st_atom(B)

    def clear(self):
        self._fragments.clear()
        # self._fb_fragments.clear()

    def update_dimer_intco_reference_points(self):
        for DI in self._dimer_intcos:
            xA = self.frag_geom(DI.A_idx)
            xB = self.frag_geom(DI.B_idx)
            DI.update_reference_geometry(xA, xB)

    def update_dihedral_orientations(self):
        """ See description in Fragment class. """
        for F in self._fragments:
            F.update_dihedral_orientations()
        self.update_dimer_intco_reference_points()
        for DI in self._dimer_intcos:
            DI.pseudo_frag.update_dihedral_orientations()

    def fix_bend_axes(self):
        """ See description in Fragment class. """
        for F in self._fragments:
            F.fix_bend_axes()
        self.update_dimer_intco_reference_points()
        for DI in self._dimer_intcos:
            DI.pseudo_frag.fix_bend_axes()

    def unfix_bend_axes(self):
        """ See description in Fragment class. """
        for F in self._fragments:
            F.unfix_bend_axes()
        for DI in self._dimer_intcos:
            DI.pseudo_frag.unfix_bend_axes()

    def interfrag_dq_discontinuity_correction(self, dq):
        for iDI, DI in enumerate(self._dimer_intcos):
            DI.dq_discontinuity_correction(dq[self.dimerfrag_intco_slice(iDI)])

    # Returns mass-weighted Bmatrix if use_masses is True.
    def Bmat(self, massWeight=False):
        # Allocate memory for full system.
        Nint = self.num_intcos
        Ncart = 3 * self.natom
        B = np.zeros((Nint, Ncart))

        for iF, F in enumerate(self._fragments):
            fB = F.Bmat()
            cart_offset = 3 * self.frag_1st_atom(iF)
            intco_offset = self.frag_1st_intco(iF)

            for i in range(F.num_intcos):
                for xyz in range(3 * F.natom):
                    B[intco_offset + i, cart_offset + xyz] = fB[i, xyz]

        if self._dimer_intcos:
            # xyz = self.geom
            for i, DI in enumerate(self._dimer_intcos):
                # print('Aidx:' + str(DI.A_idx) )
                A1stAtom = self.frag_1st_atom(DI.A_idx)
                B1stAtom = self.frag_1st_atom(DI.B_idx)
                Axyz = self.frag_geom(DI.A_idx)
                Bxyz = self.frag_geom(DI.B_idx)
                DI.Bmat(Axyz, Bxyz, B[self.dimerfrag_intco_slice(i)], 3 * A1stAtom, 3 * B1stAtom)  # column offsets

        if massWeight:
            sqrtm = np.broadcast_to(np.repeat(np.sqrt(self.masses), 3), (Nint, Ncart))
            B[:] = np.divide(B, sqrtm)
        return B

    def q_show_forces(self, forces):
        """ Returns scaled forces as array. """

        c = [intco.f_show_factor for f in self.all_fragments for intco in f.intcos]
        c = np.asarray(c)
        qaJ = c * forces
        return qaJ

    def Gmat(self, massWeight=False):
        """Calculates BuB^T (calculates B matrix)

        Parameters
        ----------
        masses : List, optional

        """
        B = self.Bmat(massWeight)
        return np.dot(B, B.T)

    def gradient_to_internals(self, g_x, coeff=1.0, B=None, useMasses=False):
        """Transform cartesian gradient to internals
        Parameters
        ----------
        g_x : np.ndarray
            (3nat, 1) cartesian gradient
        coeff : float
            prefactor coefficient; -1 for forces
        B : np.ndarray, optional
            B matrix to use
        useMasses : boolean
            instead of identity, use u = 1/masses in transformation

        Returns
        -------
        ndarray
            gradient in internal coordinates (coeff==1)
        Notes
        -----
        g_q = (BuB^T)^(-1)*B*g_x

        """
        if not self.intcos_present or self.natom == 0:
            return np.zeros(0)

        if B is None:
            B = self.Bmat()

        if useMasses:
            u = np.diag(np.repeat(1.0 / self.masses, 3))
            G = np.dot(np.dot(B, u), B.T)
            Ginv = symm_mat_inv(G, redundant=True)
            g_q = coeff * np.dot(np.dot(np.dot(Ginv, B), u), g_x)
        else:
            G = np.dot(B, B.T)
            Ginv = symm_mat_inv(G, redundant=True)
            g_q = coeff * np.dot(np.dot(Ginv, B), g_x)

        return g_q

    def hessian_to_internals(self, H, g_x=None, useMasses=False):
        """converts the hessian from cartesian coordinates into internal coordinates
        Hq = A^t (Hxy - Kxy) A, where K_xy = sum_q ( grad_q[I] d^2(q_I)/(dx dy)
        and A = (BuB^t)^-1 Bu


        Parameters
        ----------
        H : np.ndarray
            Hessian in cartesians
        g_x : np.ndarray
            (nat, 3) gradient in cartesians (optional)
        massWeight : boolean
            whether to keep arbitrary transformation matrix u=I or use 1/mass_i
            as in spectroscopy or cases where rotations/translations matter.

        Returns
        -------
        Hq : np.ndarray
            hessian in internal coordinates
        """
        logger.info("Converting Hessian from cartesians to internals.")

        B = self.Bmat()

        if useMasses:
            u = np.diag(np.repeat(1.0 / self.masses, 3))
            G = np.dot(np.dot(B, u), B.T)
            Ginv = symm_mat_inv(G, redundant=True)
            Atranspose = np.dot(np.dot(Ginv, B), u)
        else:
            G = np.dot(B, B.T)
            Ginv = symm_mat_inv(G, redundant=True)
            Atranspose = np.dot(Ginv, B)

        Hworking = H.copy()
        if g_x is None:  # A^t Hxy A
            logger.info("Neglecting force/B-matrix derivative term, only correct at stationary points.")
        else:  # A^t (Hxy - Kxy) A;    K_xy = sum_q ( grad_q[I] d^2(q_I)/(dx dy) )
            logger.info("Including force/B-matrix derivative term.\n")

            g_q = self.gradient_to_internals(g_x, useMasses=useMasses)

            for iF, F in enumerate(self._fragments):
                dq2dx2 = np.zeros((3 * F.natom, 3 * F.natom))
                geom = F.geom
                # Find start index for this fragment
                cart_offset = 3 * self.frag_1st_atom(iF)
                intco_offset = self.frag_1st_intco(iF)

                for iIntco, Intco in enumerate(F.intcos):
                    dq2dx2[:] = 0
                    Intco.Dq2Dx2(geom, dq2dx2)  # d^2(q_I)/ dx_i dx_j

                    # Loop over Cartesian pairs in fragment
                    for a in range(3 * F.natom):
                        for b in range(3 * F.natom):
                            Hworking[cart_offset + a, cart_offset + b] -= g_q[intco_offset + iIntco] * dq2dx2[a, b]

            # TODO: dimer coordinates, akin to this
            if self._dimer_intcos:
                raise NotImplementedError("transformations with dimer gradients")
            # if self._dimer_intcos:
            #    # xyz = self.geom
            #    for i, DI in enumerate(self._dimer_intcos):
            #        # print('Aidx:' + str(DI.A_idx) )
            #        A1stAtom = self.frag_1st_atom(DI.A_idx)
            #        B1stAtom = self.frag_1st_atom(DI.B_idx)
            #        Axyz = self.frag_geom(DI.A_idx)
            #        Bxyz = self.frag_geom(DI.B_idx)
            #        DI.Bmat(Axyz, Bxyz, B[self.dimerfrag_intco_slice(i)],
            #            A1stAtom, 3 * B1stAtom)  # column offsets

        Hq = np.dot(Atranspose, np.dot(Hworking, Atranspose.T))
        return Hq

    def project_redundancies_and_constraints(self, fq, H):
        """Project redundancies and constraints out of forces and Hessian"""
        Nint = self.num_intcos
        # compute projection matrix = G G^-1
        G = self.Gmat()
        G_inv = symm_mat_inv(G, redundant=True)
        Pprime = np.dot(G, G_inv)
        # logger.debug("\tProjection matrix for redundancies.\n\n" + print_mat_string(Pprime))
        # Add constraints to projection matrix
        C = self.constraint_matrix(fq)  # returns None, if aren't any
        # fq is passed to Supplement matrix with ranged variables that are at their limit

        if C is not None:
            logger.debug("Adding constraints for projection.\n" + print_mat_string(C))
            CPC = np.zeros((Nint, Nint))
            CPC[:, :] = np.dot(C, np.dot(Pprime, C))
            CPCInv = symm_mat_inv(CPC, redundant=True)
            P = np.zeros((Nint, Nint))
            P[:, :] = Pprime - np.dot(Pprime, np.dot(C, np.dot(CPCInv, np.dot(C, Pprime))))
        else:
            P = Pprime

        # Project redundancies out of forces.
        # fq~ = P fq
        fq[:] = np.dot(P, fq.T)

        # if op.Params.print_lvl >= 3:
        logger.debug(
            "\n\tInternal forces in au, after projection of redundancies"
            + " and constraints.\n"
            + print_array_string(fq)
        )
        # Project redundancies out of Hessian matrix.
        # Peng, Ayala, Schlegel, JCC 1996 give H -> PHP + 1000(1-P)
        # The second term appears unnecessary and sometimes messes up Hessian updating.
        tempMat = np.dot(H, P)
        H[:, :] = np.dot(P, tempMat)
        # for i in range(dim)
        #    H[i,i] += 1000 * (1.0 - P[i,i])
        # for i in range(dim)
        #    for j in range(i):
        #        H[j,i] = H[i,j] = H[i,j] + 1000 * (1.0 - P[i,j])
        logger.debug("Projected (PHP) Hessian matrix\n" + print_mat_string(H))

    def apply_external_forces(self, fq, H, stepNumber):
        report = "Adding external forces\n"

        for iF, F in enumerate(self.fragments):
            for i, intco in enumerate(F.intcos):
                if intco.has_ext_force:
                    val = intco.q_show(self.geom)
                    ext_force = intco.ext_force_val(self.geom)

                    location = self.frag_1st_intco(iF) + i
                    fq[location] += ext_force
                    report += "Frag {:d}, Coord {:d}, Value {:10.5f}, Force {:12.6f}\n".format(
                        iF + 1, i + 1, val, ext_force
                    )
                    # modify Hessian later ?
                    # H[location][location] = k
                    # Delete coupling between this coordinate and others.
                    # logger.info("\t\tRemoving off-diagonal coupling between coordinate"
                    #            + "%d and others." % (location + 1))
                    # for j in range(len(H)):  # gives first dimension length
                    #    if j != location:
                    #        H[j][location] = H[location][j] = 0.0

        if "Frag" in report:
            logger.info(report)

    def hessian_to_cartesians(self, Hint, g_q=None):
        logger.info("Converting Hessian from internals to cartesians.\n")

        B = self.Bmat()
        # Hxy =  B^t Hij B
        Hxy = np.dot(B.T, np.dot(Hint, B))

        if g_q is None:  # Hxy =  B^t Hij B
            s = "Neglecting force/B-matrix derivative term, result is only "
            s += "strictly correct at stationary points.\n"
            logger.info(s)
        else:  # Hxy += dE/dq_I d2(q_I)/dxdy
            logger.info("Including force/B-matrix derivative term.\n")

            for iF, F in enumerate(self._fragments):
                dq2dx2 = np.zeros((3 * F.natom, 3 * F.natom))
                geom = F.geom
                cart_offset = 3 * self.frag_1st_atom(iF)
                intco_offset = self.frag_1st_intco(iF)

                for iIntco, Intco in enumerate(F.intcos):
                    dq2dx2[:] = 0
                    Intco.Dq2Dx2(geom, dq2dx2)  # d^2(q_I)/ dx_i dx_j

                    # Loop over Cartesian pairs in fragment
                    for a in range(3 * F.natom):
                        for b in range(3 * F.natom):
                            Hxy[cart_offset + a, cart_offset + b] += g_q[intco_offset + iIntco] * dq2dx2[a, b]

            # TODO: dimer coordinates
            if self._dimer_intcos:
                raise NotImplementedError("transformations with dimer gradients")

        return Hxy

    def gradient_to_cartesians(self, g_q):
        """converts the gradient from internal into Cartesian coordinates

        Parameters
        ----------
        g_q : ndarray
            internal coordinate gradient
        Returns
        -------
        g_x : ndarray
            Cartesian coordinate gradient
        """
        logger.debug("Converting gradient from internals to Cartesians.\n")
        B = self.Bmat()
        g_x = np.dot(B.T, g_q)
        return g_x

    def test_Bmat(self):
        """ Test the analytic B matrix (dq/dx) via finite differences.
        The 5-point formula should be good to DISP_SIZE^4 - a few
        unfortunates will be slightly worse.

        Returns
        -------
        passes : boolean
            Returns True or False, doesn't raise exceptions
        """
        Natom = self.natom
        Nintco = self.num_intcos
        DISP_SIZE = 0.01
        MAX_ERROR = 50 * DISP_SIZE * DISP_SIZE * DISP_SIZE * DISP_SIZE

        logger.info("\tTesting B-matrix numerically...")

        B_analytic = self.Bmat()

        if op.Params.print_lvl >= 3:
            logger.debug("Analytic B matrix in au")
            logger.debug(print_mat_string(B_analytic))

        B_fd = np.zeros((Nintco, 3 * Natom))

        self.update_dihedral_orientations()
        self.fix_bend_axes()

        geom_orig = self.geom  # to restore below
        coord = self.geom  # returns a copy

        for atom in range(Natom):
            for xyz in range(3):
                coord[atom, xyz] -= DISP_SIZE
                self.geom = coord
                q_m = np.array(self.q())

                coord[atom, xyz] -= DISP_SIZE
                self.geom = coord
                q_m2 = np.array(self.q())

                coord[atom, xyz] += 3 * DISP_SIZE
                self.geom = coord
                q_p = np.array(self.q())

                coord[atom, xyz] += DISP_SIZE
                self.geom = coord
                q_p2 = np.array(self.q())

                coord[atom, xyz] -= 2 * DISP_SIZE  # restore to original
                B_fd[:, 3 * atom + xyz] = (q_m2 - 8 * q_m + 8 * q_p - q_p2) / (12.0 * DISP_SIZE)

        if op.Params.print_lvl >= 3:
            logger.debug("Numerical B matrix in au, DISP_SIZE = %lf\n" % DISP_SIZE + print_mat_string(B_fd))

        self.geom = geom_orig  # restore original
        self.unfix_bend_axes()

        # max_error = -1.0
        # max_error_intco = -1
        # for i in range(Nintco):
        #    for j in range(3 * Natom):
        #        if np.fabs(B_analytic[i, j] - B_fd[i, j]) > max_error:
        #            max_error = np.fabs(B_analytic[i][j] - B_fd[i][j])
        #            max_error_intco = i
        B_delta = np.fabs(B_analytic - B_fd)
        max_index = np.unravel_index(np.argmax(B_delta), B_delta.shape)
        max_error_intco = max_index[0]
        max_error = B_delta[max_index]

        logger.info("\t\tMaximum difference is %.1e for internal coordinate %d." % (max_error, max_error_intco + 1))
        # logger.info("\t\tThis coordinate is %s" % str(intcos[max_error_intco]))

        if max_error > MAX_ERROR:
            logger.warning(
                "\tB-matrix could be in error. However, numerical tests may fail for\n"
                + "\ttorsions at 180 degrees, and slightly for linear bond angles."
                + "This is OK.\n"
            )
            return False
        else:
            logger.info("\t...Passed.")
            return True

    # Test the analytic derivative B matrix (d2q/dx2) via finite differences
    # The 5-point formula should be good to DISP_SIZE^4 -
    #  a few unfortunates will be slightly worse
    def test_derivative_Bmat(self):
        """ Test the analytic derivative B matrix (d2q/dx2) via finite
        differences.  The 5-point formula should be good to DISP_SIZE^4 - a few
        unfortunates will be slightly worse.

        Returns
        -------
        passes : boolean
            Returns True or False, doesn't raise exceptions
        """
        from . import intcosMisc
        DISP_SIZE = 0.01
        MAX_ERROR = 10 * DISP_SIZE * DISP_SIZE * DISP_SIZE * DISP_SIZE

        geom_orig = self.geom  # to restore below

        logger.info("\tTesting Derivative B-matrix numerically.")
        if self._dimer_intcos:
            logger.info("\tDerivative B-matrix for interfragment modes not yet implemented.")

        warn = False
        for iF, F in enumerate(self._fragments):
            logger.info("\t\tTesting fragment %d." % (iF + 1))

            Natom = F.natom
            Nintco = F.num_intcos
            coord = F.geom  # not a copy
            dq2dx2_fd = np.zeros((3 * Natom, 3 * Natom))
            dq2dx2_analytic = np.zeros((3 * Natom, 3 * Natom))

            for i, I in enumerate(F._intcos):
                logger.info("\t\tTesting internal coordinate %d :" % (i + 1))

                dq2dx2_analytic.fill(0)
                I.Dq2Dx2(coord, dq2dx2_analytic)

                if op.Params.print_lvl >= 3:
                    logger.info("Analytic B' (Dq2Dx2) matrix in au\n" + print_mat_string(dq2dx2_analytic))

                # compute B' matrix from B matrices
                for atom_a in range(Natom):
                    for xyz_a in range(3):

                        coord[atom_a, xyz_a] += DISP_SIZE
                        B_p = intcosMisc.Bmat(F.intcos, coord)

                        coord[atom_a, xyz_a] += DISP_SIZE
                        B_p2 =intcosMisc.Bmat(F.intcos, coord)

                        coord[atom_a, xyz_a] -= 3.0 * DISP_SIZE
                        B_m = intcosMisc.Bmat(F.intcos, coord)

                        coord[atom_a, xyz_a] -= DISP_SIZE
                        B_m2 = intcosMisc.Bmat(F.intcos, coord)

                        coord[atom_a, xyz_a] += 2 * DISP_SIZE  # restore coord to orig

                        for atom_b in range(Natom):
                            for xyz_b in range(3):
                                dq2dx2_fd[3 * atom_a + xyz_a, 3 * atom_b + xyz_b] = (
                                    B_m2[i, 3 * atom_b + xyz_b]
                                    - 8 * B_m[i, 3 * atom_b + xyz_b]
                                    + 8 * B_p[i, 3 * atom_b + xyz_b]
                                    - B_p2[i][3 * atom_b + xyz_b]
                                ) / (12.0 * DISP_SIZE)

                if op.Params.print_lvl >= 3:
                    logger.info(
                        "\nNumerical B' (Dq2Dx2) matrix in au, DISP_SIZE = %f\n" % DISP_SIZE
                        + print_mat_string(dq2dx2_fd)
                    )

                max_error = -1.0
                max_error_xyz = (-1, -1)
                for I in range(3 * Natom):
                    for J in range(3 * Natom):
                        if np.fabs(dq2dx2_analytic[I, J] - dq2dx2_fd[I, J]) > max_error:
                            max_error = np.fabs(dq2dx2_analytic[I][J] - dq2dx2_fd[I][J])
                            max_error_xyz = (I, J)

                logger.info(
                    "\t\tMax. difference is %.1e; 2nd derivative wrt %d and %d."
                    % (max_error, max_error_xyz[0], max_error_xyz[1])
                )

                if max_error > MAX_ERROR:
                    warn = True

        self.geom = geom_orig  # restore original
        self.unfix_bend_axes()

        if warn:
            logger.warning(
                """
            \tSome values did not agree.  However, numerical tests may fail for
            \ttorsions at 180 degrees and linear bond angles. This is OK
            \tIf discontinuities are interfering with a geometry optimization
            \ttry restarting your optimization at an updated geometry, and/or
            \tremove angular coordinates that are fixed by symmetry."""
            )
            return False
        else:
            logger.info("\t...Passed.")
            return True
