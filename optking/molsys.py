import json
import logging

import numpy as np
from itertools import permutations

import qcelemental as qcel
from qcelemental.models import Molecule
from qcelemental.util.serialization import json_dumps

from . import frag
from .exceptions import AlgError, OptError
from . import v3d
from .addIntcos import connectivity_from_distances, add_cartesian_intcos
from .linearAlgebra import symm_mat_inv
from .printTools import print_mat_string


class Molsys(object):
    """ The molecular system consisting of a collection of fragments

    Parameters
    ----------
    fragments : list(Frag)
    #fb_fragments : list
    #    NYI fixed body fragments
    intcos : list(Simple), optional

    """

    def __init__(self, fragments, multiplicity=1, dimer_intcos=None):
        # def __init__(self, fragments, fb_fragments=None, intcos=None, multiplicity=1):
        # ordinary fragments with internal structure
        self.logger = logging.getLogger(__name__)

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
        self._multiplicity = multiplicity

    def __str__(self):
        s = ''
        for iF, F in enumerate(self._fragments):
            s += "\n\tFragment %d\n" % (iF + 1)
            s += F.__str__()
        for di in self._dimer_intcos:
            s += di.__str__()
        # for iB, atom_b in enumerate(self._fb_fragments):
        #    s += "\tFixed body Fragment %d\n" % (iB + 1)
        #    s += atom_b.__str__()
        return s

    @classmethod
    def from_psi4_molecule(cls, mol):
        """ Creates a optking molecular system from psi4 mol. Note that not all information
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

        logger = logging.getLogger(__name__)
        logger.info("\tGenerating molecular system for optimization from PSI4.")

        if not isinstance(mol, psi4.core.Molecule):
            logger.critical("from_psi4_molecule cannot handle a non psi4 molecule")
            raise OptError("Cannot make molecular system from this molecule")

        NF = mol.nfragments()
        logger.info("\t%d fragments in PSI4 molecule object." % NF)
        frags = []

        for iF in range(NF):
            frag_mol = mol.extract_subsets(iF + 1)
            frag_natom = frag_mol.natom()
            logger.info("\tCreating fragment %d with %d atoms" % (iF + 1, frag_natom))

            frag_geom = frag_mol.geometry().np
            frag_z = [frag_mol.Z(i) for i in range(frag_natom)]
            frag_masses = [frag_mol.mass(i) for i in range(frag_natom)]

            frags.append(frag.Frag(frag_z, frag_geom, frag_masses))

        m = mol.multiplicity()
        return cls(frags, multiplicity=m)

    @classmethod
    def from_json_molecule(cls, qc_molecule):
        """ Creates optking molecular system from JSON input.

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
        logger = logging.getLogger(__name__)
        logger.info("\tGenerating molecular system for optimization from QC Schema.\n")

        geom = np.asarray(qc_molecule['geometry'])
        geom = geom.reshape(-1, 3)

        Z_list = [qcel.periodictable.to_Z(atom) for atom in qc_molecule['symbols']]

        masses_list = qc_molecule.get('masses')
        if masses_list is None:
            masses_list = [qcel.periodictable.to_mass(atom) for atom in qc_molecule['symbols']]

        frags = []
        if 'fragments' in qc_molecule:
            for fr in qc_molecule['fragments']:
                frags.append(frag.Frag(np.array(Z_list)[fr], geom[fr], np.array(masses_list)[fr]))
        else:
            frags.append(frag.Frag(Z_list, geom, masses_list))

        return cls(frags)

    @property
    def natom(self):
        return sum(F.natom for F in self._fragments)

    @property
    def multiplicity(self):
        return self._multiplicity

    @property
    def num_frags(self):
        return len(self._fragments)
        # return len(self._fragments) + len(self._fb_fragments)

    # not supposed to have property that has anything other than self
    @property
    def frag_natom(self, iF):
        return self._fragments[iF].natom

    @property
    def fragments(self):
        return self._fragments

    @property
    def dimer_intcos(self):
        return self._dimer_intcos

    @property
    def intcos(self):
        """ Collect intcos for all fragments. Add dimer coords to end.

        Returns
        -------

        """
        logger = logging.getLogger(__name__)
        logger.warning("""This method is currently implemented as a last resort used as a last
                       resort. Should be safe assuming no dimer coordinates, otherwise unknown.""")
        coords = [coord for f in self._fragments for coord in f.intcos]
        for d_coord in self.dimer_intcos:
            coords.append(d_coord)
        return coords

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
    def atom_2_frag_index(self, atom_index):
        for iF in range(self.num_frags):
            if atom_index in self.frag_atom_range(iF):
                return iF
        raise OptError("atom_2_frag_index: atom_index impossibly large")

    # Given a list of atoms, return all the fragments to which they belong
    def atom_list_2_unique_frag_list(self, atomList):
        fragList = []
        for a in atomList:
            f = self.atom_2_frag_index(a)
            if f not in fragList:
                fragList.append(f)
        return fragList

    @property
    def geom(self):
        """cartesian geometry [a0]"""
        geom = np.zeros((self.natom, 3))
        for iF, F in enumerate(self._fragments):
            row = self.frag_1st_atom(iF)
            geom[row:(row + F.natom), :] = F.geom
        return geom

    def frag_geom(self, iF):
        """cartesian geometry for fragment i"""
        return self._fragments[iF].geom
        # return copy instead?
        # using in displace_molsys

    @geom.setter
    def geom(self, newgeom):
        """ setter for geometry"""
        for iF, F in enumerate(self._fragments):
            row = self.frag_1st_atom(iF)
            F.geom[:] = newgeom[row:(row + F.natom), :]

    @property
    def masses(self):
        m = np.zeros(self.natom)
        for iF, F in enumerate(self._fragments):
            m[self.frag_atom_slice(iF)] = F.masses
        return m

    @property
    def z(self):
        z = [0] * self.natom
        for iF, F in enumerate(self._fragments):
            first = self.frag_1st_atom(iF)
            z[first:(first + F.natom)] = F.z
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
        N = 0
        for F in self._fragments:
            N += F.num_intcos
        for DI in self._dimer_intcos:
            N += DI.num_intcos
        return N

    @property
    def frozen_intco_list(self):
        """Determine vector with 1 for any frozen internal coordinate"""
        frozen = np.zeros(self.num_intcos)
        cnt = 0
        for F in self._fragments:
            for intco in F.intcos:
                if intco.frozen:
                    frozen[cnt] = 1
                cnt += 1
        for DI in self._dimer_intcos:
            for intco in DI.pseudo_frag.intcos:
                if intco.frozen:
                    frozen[cnt] = 1
                cnt += 1
        return frozen

    @property
    def constraint_matrix(self):
        """Returns constraint matrix with 1 on diagonal for frozen coordinates"""
        frozen = self.frozen_intco_list
        if np.any(frozen):
            return np.diagflat(frozen)
        else:
            return None

    @property
    def intcos_present(self):
        for F in self._fragments:
            if F.intcos:
                return True
        for DI in self._dimer_intcos:
            if DI.pseudo_frag.intcos:
                return True
        return False

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
        for iF, F in enumerate(self._fragments):
            self.logger.info("Fragment %d\n" % (iF + 1))
            F.print_intcos()
        return

    # If connectivity is provided, only intrafragment connections
    # are used.  Interfragment connections are ignored here.
    # def add_intcos_from_connectivity(self, connectivity_mat=None):
    #    for F in self._fragments:
    #        if connectivity_mat is None:
    #            connectivity_mat = F.connectivity_from_distances()
    #        F.add_intcos_from_connectivity(connectivity_mat)

    def add_cartesian_intcos(self):
        for F in self._fragments:
            add_cartesian_intcos(F.intcos, F.geom)

    def print_geom(self):
        """Returns a string of the geometry for logging in [a0]"""
        for iF, F in enumerate(self._fragments):
            self.logger.info("\tFragment %d\n" % (iF + 1))
            F.print_geom()

    def show_geom(self):
        """Return a string of the geometry in [atom_a]"""
        molsys_geometry = ''
        for iF, F in enumerate(self._fragments):
            molsys_geometry += ("\tFragment %d\n" % (iF + 1))
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
        for iF in range(self.num_frags):
            fl = [i for i in self.frag_atom_range(iF)]
            l.append(fl)
        # print("l",l)
        return l

    def molsys_to_qc_molecule(self) -> qcel.models.Molecule:
        """ Creates a qcschema molecule. version 1 """
        # print("molsys_to_qc_molecule: input molsys:")
        # print(self)

        geom = [i for i in self.geom.flat]
        qc_mol = {"symbols": self.atom_symbols,
                  "geometry": geom,
                  "masses": self.masses.tolist(),
                  "molecular_multiplicity": self.multiplicity,
                  "fragments": self.fragments_atom_list,
                  # "molecular_charge": self.charge, Should be unnecessary
                  "fix_com": True,
                  "fix_orientation": True}
        qc_mol = Molecule(**qc_mol)
        qc_mol = json.loads(json_dumps(qc_mol))
        return qc_mol

    def q(self):
        """Returns internal coordinate values in au as list.

        Returns
        -------
        list[float]
        """
        vals = []
        for F in self._fragments:
            vals += F.q()
        self.update_dimer_intco_reference_points()
        for DI in self._dimer_intcos:
            vals += DI.q()
        return vals

    def q_array(self):
        """Returns internal coordinate values in au as array.

        Returns
        -------
        list[float]
        """
        return np.asarray(self.q())

    def q_show(self):
        """returns internal coordinates values in Angstroms/degrees as list.

        Returns
        -------
        list[float]
        """
        vals = []
        for F in self._fragments:
            vals += F.q_show()

        for DI in self._dimer_intcos:
            vals += DI.q_show()
        return vals

    def q_show_array(self):
        """ Returns internal coordinates values in Angstroms/degrees.
        Returns
        -------
        np.ndarray
        """
        return np.asarray(self.q_show())

    def consolidate_frags(self):
        logger = logging.getLogger(__name__)
        if self.num_frags == 1:
            return
        self.logger.info("\tConsolidating multiple fragments into one for optimization.")
        logger.debug("consolidating fragments")
        Z = self._fragments[0].z
        g = self._fragments[0].geom
        logger.debug('frag0 geom in consolidate_frags:')
        logger.debug(self._fragments[0].geom)
        logger.debug('frag1 geom in consolidate_frags:')
        logger.debug(self._fragments[1].geom)
        m = self._fragments[0].masses
        for i in range(1, self.num_frags):
            Z = np.concatenate((Z, self._fragments[i].z))
            g = np.concatenate((g, self._fragments[i].geom))
            m = np.concatenate((m, self._fragments[i].masses))
        # self._fragments.append(consolidatedFrag)
        del self._fragments[:]
        consolidatedFrag = frag.Frag(Z, g, m)
        self._fragments.append(consolidatedFrag)
        logger.debug("consolidated fragment")
        logger.debug(self._fragments[0])

    def split_frags_by_connectivity(self):
        """ Split any fragment not connected by bond connectivity."""
        tempZ = np.copy(self.z)
        tempGeom = np.copy(self.geom)
        tempMasses = np.copy(self.masses)

        newFragments = []
        for F in self._fragments:
            C = connectivity_from_distances(F.geom, F.z)
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
                subZ = np.zeros(subNatom)
                subGeom = np.zeros((subNatom, 3))
                subMasses = np.zeros(subNatom)
                for i, I in enumerate(frag_atoms):
                    subZ[i] = tempZ[I]
                    subGeom[i, 0:3] = tempGeom[I, 0:3]
                    subMasses[i] = tempMasses[I]
                newFragments.append(frag.Frag(subZ, subGeom, subMasses))

        del self._fragments[:]
        self._fragments = newFragments

    def purge_interfrag_connectivity(self, C):
        for f1, f2 in permutations([i for i in range(self.num_frags)], 2):
            for a in self.frag_atom_range(f1):
                for b in self.frag_atom_range(f2):
                    C[a, b] = 0.0
        return

    def augment_connectivity_to_single_frag(self, C):
        """ Supplements a connectivity matrix to connect all fragments.  Assumes the
        definition of the fragments has ALREADY been determined before function called.

        Parameters
        ----------
        C

        Returns
        -------

        """
        self.logger.info('\tAugmenting connectivity matrix to join fragments.')
        fragAtoms = []
        geom = self.geom
        for iF, F in enumerate(self._fragments):
            fragAtoms.append(
                range(self.frag_1st_atom(iF),
                      self.frag_1st_atom(iF) + F.natom))

        # Which fragments are connected?
        nF = self.num_frags
        self.logger.critical(str(self.num_frags))
        if self.num_frags == 1:
            return

        frag_connectivity = np.zeros((nF, nF))
        for iF in range(nF):
            frag_connectivity[iF, iF] = 1

        Z = self.z

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

                    self.logger.info("\tConnecting fragments with atoms %d and %d"
                                     % (i + 1, j + 1))
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
                                self.logger.info("\tAlso, with atoms %d and %d\n"
                                                 % (i + 1, j + 1))
                                C[i][j] = C[j][i] = True

            # Test whether all frags are connected using current distance threshold
            if np.sum(frag_connectivity[0]) == nF:
                self.logger.info("\tAll fragments are connected in connectivity matrix.")
                all_connected = True
            else:
                scale_dist += 0.2
                self.logger.info(
                    "\tIncreasing scaling to %6.3f to connect fragments." % scale_dist)
        return

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

    # Returns mass-weighted atom_b matrix if masses are supplied.
    def wilson_b_mat(self, masses=None):
        logger = logging.getLogger(__name__)

        # Allocate memory for full system.
        Nint = self.num_intcos
        Ncart = 3 * self.natom
        B = np.zeros((Nint, Ncart))

        for iF, F in enumerate(self._fragments):
            fB = F.wilson_b_mat()
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
                DI.compute_B(Axyz, Bxyz, B[self.dimerfrag_intco_slice(i)],
                             A1stAtom, 3 * B1stAtom)  # column offsets

        if isinstance(masses, np.ndarray):
            sqrtm = np.array([np.repeat(np.sqrt(masses), 3)] * Nint, float)
            B[:] = np.divide(B, sqrtm)
        # print('end of addIntcos.Bmatrix:')
        # print(atom_b)
        logger.debug(print_mat_string(B, title="B matrix"))

        return B

    def q_show_forces(self, forces):
        """ Returns scaled forces as array. """
        c = []
        for F in self._fragments:
            c += [intco.f_show_factor for intco in F.intcos]
        for DI in self._dimer_intcos:
            c += [intco.f_show_factor for intco in DI.pseudo_frag.intcos]
        c = np.asarray(c)
        qaJ = c * forces
        return qaJ

    # Returns mass-weighted Gmatrix if masses are supplied.
    def wilson_g_mat(self, masses=None):
        """ Calculates BuB^T (calculates atom_b matrix)

        Parameters
        ----------
        masses : list[float]
            length natom
        """
        B = self.wilson_b_mat(masses)
        return np.dot(B, B.T)

    def q_forces(self, gradient_x, B=None):
        """Transform cartesian gradient to internals
        Parameters
        ----------
        gradient_x :
            (3nat, 1) cartesian gradient
        Returns
        -------
        ndarray
            forces in internal coordinates (-1 * gradient)
        Notes
        -----
        fq = (BuB^T)^(-1)*B*f_x
    
        """
        if not self.intcos_present or self.natom == 0:
            return np.zeros(0)

        if B is None:
            B = self.wilson_b_mat()

        fx = np.multiply(-1.0, gradient_x)  # gradient -> forces
        G = np.dot(B, B.T)
        Ginv = np.linalg.pinv(G)
        fq = np.dot(np.dot(Ginv, B), fx)
        return fq
