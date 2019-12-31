import json
import logging

import numpy as np

import qcelemental as qcel
from qcelemental.models import Molecule
from qcelemental.util.serialization import json_dumps

from . import frag
from .exceptions import AlgError, OptError
from . import v3d
from .addIntcos import connectivityFromDistances, addCartesianIntcos


class Molsys(object):
    """ The molecular system consisting of a collection of fragments

    Parameters
    ----------
    fragments : list(Frag)
    fb_fragments : list
        NYI fixed body fragments
    intcos : list(Simple), optional

    """
    def __init__(self, fragments, fb_fragments=None, intcos=None, multiplicity=1):
        # ordinary fragments with internal structure
        self.logger = logging.getLogger(__name__)
        self._fragments = []
        if fragments:
            self._fragments = fragments
        # fixed body fragments defined by Euler/rotation angles
        self._fb_fragments = []
        if fb_fragments:
            self._fb_fragments = fb_fragments
        self._multiplicity = multiplicity

    def __str__(self):
        s = ''
        for iF, F in enumerate(self._fragments):
            s += "\n\tFragment %d\n" % (iF + 1)
            s += F.__str__()
        for iB, B in enumerate(self._fb_fragments):
            s += "\tFixed body Fragment %d\n" % (iB + 1)
            s += B.__str__()
        return s

    @classmethod
    def from_psi4_molecule(cls, mol):
        """ Creates a optking molecular system from psi4 mol. Note that not all information is preserved.

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
            logger.critical("from_psi4_molecule cannot handle a non psi4 molecule, why are you calling this method?")
            raise OptError("Cannot make molecular system from this molecule")

        NF = mol.nfragments()
        logger.info("\t%d fragments in PSI4 molecule object." % NF)
        frags = []

        for iF in range(NF):
            fragMol = mol.extract_subsets(iF + 1)

            fragNatom = fragMol.natom()
            logger.info("\tCreating fragment %d with %d atoms" % (iF + 1, fragNatom))

            fragGeom = np.zeros((fragNatom, 3), float)
            fragGeom[:] = fragMol.geometry()

            fragZ = []
            for i in range(fragNatom):
                fragZ.append(int(fragMol.Z(i)))

            fragMasses = []
            for i in range(fragNatom):
                fragMasses.append(fragMol.mass(i))

            frags.append(frag.Frag(fragZ, fragGeom, fragMasses))

        m = mol.multiplicity()
        return cls(frags, multiplicity=m)

    @classmethod
    def from_JSON_molecule(cls, qc_molecule):
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
    def Natom(self):
        return sum(F.Natom for F in self._fragments)

    @property
    def multiplicity(self):
        return self._multiplicity

    @property
    def Nfragments(self):
        return len(self._fragments) + len(self._fb_fragments)

    # Return overall index of first atom in fragment, beginning 0,1,...
    def frag_1st_atom(self, iF):
        if iF >= len(self._fragments):
            return ValueError()
        start = 0
        for i in range(0, iF):
            start += self._fragments[i].Natom
        return start

    def frag_atom_range(self, iF):
        start = self.frag_1st_atom(iF)
        return range(start, start + self._fragments[iF].Natom)

    # accepts absolute atom index, returns fragment index
    def atom2frag_index(self, atom_index):
        for iF, F in enumerate(self._fragments):
            if atom_index in self.frag_atom_range(iF):
                return iF
        raise OptError("atom2frag_index: atom_index impossibly large")

    # Given a list of atoms, return all the fragments to which they belong
    def atomList2uniqueFragList(self, atomList):
        fragList = []
        for a in atomList:
            f = self.atom2frag_index(a)
            if f not in fragList:
                fragList.append(f)
        return fragList

    @property
    def geom(self):
        """cartesian geometry [a0]"""
        geom = np.zeros((self.Natom, 3), float)
        for iF, F in enumerate(self._fragments):
            row = self.frag_1st_atom(iF)
            geom[row:(row + F.Natom), :] = F.geom
        return geom

    @geom.setter
    def geom(self, newgeom):
        """ setter for geometry"""
        for iF, F in enumerate(self._fragments):
            row = self.frag_1st_atom(iF)
            F.geom[:] = newgeom[row:(row + F.Natom), :]

    @property
    def masses(self):
        m = np.zeros(self.Natom, float)
        for iF, F in enumerate(self._fragments):
            start = self.frag_1st_atom(iF)
            m[start:(start + F.Natom)] = F.masses
        return m

    @property
    def Z(self):
        z = [0 for i in range(self.Natom)]
        for iF, F in enumerate(self._fragments):
            first = self.frag_1st_atom(iF)
            z[first:(first + F.Natom)] = F.Z
        return z

    @property
    def intcos(self):
        _intcos = []
        for F in self._fragments:
            _intcos += F.intcos
        return _intcos

    def frag_1st_intco(self, iF):
        if iF >= len(self._fragments):
            return ValueError()
        start = 0
        for i in range(0, iF):
            start += len(self._fragments[i]._intcos)
        return start

    def printIntcos(self):
        for iF, F in enumerate(self._fragments):
            self.logger.info("Fragment %d\n" % (iF + 1))
            F.printIntcos()
        return

    def addIntcosFromConnectivity(self, C=None):
        for F in self._fragments:
            if C is None:
                C = F.connectivityFromDistances()
            F.addIntcosFromConnectivity(C)

    def addCartesianIntcos(self):
        for F in self._fragments:
            addCartesianIntcos(F._intcos, F._geom)

    def printGeom(self):
        """Returns a string of the geometry for logging in [a0]"""
        for iF, F in enumerate(self._fragments):
            self.logger.info("\tFragment %d\n" % (iF + 1))
            F.printGeom()

    def showGeom(self):
        """Return a string of the geometry in [A]"""
        molsys_geometry = ''
        for iF, F in enumerate(self._fragments):
            molsys_geometry += ("\tFragment %d\n" % (iF + 1))
            molsys_geometry += F.showGeom()
        return molsys_geometry

    @property
    def atom_symbols(self):
        symbol_list = []
        for F in self._fragments:
            frag_symbol_list = F.get_atom_symbol_list()
            for j in frag_symbol_list:
                symbol_list.append(j)
        return symbol_list


    def molsys_to_qc_molecule(self) -> qcel.models.Molecule:
            """
                Creates a qcschema molecule. version 1
    
            """
            geom = [i for i in self.geom.flat]
            qc_mol = {"symbols": self.atom_symbols, "geometry": geom, "masses": self.masses.tolist(),
                           "molecular_multiplicity": self.multiplicity, 
                           #"molecular_charge": self.charge, Should be unnessecary
                           "fix_com": True, "fix_orientation": True}
            qc_mol = Molecule(**qc_mol)
            qc_mol = json.loads(json_dumps(qc_mol))
            return qc_mol
    

    def consolidateFragments(self):
        if self.Nfragments == 1:
            return
        self.logger.info("\tConsolidating multiple fragments into one for optimization.")
        consolidatedFrag = frag.Frag(self.Z, self.geom, self.masses)
        del self._fragments[:]
        self._fragments.append(consolidatedFrag)

    def splitFragmentsByConnectivity(self):
        """ Split any fragment not connected by bond connectivity."""
        tempZ = np.copy(self.Z)
        tempGeom = np.copy(self.geom)
        tempMasses = np.copy(self.masses)

        newFragments = []
        for F in self._fragments:
            C = connectivityFromDistances(F.geom, F.Z)
            atomsToAllocate = list(reversed(range(F.Natom)))
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
                subZ = np.zeros(subNatom, float)
                subGeom = np.zeros((subNatom, 3), float)
                subMasses = np.zeros(subNatom, float)
                for i, I in enumerate(frag_atoms):
                    subZ[i] = tempZ[I]
                    subGeom[i, 0:3] = tempGeom[I, 0:3]
                    subMasses[i] = tempMasses[I]
                newFragments.append(frag.Frag(subZ, subGeom, subMasses))

        del self._fragments[:]
        self._fragments = newFragments

    # Supplements a connectivity matrix to connect all fragments.  Assumes the
    # definition of the fragments has ALREADY been determined before function called.
    def augmentConnectivityToSingleFragment(self, C):
        self.logger.info('\tAugmenting connectivity matrix to join fragments.')
        fragAtoms = []
        geom = self.geom
        for iF, F in enumerate(self._fragments):
            fragAtoms.append(
                range(self.frag_1st_atom(iF),
                      self.frag_1st_atom(iF) + F.Natom))

        # Which fragments are connected?
        nF = self.Nfragments
        self.logger.critical(str(self.Nfragments))
        if self.Nfragments == 1:
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
        self._fb_fragments.clear()

