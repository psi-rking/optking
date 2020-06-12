import logging

import numpy as np
import qcelemental as qcel

from . import bend, tors, oofp
from . import addIntcos
from .printTools import print_array_string, print_mat_string

class Frag:
    """ Group of bonded atoms

    Parameters
    ----------
    Z : list
        atomic numbers
    geom : ndarray
        (nat, 3) cartesian geometry
    masses : list
        atomic masses
    intcos : list(Simple), optional
        internal coordinates (stretches, bends, etch...)

    """
    def __init__(self, Z, geom, masses, intcos=None):
        self._Z = Z
        self._geom = geom
        self._masses = masses
        self._frozen = False

        self._intcos = []
        if intcos:
            self._intcos = intcos

    def __str__(self):
        #s = "\n\tZ (Atomic Numbers)\n\t"
        #print('num of self._intcos: %d' % len(self._intcos))
        s = print_array_string(self._Z, title="Z (Atomic Numbers)")
        #s += "\tGeom\n"
        s += print_mat_string(self._geom, title="Geom")
        #s += "\tMasses\n\t"
        s += print_array_string(self._masses, title="Masses")
        s += "\t - Coordinate -           - BOHR/RAD -       - ANG/DEG -\n"
        for x in self._intcos:
            s += ("\t%-18s=%17.6f%19.6f\n" % (x, x.q(self._geom), x.q_show(self._geom)))
        s += "\n"
        return s

#    @classmethod
#    def fromPsi4Molecule(cls, mol):
#        mol.update_geometry()
#        geom = np.array(mol.geometry(),float)
#        natom = mol.natom()
#
#        #Z = np.zeros( natom, int)
#        Z = []
#        for i in range(natom):
#            Z.append(int(mol.Z(i)))
#
#        masses = np.zeros(natom)
#        for i in range(natom):
#            masses[i] = mol.mass(i)
#
#        return cls(Z, geom, masses)
#
#
#    #todo
#    @classmethod
#    def from_json_molecule(cls, pmol):
#        #taking in psi4.core.molecule and converting to schema
#        jmol = pmol.to_schema()
#        print(jmol)
#        natom = len(jmol['symbols'])
#        geom = np.asarray(molecule['geometry']).reshape(-1,3) #?need to reshape in some way todo
#        print(geom)
#        Z = []
#        for i in range(natom):
#            Z.append(qcel.periodictable.to_Z(jmol['symbols'][i]))
#
#        print(Z)
#
#        masses = jmol['masses']
#        print(masses)
#
#        return cls(Z, geom, masses)

    @property
    def natom(self):
        return len(self._Z)

    @property
    def Z(self):
        return self._Z

    @property
    def geom(self):
        return self._geom

    @property
    def masses(self):
        return self._masses

    @property
    def intcos(self):
        return self._intcos

    @property
    def frozen(self):
        return self._frozen

    @property
    def num_intcos(self):
        return len(self._intcos)

    def q(self):
        return [intco.q(self.geom) for intco in self._intcos]

    def q_array(self):
        return np.asarray( self.q() )

    def q_show(self):
        return [intco.q_show(self.geom) for intco in self._intcos]

    def q_show_array(self):
        return np.asarray(self.q_show())

    def print_intcos(self):
        logger = logging.getLogger(__name__)
        intcos_report = "\tInternal Coordinate Values\n"
        intcos_report += "\n\t - Coordinate -           - BOHR/RAD -       - ANG/DEG -\n"
        for coord in self._intcos:
            intcos_report += ('\t%-18s=%17.6f%19.6f\n'
                              % (coord, coord.q(self._geom), coord.q_show(self._geom)))
        intcos_report += "\n"
        logger.info(intcos_report)
        return

    def connectivity_from_distances(self):
        return addIntcos.connectivity_from_distances(self._geom, self._Z)

    def add_intcos_from_connectivity(self, connectivity=None):
        if connectivity is None:
            connectivity = self.connectivity_from_distances()
        addIntcos.add_intcos_from_connectivity(connectivity, self._intcos, self._geom)

    def add_cartesian_intcos(self):
        addIntcos.add_cartesian_intcos(self._intcos, self._geom)

#    def print_geom(self):
#        for i in range(self._geom.shape[0]):
#            print_opt("\t%5s%15.10f%15.10f%15.10f\n" % \
#            (qcel.periodictable.to_E(self._Z[i]), self._geom[i,0], self._geom[i,1], self._geom[i,2]))
#        print_opt("\n")

    def show_geom(self):
        geometry = ''
        Ang = self._geom * qcel.constants.bohr2angstroms
        for i in range(self._geom.shape[0]):
            geometry += ("\t%5s%15.10f%15.10f%15.10f\n"
                         % (qcel.periodictable.to_E(self._Z[i]), Ang[i, 0], Ang[i, 1], Ang[i, 2]))
        geometry += "\n"
        return geometry

    def get_atom_symbol_list(self):
        frag_atom_symbol_list = []
        for i in range(self._geom.shape[0]):
            frag_atom_symbol_list.append(qcel.periodictable.to_E(self._Z[i]))
        return frag_atom_symbol_list

    def compute_b_mat(self):
        B = np.zeros((self.num_intcos, 3 * self.natom))
        for i, intco in enumerate(self._intcos):
            intco.DqDx(self.geom, B[i])
        return B

    def fix_bend_axes(self):
        for intco in self._intcos:
            if isinstance(intco, bend.Bend):
                intco.fix_bend_axes(self.geom)

    def unfix_bend_axes(self):
        for intco in self._intcos:
            if isinstance(intco, bend.Bend):
                intco.unfix_bend_axes()

    def freeze(self):
        for intco in self._intcos:
            intco.frozen = True
        self._frozen = True

    def update_dihedral_orientations(self):
        """ Update orientation of each dihedrals/tors coordinate
         This saves an indicator if dihedral is slightly less than pi,
         or slighly more than -pi.  Subsequently, computation of values
         can be greater than pi or less than -pi to enable computation
         of Delta(q) when q passed through pi.
        """
        for intco in self._intcos:
            if isinstance(intco, tors.Tors) or isinstance(intco, oofp.Oofp):
                intco.update_orientation(self.geom)


