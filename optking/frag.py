import logging

import numpy as np
import qcelemental as qcel

from . import stre, bend, tors, oofp, cart
from . import addIntcos
from .printTools import print_array_string, print_mat_string
from itertools import combinations
from .v3d import are_collinear


class Frag:

    def __init__(self, Z, geom, masses, charge=0, multiplicity=1,
                 intcos=None, frozen=False):
        """ Group of bonded atoms
        Parameters
        ----------
        Z : list[int]
            atomic numbers
        geom : np.ndarray
            (nat, 3) cartesian geometry
        charge : int
            charge of fragment
        multiplicity: int
            multiplicit of fragment
        masses : list[float]
            atomic masses
        intcos : list(Simple), optional
            internal coordinates (stretches, bends, etch...)
        """

        self._Z = Z
        self._geom = geom
        self._masses = masses
        self._charge = charge
        self._multiplicity = multiplicity
        self._frozen = frozen

        self._intcos = []
        if intcos:
            self._intcos = intcos

    def __str__(self):
        # s = "\n\tZ (Atomic Numbers)\n\t"
        # print('num of self._intcos: %d' % len(self._intcos))
        s = print_array_string(self._Z, title="Z (Atomic Numbers)")
        # s += "\tGeom\n"
        s += print_mat_string(self._geom, title="Geom")
        # s += "\tMasses\n\t"
        s += print_array_string(self._masses, title="Masses")
        s += "\t - Coordinate -           - BOHR/RAD -       - ANG/DEG -\n"
        for x in self._intcos:
            s += ("\t%-18s=%17.6f%19.6f\n" % (x, x.q(self._geom), x.q_show(self._geom)))
        s += "\tCharge: {:d}     Multiplicity: {:d}     Frozen: {:s}".format(
            self._charge, self._multiplicity, str(self._frozen))
        s += "\n"
        return s


    def to_dict(self):
        d = {}
        d['Z'] = self._Z.copy()
        d['geom'] = self._geom.copy()
        d['masses'] = self._masses.copy()
        d['charge'] = self._charge
        d['multiplicty'] = self._multiplicity
        d['frozen'] = self._frozen
        d['intcos'] = [i.to_dict() for i in self._intcos]
        return d


    @classmethod
    def from_dict(cls, D):
        if 'Z' not in D or 'geom' not in D or 'masses' not in D:
            raise OptError('Missing required Z/geom/masses in dict input')
        Z =  D['Z']
        geom = D['geom']
        masses =  D['masses']
        charge =  D.get('charge', 0)
        multiplicity =  D.get('multiplicity', 1)
        frozen =  D.get('frozen', False)
        if 'intcos' in D: # class constructor (cls), e.g., stre.Stre
            intcos = []
            for i in D['intcos']:
                clc = str.lower(i['type']) + '.' + i['type']
                intcos.append( eval(clc).from_dict(i) )
        else:
            intcos = None
        return cls(Z, geom, masses, charge, multiplicity, intcos, frozen)


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
    def charge(self):
        return self._charge

    @property
    def multiplicity(self):
        return self._multiplicity

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
        self.add_h_bonds()

    def add_cartesian_intcos(self):
        addIntcos.add_cartesian_intcos(self._intcos, self._geom)

#    def print_geom(self):
#        for i in range(self._geom.shape[0]):
#            print_opt("\t%5s%15.10f%15.10f%15.10f\n" % \
#               (qcel.periodictable.to_E(self._Z[i]), self._geom[i,0], self._geom[i,1],
#                  self._geom[i,2]))
#        print_opt("\n")

    def add_h_bonds(self):
        """ Prepend h_bonds because that's where optking 2 places them """
        h_bonds = addIntcos.add_h_bonds(self.geom, self.Z, self.natom)
        for h_bond in h_bonds:
            if stre.Stre(h_bond.A, h_bond.B) in self._intcos:
                self._intcos.pop(self._intcos.index(stre.Stre(h_bond.A, h_bond.B)))
        self._intcos = h_bonds + self._intcos  # prepend internal coordinates

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
            intco.freeze()
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

    def is_atom(self):
        if self.natom == 1:
           return True
        else:
           return False

    def is_linear(self):
        if self.natom in [1,2]:  # lets tentatively call an atom linear here
            return True
        else:
            xyz = self.geom
            for (i,j,k) in combinations(range(self.natom),r=3):
                if not are_collinear(xyz[i], xyz[j], xyz[k]):
                    return False
            return True

