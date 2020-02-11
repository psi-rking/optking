import logging

import numpy as np
import qcelemental as qcel

from . import bend, tors, oofp
from . import addIntcos
from .printTools import printArrayString, printMatString

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

        self._intcos = []
        if intcos:
            self._intcos = intcos

    def __str__(self):
        #s = "\n\tZ (Atomic Numbers)\n\t"
        #print('num of self._intcos: %d' % len(self._intcos))
        s = printArrayString(self._Z, title="Z (Atomic Numbers)")
        #s += "\tGeom\n"
        s += printMatString(self._geom, title="Geom")
        #s += "\tMasses\n\t"
        s += printArrayString(self._masses, title="Masses")
        s += "\t - Coordinate -           - BOHR/RAD -       - ANG/DEG -\n"
        for x in self._intcos:
            s += ("\t%-18s=%17.6f%19.6f\n" % (x, x.q(self._geom), x.qShow(self._geom)))
        s += "\n"
        return s

#    @classmethod
#    def fromPsi4Molecule(cls, mol):
#        mol.update_geometry()
#        geom = np.array(mol.geometry(),float)
#        Natom = mol.natom()
#
#        #Z = np.zeros( Natom, int)
#        Z = []
#        for i in range(Natom):
#            Z.append(int(mol.Z(i)))
#
#        masses = np.zeros(Natom)
#        for i in range(Natom):
#            masses[i] = mol.mass(i)
#
#        return cls(Z, geom, masses)
#
#
#    #todo
#    @classmethod
#    def from_JSON_molecule(cls, pmol):
#        #taking in psi4.core.molecule and converting to schema
#        jmol = pmol.to_schema()
#        print(jmol)
#        Natom = len(jmol['symbols'])
#        geom = np.asarray(molecule['geometry']).reshape(-1,3) #?need to reshape in some way todo
#        print(geom)
#        Z = []
#        for i in range(Natom):
#            Z.append(qcel.periodictable.to_Z(jmol['symbols'][i]))
#
#        print(Z)
#
#        masses = jmol['masses']
#        print(masses)
#
#        return cls(Z, geom, masses)

    @property
    def Natom(self):
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
    def Nintcos(self):
        return len(self._intcos)

    def q(self):
        return [intco.q(self.geom) for intco in self._intcos]

    def qArray(self):
        return np.asarray( self.q() )

    def qShow(self):
        return [intco.qShow(self.geom) for intco in self._intcos]

    def qShowArray(self):
        return np.asarray( self.qShow() )

    def printIntcos(self):
        logger = logging.getLogger(__name__)
        intcos_report = ("\tInternal Coordinate Values\n")
        intcos_report += ("\n\t - Coordinate -           - BOHR/RAD -       - ANG/DEG -\n")
        for coord in self._intcos:
            intcos_report += ('\t%-18s=%17.6f%19.6f\n'
                              % (coord, coord.q(self._geom), coord.qShow(self._geom)))
        intcos_report += ("\n")
        logger.info(intcos_report)
        return

    def connectivityFromDistances(self):
        return addIntcos.connectivityFromDistances(self._geom, self._Z)

    def addIntcosFromConnectivity(self, connectivity=None):
        if connectivity is None:
            connectivity = self.connectivityFromDistances()
        addIntcos.addIntcosFromConnectivity(connectivity, self._intcos, self._geom)

    def addCartesianIntcos(self):
        addIntcos.addCartesianIntcos(self._intcos, self._geom)

#    def printGeom(self):
#        for i in range(self._geom.shape[0]):
#            print_opt("\t%5s%15.10f%15.10f%15.10f\n" % \
#            (qcel.periodictable.to_E(self._Z[i]), self._geom[i,0], self._geom[i,1], self._geom[i,2]))
#        print_opt("\n")

    def showGeom(self):
        geometry = ''
        Ang = self._geom * qcel.constants.bohr2angstroms
        for i in range(self._geom.shape[0]):
            geometry += ("\t%5s%15.10f%15.10f%15.10f\n"
                         % (qcel.periodictable.to_E(self._Z[i]), Ang[i, 0], Ang[i, 1], Ang[i, 2]))
        geometry += ("\n")
        return geometry

    def get_atom_symbol_list(self):
        frag_atom_symbol_list = []
        for i in range(self._geom.shape[0]):
            frag_atom_symbol_list.append(qcel.periodictable.to_E(self._Z[i]))
        return frag_atom_symbol_list

    def Bmat(self):
        B = np.zeros( (self.Nintcos, 3*self.Natom) )
        for i, intco in enumerate(self._intcos):
            intco.DqDx(self.geom, B[i])
        return B

    def fixBendAxes(self):
        for intco in self._intcos:
            if isinstance(intco, bend.Bend):
                intco.fixBendAxes(self.geom)

    def unfixBendAxes(self):
        for intco in self._intcos:
            if isinstance(intco, bend.Bend):
                intco.unfixBendAxes()

    def updateDihedralOrientations(self):
        """ Update orientation of each dihedrals/tors coordinate
         This saves an indicator if dihedral is slightly less than pi,
         or slighly more than -pi.  Subsequently, computation of values
         can be greater than pi or less than -pi to enable computation
         of Delta(q) when q passed through pi.
        """
        for intco in self._intcos:
            if isinstance(intco, tors.Tors) or isinstance(intco, oofp.Oofp):
                intco.updateOrientation(self.geom)


