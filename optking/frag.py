# import numpy as np
import logging

from . import addIntcos
from . import atomData
from . import physconst
from .printTools import printArrayString, printMatString

# Geometry is 2D object (atom,xyz)


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
        s = "\n\tZ (Atomic Numbers)\n\t"
        s += printArrayString(self._Z)
        s += "\tGeom\n\t"
        s += printMatString(self._geom)
        s += "\tMasses\n\t"
        s += printArrayString(self._masses)
        s += "\t - Coordinate -           - BOHR/RAD -       - ANG/DEG -\n"
        for x in self._intcos:
            s += ("\t%-18s=%17.6f%19.6f\n" % (x, x.q(self._geom), x.qShow(self._geom)))
        s += "\n"
        return s

#    @classmethod
#    def fromPsi4Molecule(cls, mol):
#        mol.update_geometry()
#        geom = np.array(mol.geometry())
#        Natom = mol.natom()
#
#        #Z = np.zeros( Natom, int)
#        Z = []
#        for i in range(Natom):
#            Z.append(int(mol.Z(i)))
#
#        masses = np.zeros(Natom, float)
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
#            Z.append(atomDatas.symbol_to_Z(jmol['symbols'][i]))
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

    def addIntcosFromConnectivity(self, connectivity):
        addIntcos.addIntcosFromConnectivity(connectivity, self._intcos, self._geom)

    def addCartesianIntcos(self):
        addIntcos.addCartesianIntcos(self._intcos, self._geom)

#    def printGeom(self):
#        for i in range(self._geom.shape[0]):
#            print_opt("\t%5s%15.10f%15.10f%15.10f\n" % \
#            (atomData.Z_to_symbol[self._Z[i]], self._geom[i,0], self._geom[i,1], self._geom[i,2]))
#        print_opt("\n")

    def showGeom(self):
        geometry = ''
        Ang = self._geom * physconst.bohr2angstroms
        for i in range(self._geom.shape[0]):
            geometry += ("\t%5s%15.10f%15.10f%15.10f\n"
                         % (atomData.Z_to_symbol[self._Z[i]], Ang[i, 0], Ang[i, 1], Ang[i, 2]))
        geometry += ("\n")
        return geometry

    def get_atom_list(self):
        frag_atom_list = []
        for i in range(self._geom.shape[0]):
            frag_atom_list.append(atomData.Z_to_symbol[self._Z[i]])
        return frag_atom_list
