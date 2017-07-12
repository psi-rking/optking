import numpy as np
import misc
import addIntcos
import physconst
from printTools import printArrayString, printMatString,print_opt
import atomData

# Geometry is 2D object (atom,xyz)

class FRAG():
    def __init__(self, Z, geom, masses, intcos=None):
        self._Z = Z
        self._geom = geom
        self._masses = masses

        self._intcos = []
        if intcos:
            self._intcos = intcos

    def __str__(self):
        s = "Z (Atomic Numbers)\n"
        s += printArrayString(self._Z)
        s += "Geom\n"
        s += printMatString(self._geom)
        s += "Masses\n"
        s += printArrayString(self._masses)

        s += "\t - Coordinate -           - BOHR/RAD -       - ANG/DEG -\n"
        for x in self._intcos:
            s += "\t%-18s=%17.6f%19.6f\n" % (x, x.q(self._geom), x.qShow(self._geom))
        s += "\n"
        return s

    @classmethod
    def fromPsi4Molecule(cls, mol):
        mol.update_geometry()
        geom = np.array( mol.geometry() )
        Natom = mol.natom()
         
        #Z = np.zeros( Natom, int)
        Z = []
        for i in range(Natom):
            Z.append( int(mol.Z(i)) )
         
        masses = np.zeros( Natom, float)
        for i in range(Natom):
            masses[i] = mol.mass(i)
         
        return cls(Z,geom,masses)

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
        print_opt("\tInternal Coordinate Values\n")
        print_opt("\t - Coordinate -           - BOHR/RAD -       - ANG/DEG -\n")
        for coord in self._intcos:
            print_opt('\t%-18s=%17.6f%19.6f\n' % (coord, coord.q(self._geom), coord.qShow(self._geom)))
        print_opt("\n")
        return

    def connectivityFromDistances(self):
        return addIntcos.connectivityFromDistances(F._geom, F._Z)

    def addIntcosFromConnectivity(self, connectivity):
        addIntcos.addIntcosFromConnectivity(connectivity, self._intcos, self._geom)

    def addCartesianIntcos(self):
        addIntcos.addCartesianIntcos(self._intcos, self._geom)

    def printGeom(self):
        for i in range(self._geom.shape[0]):
            print_opt("\t%5s%15.10f%15.10f%15.10f\n" % \
            (atomData.Z_to_symbol[self._Z[i]], self._geom[i,0], self._geom[i,1], self._geom[i,2]))
        print_opt("\n")

    def showGeom(self):
        Ang = self._geom * physconst.bohr2angstroms
        for i in range(self._geom.shape[0]):
            print_opt("\t%5s%15.10f%15.10f%15.10f\n" % \
            (atomData.Z_to_symbol[self._Z[i]], Ang[i,0], Ang[i,1], Ang[i,2]))
        print_opt("\n")


