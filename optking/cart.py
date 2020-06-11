import qcelemental as qcel

from .exceptions import AlgError, OptError
from .simple import Simple


class Cart(Simple):
    def __init__(self, a, xyz_in, frozen=False, fixedEqVal=None):

        self.xyz = xyz_in  # uses setter below
        atoms = (a, )
        Simple.__init__(self, atoms, frozen, fixedEqVal)

    def __str__(self):
        if self.frozen: s = '*'
        else: s = ' '

        if self._xyz == 0:
             s += 'X'
        elif self._xyz == 1:
             s += 'Y'
        elif self._xyz == 2:
             s += 'Z'

        s += "(%d)" % (self.A + 1)
        if self.fixedEqVal:
            s += "[%.4f]" % (self.fixedEqVal * self.qShowFactor)
        return s

    def __eq__(self, other):
        if self.atoms != other.atoms:
             return False
        elif not isinstance(other, Cart):
             return False
        elif self.xyz != other.xyz:
             return False
        else:
             return True

    @property
    def xyz(self):
        return self._xyz

    @xyz.setter
    def xyz(self, setval):
        if setval in [0, 'x', 'X']:
            self._xyz = 0
        elif setval in [1, 'y', 'Y']:
            self._xyz = 1
        elif setval in [2, 'z', 'Z']:
            self._xyz = 2
        else:
            raise OptError("Cartesian coordinate must be set to 0-2 or X-Z")

    def q(self, geom):
        return geom[self.A, self._xyz]

    @property
    def qShowFactor(self):
        return qcel.constants.bohr2angstroms

    def qShow(self, geom):
        return self.qShowFactor * self.q(geom)

    @property
    def fShowFactor(self):
        return qcel.constants.hartree2aJ / qcel.constants.bohr2angstroms

    # Compute and return in-place array of first derivative (row of B matrix)
    def DqDx(self, geom, dqdx):
        dqdx[3 * self.A + self._xyz] = 1.0
        return

    # Do nothing, derivative B matrix is zero.
    def Dq2Dx2(self, geom, dq2dx2):
        pass

    def diagonalHessianGuess(self, geom, Z, connectivity, guessType):
        """ Generates diagonal empirical Hessians in a.u. such as 
          Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
          Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
        """
        return 0.1
