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
        if self.fixed_eq_val:
            s += "[%.4f]" % (self.fixed_eq_val * self.q_show_factor)
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
    def q_show_factor(self):
        return qcel.constants.bohr2angstroms

    def q_show(self, geom):
        return self.q_show_factor * self.q(geom)

    @property
    def f_show_factor(self):
        return qcel.constants.hartree2aJ / qcel.constants.bohr2angstroms

    # Compute and return in-place array of first derivative (row of B matrix)
    def DqDx(self, geom, dqdx, mini=False):
        dqdx[3 * self.A + self._xyz] = 1.0
        return

    # Do nothing, derivative B matrix is zero.
    def Dq2Dx2(self, geom, dq2dx2):
        pass

    def diagonal_hessian_guess(self, geom, Z, connectivity, guess_type="Simple"):
        """ Generates diagonal empirical Hessians in a.u. such as 
          Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
          Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
        """
        return 0.1
