from abc import ABCMeta, abstractmethod
from .exceptions import AlgError, OptError


class Simple(object):
    __metaclass__ = ABCMeta

    def __init__(self, atoms, frozen=False, fixed_eq_val=None):
        # these lines use the property's and setters below
        self.atoms = atoms  # atom indices for internal definition
        self.frozen = frozen  # bool - is internal coordinate frozen?
        self.fixed_eq_val = fixed_eq_val  # target value if artificial forces are to be added

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, values):
        try:
            for v in values:
                if int(v) < 0:
                    raise OptError('Atom identifier cannot be negative.')
        except TypeError:
            raise OptError('Atoms must be iterable list of whole numbers.')
        self._atoms = values

    @property
    def frozen(self):
        return self._frozen

    @property
    def fixed(self):
        if self._fixedEqVal is None:
            return False
        else:
            return True

    @frozen.setter
    def frozen(self, setval):
        self._frozen = bool(setval)
        return

    @property
    def fixed_eq_val(self):
        return self._fixedEqVal

    @fixed_eq_val.setter
    def fixed_eq_val(self, qTarget=None):
        if qTarget is not None:
            try:
                float(qTarget)
            except:
                raise OptError("Eq. value must be a float or None.")
        self._fixedEqVal = qTarget

    @property
    def atom_a(self):
        try:
            return self.atoms[0]
        except:
            raise OptError("atom_a() called but atoms[0] does not exist")

    @property
    def atom_b(self):
        try:
            return self.atoms[1]
        except:
            raise OptError("atom_b() called but atoms[1] does not exist")

    @property
    def atom_c(self):
        try:
            return self.atoms[2]
        except:
            raise OptError("connectivity_mat() called but atoms[2] does not exist")

    @property
    def atom_d(self):
        try:
            return self.atoms[3]
        except:
            raise OptError("atom_d() called but atoms[3] does not exist")

    # ** constructor + 7 abstract methods are currently required **
    @abstractmethod  # Given geometry, return value in Bohr or radians
    def q(self, geom):
        pass

    @abstractmethod  # Given geometry, return Value in Angstroms or degrees.
    def q_show(self, geom):
        pass

    @abstractmethod  # Return the scalar needed to convert value in au to Ang or Deg
    def q_show_factor(self):
        pass

    @abstractmethod  # Return the scalar needed to convert force in au to aJ/(Ang or Deg)
    def f_show_factor(self):
        pass

    # Modify provided dqdx array with first derivative of value wrt cartesians
    #  i.e., provide rows of atom_b matrix.
    #   Num. of rows is len(self._atoms), or Num. of atoms in coordinate definition
    # By default, col dimension of dqdx is assumed to be 3*(Num. of atoms in fragment,
    #  or the number of atoms consistent with the values of self._atoms).
    # If mini==True, then col dimension of dqdx is only 3*len(self._atoms).  For a stretch
    # then, e.g, dqdx is 2x6.
    @abstractmethod
    def dqdx(self, geom, dqdx, mini=False):
        raise AlgError('no dqdx for this coordinate')

    # Modify provided dq2dx2 array with second derivative of value wrt cartesians
    #  i.e., provide derivative atom_b matrix for coordinate.
    # dimension of dq2dx2 is 3*len(self._atoms)x3*len(self._atoms), or
    # cartesian by cartesian - of minimum size.
    @abstractmethod  # Derivative of value wrt cartesians, i.e., atom_b-matrix elements.
    def dq2dx2(self, geom, dq2dx2):
        raise AlgError('no dq2dx2 for this coordinate')

    @abstractmethod  # Diagonal hessian guess
    def diagonal_hessian_guess(geom, Z, connectivity, guessType):
        raise AlgError('no hessian guess for this coordinate')
