from abc import ABC, abstractmethod

from .exceptions import AlgError, OptError

supported_constraint_types = ("free", "frozen", "ranged")


class Simple(ABC):
    def __init__(self, atoms, constraint, range_min, range_max, ext_force):
        # these lines use the property's and setters below
        self.atoms = atoms  # atom indices for internal definition
        if constraint.lower() not in supported_constraint_types:
            raise OptError("status for simple intco unknown")
        self.constraint = constraint.lower()
        self._range_min = range_min
        self._range_max = range_max
        self._ext_force = ext_force

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, values):
        try:
            for v in values:
                if int(v) < 0:
                    raise OptError("Atom identifier cannot be negative.")
        except TypeError:
            raise OptError("Atoms must be iterable list of whole numbers. Received %s instead", values)
        self._atoms = values

    @property
    def frozen(self):
        return self.constraint == "frozen"

    @property
    def ranged(self):
        return self.constraint == "ranged"

    def freeze(self):
        if self.constraint == "free":
            self.constraint = "frozen"
        # for now if 'ranged', don't change
        return

    def unfreeze(self):
        if self.constraint == "frozen":
            self.constraint = "free"
        # for now if 'ranged', don't change

    def set_range(self, range_min, range_max):
        self.constraint = "ranged"
        self._range_min = range_min
        self._range_max = range_max

    @property
    def range_min(self):
        return self._range_min

    @property
    def range_max(self):
        return self._range_max

    @property
    def has_ext_force(self):
        return bool(self._ext_force is not None)

    @property
    def ext_force(self):
        return self._ext_force

    # could add checking here later
    @ext_force.setter
    def ext_force(self, eqn):
        self._ext_force = eqn

    def ext_force_val(self, geom):
        val = self.q_show(geom)  # angstroms or degrees
        return self._ext_force.evaluate(val)

    @property
    def A(self):
        try:
            return self.atoms[0]
        except:
            raise OptError("A() called but atoms[0] does not exist")

    @property
    def B(self):
        try:
            return self.atoms[1]
        except:
            raise OptError("B() called but atoms[1] does not exist")

    @property
    def C(self):
        try:
            return self.atoms[2]
        except:
            raise OptError("C() called but atoms[2] does not exist")

    @property
    def D(self):
        try:
            return self.atoms[3]
        except:
            raise OptError("d() called but atoms[3] does not exist")

    # ** constructor + 9 abstract methods are currently required
    # for full functionality **
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

    @abstractmethod  # print type/classname plus contents to export
    def to_dict(self):
        pass

    @abstractmethod  # construct from dictionary
    def from_dict(self, d):
        pass

    # Modify provided DqDx array with first derivative of value wrt cartesians
    #  i.e., provide rows of B matrix.
    #   Num. of rows is len(self._atoms), or Num. of atoms in coordinate definition
    # By default, col dimension of dqdx is assumed to be 3*(Num. of atoms in fragment,
    #  or the number of atoms consistent with the values of self._atoms).
    # If mini==True, then col dimension of dqdx is only 3*len(self._atoms).  For a stretch
    # then, e.g, DqDx is 2x6.
    @abstractmethod
    def DqDx(self, geom, dqdx, mini=False):
        raise AlgError("no DqDx for this coordinate")

    # Modify provided Dq2Dx2 array with second derivative of value wrt cartesians
    #  i.e., provide derivative B matrix for coordinate.
    # dimension of dq2dx2 is 3*len(self._atoms)x3*len(self._atoms), or
    # cartesian by cartesian - of minimum size.
    @abstractmethod  # Derivative of value wrt cartesians, i.e., B-matrix elements.
    def Dq2Dx2(self, geom, dq2dx2):
        raise AlgError("no Dq2Dx2 for this coordinate")

    @abstractmethod  # Diagonal hessian guess
    def diagonal_hessian_guess(geom, Z, connectivity, guess_type):
        raise AlgError("no hessian guess for this coordinate")
