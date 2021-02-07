import logging
import math

import numpy as np
import qcelemental as qcel

from . import v3d
from .exceptions import AlgError, OptError
from .misc import delta, hguess_lindh_rho, string_math_fx
from .simple import Simple


class Bend(Simple):
    """bend coordinate between three atoms a-b-c

    Parameters
    ----------
    a : int
        first atom
    b : int
        second (middle) atom
    c : int
        third atom
    constraint : string
        set stretch as 'free', 'frozen', 'ranged', etc.
    bend_type : string, optional
        can be regular, linear, or complement (used to describe linear bends)
    range_min : float
        don't let value get smaller than this
    range_max : float
        don't let value get larger than this
    ext_force : string_math_fx
        class for evaluating additional external force
    """

    def __init__(
        self,
        a,
        b,
        c,
        constraint="free",
        bend_type="REGULAR",
        axes_fixed=False,
        range_min=None,
        range_max=None,
        ext_force=None,
    ):

        if a < c:
            atoms = (a, b, c)
        else:
            atoms = (c, b, a)

        self.bend_type = bend_type
        self._axes_fixed = axes_fixed
        self._x = np.zeros(3)
        self._w = np.zeros(3)

        Simple.__init__(self, atoms, constraint, range_min, range_max, ext_force)

    def __str__(self):
        if self.frozen:
            s = "*"
        elif self.ranged:
            s = "["
        else:
            s = " "

        if self.has_ext_force:
            s += ">"

        if self.bend_type == "REGULAR":
            s += "B"
        elif self.bend_type == "LINEAR":
            s += "L"
        elif self.bend_type == "COMPLEMENT":
            s += "l"

        s += "(%d,%d,%d)" % (self.A + 1, self.B + 1, self.C + 1)
        if self.ranged:
            s += "[{:.2f},{:.2f}]".format(self.range_min * self.q_show_factor, self.range_max * self.q_show_factor)
        return s

    def __eq__(self, other):
        if self.atoms != other.atoms:
            return False
        elif not isinstance(other, Bend):
            return False
        elif self.bend_type != other.bend_type:
            return False
        else:
            return True

    @property
    def axes_fixed(self):
        return self._axes_fixed

    @property
    def bend_type(self):
        return self._bendType

    @bend_type.setter
    def bend_type(self, intype):
        if intype in ["REGULAR", "LINEAR", "COMPLEMENT"]:
            self._bendType = intype
        else:
            raise OptError("Bend.bend_type must be REGULAR, LINEAR, or COMPLEMENT")

    def compute_axes(self, geom):
        u = v3d.eAB(geom[self.B], geom[self.A])  # B->A
        v = v3d.eAB(geom[self.B], geom[self.C])  # B->C

        if self._bendType == "REGULAR":  # not a linear-bend type
            self._w[:] = v3d.cross(u, v)  # orthogonal vector
            v3d.normalize(self._w)
            self._x[:] = u + v  # angle bisector
            v3d.normalize(self._x)
            return

        tv1 = np.array([1, 0, 0], float)  # hope not to create 2 bends that both break
        tv2 = np.array([0, 0, 1], float)  # a symmetry plane, so 2nd is off-axis
        v3d.normalize(tv2)

        u_tv1 = v3d.are_parallel_or_antiparallel(u, tv1)
        v_tv1 = v3d.are_parallel_or_antiparallel(v, tv1)
        u_tv2 = v3d.are_parallel_or_antiparallel(u, tv2)
        v_tv2 = v3d.are_parallel_or_antiparallel(v, tv2)

        # handle both types of linear bends
        if not v3d.are_parallel_or_antiparallel(u, v):
            self._w[:] = v3d.cross(u, v)  # orthogonal vector
            v3d.normalize(self._w)
            self._x[:] = u + v  # angle bisector
            v3d.normalize(self._x)
        # u || v but not || to tv1.
        elif not u_tv1 and not v_tv1:
            self._w[:] = v3d.cross(u, tv1)
            v3d.normalize(self._w)
            self._x[:] = v3d.cross(self._w, u)
            v3d.normalize(self._x)
        # u || v but not || to tv2.
        elif not u_tv2 and not v_tv2:
            self._w[:] = v3d.cross(u, tv2)
            v3d.normalize(self._w)
            self._x[:] = v3d.cross(self._w, u)
            v3d.normalize(self._x)

        if self._bendType == "COMPLEMENT":
            w2 = np.copy(self._w)  # x_normal -> w_complement
            self._w[:] = -1.0 * self._x  # -w_normal -> x_complement
            self._x[:] = w2
            del w2

        return

    def q(self, geom):
        logger = logging.getLogger(__name__)
        # check, phi = v3d.angle(geom[self.A], geom[self.B], geom[self.C])
        # printxopt('Traditional Angle = %15.10f\n', phi)

        if not self._axes_fixed:
            self.compute_axes(geom)

        u = v3d.eAB(geom[self.B], geom[self.A])  # B->A
        v = v3d.eAB(geom[self.B], geom[self.C])  # B->C

        # linear bend is sum of 2 angles, u.x + v.x
        origin = np.zeros(3)
        try:
            phi = v3d.angle(u, origin, self._x)
        except AlgError as error:
            logger.error("Bend.q could not compute linear bend")
            raise

        try:
            phi2 = v3d.angle(self._x, origin, v)
        except AlgError as error:
            logger.error("Bend.q could not compute linear bend")
            raise
        else:
            phi += phi2
            return phi

    @property
    def q_show_factor(self):
        return 180.0 / math.pi

    def q_show(self, geom):  # return in degrees
        return self.q(geom) * self.q_show_factor

    @property
    def f_show_factor(self):
        return qcel.constants.hartree2aJ * math.pi / 180.0

    @staticmethod
    def zeta(a, m, n):
        if a == m:
            return 1
        elif a == n:
            return -1
        else:
            return 0

    def fix_bend_axes(self, geom):
        if self.bend_type == "LINEAR" or self.bend_type == "COMPLEMENT":
            self.compute_axes(geom)
            self._axes_fixed = True

    def unfix_bend_axes(self):
        self._axes_fixed = False

    def to_dict(self):
        d = {}
        d["type"] = Bend.__name__  # 'Bend'
        d["atoms"] = self.atoms  # id to a tuple
        d["constraint"] = self.constraint
        d["range_min"] = self.range_min
        d["range_max"] = self.range_max
        d["bend_type"] = self.bend_type
        d["axes_fixed"] = self.axes_fixed
        if self.has_ext_force:
            d["ext_force_str"] = self.ext_force.formula_string
        else:
            d["ext_force_str"] = None
        return d

    @classmethod
    def from_dict(cls, d):
        a = d["atoms"][0]
        b = d["atoms"][1]
        c = d["atoms"][2]
        constraint = d.get("constraint", "free")
        range_min = d.get("range_min", None)
        range_max = d.get("range_max", None)
        bend_type = d.get("bend_type", "REGULAR")
        axes_fixed = d.get("axes_fixed", False)
        fstr = d.get("ext_force_str", None)
        if fstr is None:
            ext_force = None
        else:
            ext_force = string_math_fx(fstr)
        return cls(a, b, c, constraint, bend_type, axes_fixed, range_min, range_max, ext_force)

    def DqDx(self, geom, dqdx, mini=False):
        if not self.axes_fixed:
            self.compute_axes(geom)

        u = geom[self.A] - geom[self.B]  # B->A
        v = geom[self.C] - geom[self.B]  # B->C
        Lu = v3d.norm(u)  # RBA
        Lv = v3d.norm(v)  # RBC
        u[:] *= 1.0 / Lu  # u = eBA
        v[:] *= 1.0 / Lv  # v = eBC

        uXw = v3d.cross(u, self._w)
        wXv = v3d.cross(self._w, v)

        # B = overall index of atom; a = 0,1,2 relative index for delta's
        for a, B in enumerate(self.atoms):
            dqdx[3 * B : 3 * B + 3] = Bend.zeta(a, 0, 1) * uXw[0:3] / Lu + Bend.zeta(a, 2, 1) * wXv[0:3] / Lv
        return

    # Return derivative B matrix elements.  Matrix is cart X cart and passed in.
    # TODO update with jet turneys code
    def Dq2Dx2(self, geom, dq2dx2):

        if not self.axes_fixed:
            self.compute_axes(geom)

        u = geom[self.A] - geom[self.B]  # B->A
        v = geom[self.C] - geom[self.B]  # B->C
        Lu = v3d.norm(u)  # RBA
        Lv = v3d.norm(v)  # RBC
        u *= 1.0 / Lu  # eBA
        v *= 1.0 / Lv  # eBC

        uXw = v3d.cross(u, self._w)
        wXv = v3d.cross(self._w, v)

        # packed, or mini dqdx where columns run only over 3 atoms
        dqdx = np.zeros(9)
        for a in range(3):
            dqdx[3 * a : 3 * a + 3] = Bend.zeta(a, 0, 1) * uXw[0:3] / Lu + Bend.zeta(a, 2, 1) * wXv[0:3] / Lv

        val = self.q(geom)
        cos_q = math.cos(val)  # cos_q = v3d_dot(u,v);

        # leave 2nd derivatives empty - sin 0 = 0 in denominator
        if 1.0 - cos_q * cos_q <= 1.0e-12:
            return
        sin_q = math.sqrt(1.0 - cos_q * cos_q)

        for a in range(3):
            for i in range(3):  # i = a_xyz
                for b in range(3):
                    for j in range(3):  # j=b_xyz
                        tval = (
                            Bend.zeta(a, 0, 1)
                            * Bend.zeta(b, 0, 1)
                            * (u[i] * v[j] + u[j] * v[i] - 3 * u[i] * u[j] * cos_q + delta(i, j) * cos_q)
                            / (Lu * Lu * sin_q)
                        )

                        tval += (
                            Bend.zeta(a, 2, 1)
                            * Bend.zeta(b, 2, 1)
                            * (v[i] * u[j] + v[j] * u[i] - 3 * v[i] * v[j] * cos_q + delta(i, j) * cos_q)
                            / (Lv * Lv * sin_q)
                        )

                        tval += (
                            Bend.zeta(a, 0, 1)
                            * Bend.zeta(b, 2, 1)
                            * (u[i] * u[j] + v[j] * v[i] - u[i] * v[j] * cos_q - delta(i, j))
                            / (Lu * Lv * sin_q)
                        )

                        tval += (
                            Bend.zeta(a, 2, 1)
                            * Bend.zeta(b, 0, 1)
                            * (v[i] * v[j] + u[j] * u[i] - v[i] * u[j] * cos_q - delta(i, j))
                            / (Lu * Lv * sin_q)
                        )

                        tval -= cos_q / sin_q * dqdx[3 * a + i] * dqdx[3 * b + j]

                        dq2dx2[3 * self.atoms[a] + i, 3 * self.atoms[b] + j] = tval
        return

    def diagonal_hessian_guess(self, geom, Z, connectivity, guess_type="SIMPLE"):
        """Generates diagonal empirical Hessians in a.u. such as
        Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
        Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
        """
        if guess_type == "SIMPLE":
            return 0.2

        elif guess_type == "SCHLEGEL":
            if Z[self.A] == 1 or Z[self.C] == 1:
                return 0.160
            else:
                return 0.250

        elif guess_type == "FISCHER":
            a = 0.089
            b = 0.11
            c = 0.44
            d = -0.42
            Rcov_AB = qcel.covalentradii.get(Z[self.A], missing=4.0) + qcel.covalentradii.get(Z[self.B], missing=4.0)
            Rcov_BC = qcel.covalentradii.get(Z[self.C], missing=4.0) + qcel.covalentradii.get(Z[self.B], missing=4.0)
            R_AB = v3d.dist(geom[self.A], geom[self.B])
            R_BC = v3d.dist(geom[self.B], geom[self.C])
            return a + b / (np.power(Rcov_AB * Rcov_BC, d)) * np.exp(-c * (R_AB + R_BC - Rcov_AB - Rcov_BC))

        elif guess_type == "LINDH_SIMPLE":
            R_AB = v3d.dist(geom[self.A], geom[self.B])
            R_BC = v3d.dist(geom[self.B], geom[self.C])
            k_phi = 0.15
            Lindh_Rho_AB = hguess_lindh_rho(Z[self.A], Z[self.B], R_AB)
            Lindh_Rho_BC = hguess_lindh_rho(Z[self.B], Z[self.C], R_BC)
            return k_phi * Lindh_Rho_AB * Lindh_Rho_BC

        else:
            # printxopt("Warning: Hessian guess encountered unknown coordinate type.\n")
            return 1.0
