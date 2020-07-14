import math
import logging

import numpy as np
import qcelemental as qcel

from .exceptions import AlgError, OptError
from . import optparams as op
from . import v3d
from .misc import hguess_lindh_rho
from .simple import Simple


class Tors(Simple):
    def __init__(self, a, b, c, d, frozen=False, fixed_eq_val=None):

        if a < d: atoms = (a, b, c, d)
        else: atoms = (d, c, b, a)
        self._near180 = 0

        Simple.__init__(self, atoms, frozen, fixed_eq_val)

    def __str__(self):
        if self.frozen: s = '*'
        else: s = ' '

        s += "D"

        s += "(%d,%d,%d,%d)" % (self.A + 1, self.B + 1, self.C + 1, self.D + 1)
        if self.fixed_eq_val:
            s += "[%.1f]" % (self.fixed_eq_val * self.q_show_factor)
        return s

    def __eq__(self, other):
        if self.atoms != other.atoms:
            return False
        elif not isinstance(other, Tors):
            return False
        else:
            return True

    @property
    def near180(self):
        return self._near180

    # keeps track of orientation
    def update_orientation(self, geom):
        tval = self.q(geom)
        if tval > op.Params.fix_val_near_pi:
            self._near180 = +1
        elif tval < -1 * op.Params.fix_val_near_pi:
            self._near180 = -1
        else:
            self._near180 = 0
        return

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

    # compute angle and return value in radians
    def q(self, geom):
        try:
            tau = v3d.tors(geom[self.A], geom[self.B], geom[self.C], geom[self.D])
        except AlgError as error:
            raise AlgError("Tors.q: unable to compute torsion value") from error

        # Extend values domain of torsion angles beyond pi or -pi, so that
        # delta(values) can be calculated
        if self._near180 == -1 and tau > op.Params.fix_val_near_pi:
            return tau - 2.0 * math.pi
        elif self._near180 == +1 and tau < -1 * op.Params.fix_val_near_pi:
            return tau + 2.0 * math.pi
        else:
            return tau

    def DqDx(self, geom, dqdx, mini=False):
        u = geom[self.A] - geom[self.B]  # u=m-o eBA
        v = geom[self.D] - geom[self.C]  # v=n-p eCD
        w = geom[self.C] - geom[self.B]  # w=p-o eBC
        Lu = v3d.norm(u)  # RBA
        Lv = v3d.norm(v)  # RCD
        Lw = v3d.norm(w)  # RBC
        u *= 1.0 / Lu  # eBA
        v *= 1.0 / Lv  # eCD
        w *= 1.0 / Lw  # eBC

        cos_u = v3d.dot(u, w)
        cos_v = -v3d.dot(v, w)

        # abort and leave zero if 0 or 180 angle
        if 1.0 - cos_u * cos_u <= 1.0e-12 or 1.0 - cos_v * cos_v <= 1.0e-12:
            return

        sin_u = math.sqrt(1.0 - cos_u * cos_u)
        sin_v = math.sqrt(1.0 - cos_v * cos_v)
        uXw = v3d.cross(u, w)
        vXw = v3d.cross(v, w)

        # a = relative index; B = full index of atom
        for a, B in enumerate(self.atoms):
            for i in range(3):  # i=a_xyz
                tval = 0.0

                if a == 0 or a == 1:
                    tval += Tors.zeta(a, 0, 1) * uXw[i] / (Lu * sin_u * sin_u)

                if a == 2 or a == 3:
                    tval += Tors.zeta(a, 2, 3) * vXw[i] / (Lv * sin_v * sin_v)

                if a == 1 or a == 2:
                    tval += Tors.zeta(a, 1, 2) * uXw[i] * cos_u / (Lw * sin_u * sin_u)

                # "+" sign for zeta(a,2,1)) differs from JCP, 117, 9164 (2002)
                if a == 1 or a == 2:
                    tval += -Tors.zeta(a, 2, 1) * vXw[i] * cos_v / (Lw * sin_v * sin_v)

                if not mini:
                    dqdx[3 * B + i] = tval
                else:
                    dqdx[3 * a + i] = tval
        return

    # There are several errors in JCP, 22, 9164, (2002).
    # I identified incorrect signs by making the equations invariant to reversing the atom indices
    # (0,1,2,3) -> (3,2,1,0) and checking terms against finite differences.  Also, the last terms
    # with sin^2 in the denominator are incorrectly given as only sin^1 in the paper.
    # Torsion is m-o-p-n.  -RAK 2010
    def Dq2Dx2(self, geom, dq2dx2):
        u = geom[self.A] - geom[self.B]  # u=m-o eBA
        v = geom[self.D] - geom[self.C]  # v=n-p eCD
        w = geom[self.C] - geom[self.B]  # w=p-o eBC
        Lu = v3d.norm(u)  # RBA
        Lv = v3d.norm(v)  # RCD
        Lw = v3d.norm(w)  # RBC
        u *= 1.0 / Lu  # eBA
        v *= 1.0 / Lv  # eCD
        w *= 1.0 / Lw  # eBC

        cos_u = v3d.dot(u, w)
        cos_v = -v3d.dot(v, w)

        # Abort and leave zero if 0 or 180 angle
        if 1.0 - cos_u * cos_u <= 1.0e-12 or 1.0 - cos_v * cos_v <= 1.0e-12:
            return

        sin_u = math.sqrt(1.0 - cos_u * cos_u)
        sin_v = math.sqrt(1.0 - cos_v * cos_v)
        uXw = v3d.cross(u, w)
        vXw = v3d.cross(v, w)

        sinu4 = sin_u * sin_u * sin_u * sin_u
        sinv4 = sin_v * sin_v * sin_v * sin_v
        cosu3 = cos_u * cos_u * cos_u
        cosv3 = cos_v * cos_v * cos_v

        # int k; // cartesian ; not i or j
        for a in range(4):
            for b in range(a + 1):
                for i in range(3):  # i = a_xyz
                    for j in range(3):  # j=b_xyz
                        tval = 0

                        if (a == 0 and b == 0) or (a == 1 and b == 0) or (a == 1
                                                                          and b == 1):
                            tval +=  Tors.zeta(a,0,1) * Tors.zeta(b,0,1) * \
                             (uXw[i]*(w[j]*cos_u-u[j]) + uXw[j]*(w[i]*cos_u-u[i]))/(Lu*Lu*sinu4)

                        # above under reversal of atom indices, u->v ; w->(-w) ; uXw->(-uXw)
                        if (a == 3 and b == 3) or (a == 3 and b == 2) or (a == 2
                                                                          and b == 2):
                            tval += Tors.zeta(a,3,2) * Tors.zeta(b,3,2) * \
                             (vXw[i]*(w[j]*cos_v+v[j]) + vXw[j]*(w[i]*cos_v+v[i]))/(Lv*Lv*sinv4)

                        if (a == 1 and b == 1) or (a == 2 and b == 1) or (
                                a == 2 and b == 0) or (a == 1 and b == 0):
                            tval += (Tors.zeta(a,0,1) * Tors.zeta(b,1,2) + Tors.zeta(a,2,1) * Tors.zeta(b,1,0))*\
                             (uXw[i] * (w[j] - 2*u[j]*cos_u + w[j]*cos_u*cos_u) +
                              uXw[j] * (w[i] - 2*u[i]*cos_u + w[i]*cos_u*cos_u)) / (2*Lu*Lw*sinu4)

                        if (a == 3 and b == 2) or (a == 3 and b == 1) or (
                                a == 2 and b == 2) or (a == 2 and b == 1):
                            tval += (Tors.zeta(a,3,2) * Tors.zeta(b,2,1) + Tors.zeta(a,1,2) * Tors.zeta(b,2,3))*\
                             (vXw[i] * (w[j] + 2*v[j]*cos_v + w[j]*cos_v*cos_v) +
                              vXw[j] * (w[i] + 2*v[i]*cos_v + w[i]*cos_v*cos_v)) / (2*Lv*Lw*sinv4)

                        if (a == 1 and b == 1) or (a == 2 and b == 2) or (a == 2
                                                                          and b == 1):
                            tval +=  Tors.zeta(a,1,2) * Tors.zeta(b,2,1) * \
                             (uXw[i]*(u[j] + u[j]*cos_u*cos_u - 3*w[j]*cos_u + w[j]*cosu3) +
                              uXw[j]*(u[i] + u[i]*cos_u*cos_u - 3*w[i]*cos_u + w[i]*cosu3)) / (2*Lw*Lw*sinu4)

                        if (a == 2 and b == 1) or (a == 2 and b == 2) or (a == 1
                                                                          and b == 1):
                            tval += Tors.zeta(a,2,1) * Tors.zeta(b,1,2) * \
                             (vXw[i]*(-v[j] - v[j]*cos_v*cos_v - 3*w[j]*cos_v + w[j]*cosv3) +
                              vXw[j]*(-v[i] - v[i]*cos_v*cos_v - 3*w[i]*cos_v + w[i]*cosv3)) / (2*Lw*Lw*sinv4)

                        if (a != b) and (i != j):
                            if i != 0 and j != 0:
                                k = 0  # k is unique coordinate not i or j
                            elif i != 1 and j != 1:
                                k = 1
                            else:
                                k = 2
                            # TODO are these powers correct ?  -0.5^( |j-i| w[k]cos(u)-u[k], e.g. ?

                            if a == 1 and b == 1:
                                tval += Tors.zeta(a,0,1) * Tors.zeta(b,1,2) * (j-i) * \
                                  pow(-0.5, math.fabs(j-i)) * (+w[k]*cos_u - u[k]) / (Lu*Lw*sin_u*sin_u)

                            if (a == 3 and b == 2) or (a == 3 and b == 1) or (
                                    a == 2 and b == 2) or (a == 2 and b == 1):
                                tval += Tors.zeta(a,3,2) * Tors.zeta(b,2,1) * (j-i) * \
                                  pow(-0.5, math.fabs(j-i)) * (-w[k]*cos_v - v[k]) / (Lv*Lw*sin_v*sin_v)

                            if (a == 2 and b == 1) or (a == 2 and b == 0) or (
                                    a == 1 and b == 1) or (a == 1 and b == 0):
                                tval += Tors.zeta(a,2,1) * Tors.zeta(b,1,0) * (j-i) * \
                                  pow(-0.5, math.fabs(j-i)) * (-w[k]*cos_u + u[k]) / (Lu*Lw*sin_u*sin_u)

                            if a == 2 and b == 2:
                                tval += Tors.zeta(a,1,2) * Tors.zeta(b,2,3) * (j-i) * \
                                  pow(-0.5, math.fabs(j-i)) * (+w[k]*cos_v + v[k]) / (Lv*Lw*sin_v*sin_v)

                        dq2dx2[3*self.atoms[a]+i][3*self.atoms[b]+j] = \
                        dq2dx2[3*self.atoms[b]+j][3*self.atoms[a]+i] = tval
        return

    def diagonal_hessian_guess(self, geom, Z, connectivity, guess_type="SIMPLE"):
        """ Generates diagonal empirical Hessians in a.u. such as 
          Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
          Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
        """

        logger = logging.getLogger(__name__)

        if guess_type == "SIMPLE":
            return 0.1

        elif guess_type == "SCHLEGEL":
            R_BC = v3d.dist(geom[self.B], geom[self.C])
            Rcov = qcel.covalentradii.get(Z[self.B], missing=4.0) + qcel.covalentradii.get(Z[self.C], missing=4.0)
            a = 0.0023
            b = 0.07
            if R_BC > (Rcov + a / b):
                b = 0.0
            return a - (b * (R_BC - Rcov))

        elif guess_type == "FISCHER":
            R = v3d.dist(geom[self.B], geom[self.C])
            Rcov = qcel.covalentradii.get(Z[self.B], missing=4.0) + qcel.covalentradii.get(Z[self.C], missing=4.0)
            a = 0.0015
            b = 14.0
            c = 2.85
            d = 0.57
            e = 4.00

            # Determine connectivity factor L
            Brow = connectivity[self.B]
            Crow = connectivity[self.C]
            Bbonds = 0
            Cbonds = 0
            for i in range(len(Crow)):
                Bbonds = Bbonds + Brow[i]
                Cbonds = Cbonds + Crow[i]
            L = Bbonds + Cbonds - 2
            logger.info("Connectivity of central 2 torsional atoms - 2 = L = %d\n" % L)
            return a + b * (np.power(L, d)) / (np.power(R * Rcov, e)) * (
                np.exp(-c * (R - Rcov)))

        elif guess_type == "LINDH_SIMPLE":

            R_AB = v3d.dist(geom[self.A], geom[self.B])
            R_BC = v3d.dist(geom[self.B], geom[self.C])
            R_CD = v3d.dist(geom[self.C], geom[self.D])
            k_tau = 0.005

            Lindh_Rho_AB = hguess_lindh_rho(Z[self.A], Z[self.B], R_AB)
            Lindh_Rho_BC = hguess_lindh_rho(Z[self.B], Z[self.C], R_BC)
            Lindh_Rho_CD = hguess_lindh_rho(Z[self.C], Z[self.D], R_CD)
            return k_tau * Lindh_Rho_AB * Lindh_Rho_BC * Lindh_Rho_CD

        else:
            logger.warning("""Hessian guess encountered unknown coordinate type.\n 
                As default, identity matrix is used""")
            return 1.0
