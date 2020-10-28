import math
import logging

import qcelemental as qcel

from .exceptions import AlgError, OptError
from . import optparams as op
from . import v3d
from .simple import Simple
from .misc import string_math_fx

# Class for out-of-plane angle.  Definition (A,B,C,D) means angle AB with respect
# to the CBD plane; canonical order is C < D

class Oofp(Simple):
    def __init__(self, a, b, c, d, constraint='free', near180=0,
                 range_min=None, range_max=None, ext_force=None):

        atoms = (a, b, c, d)
        if c < d:
            self.neg = 1
        else:
            self.neg = -1
        self._near180 = near180
        Simple.__init__(self, atoms, constraint, range_min, range_max,
                        ext_force)
        
        #try:
        #    import coordinates
        #except ImportError:
        #    raise ImportError("could not import coordinates. Sympy needed for out of " 
        #        + "plane angles. please intstall sympy - conda install sympy")
        #else:
        #    self.symbolic_coord = coordinates.OutOfPlane(atoms)
    
    def __str__(self):
        if self.frozen:
            s = '*'
        elif self.ranged:
            s = '['
        else:
            s = ' '

        s += "O"

        s += "(%d,%d,%d,%d)" % (self.A + 1, self.B + 1, self.C + 1, self.D + 1)
        if self.ranged:
            s += '[{:.2f},{:.2f}]'.format(
                     self.range_min * self.q_show_factor,
                     self.range_max * self.q_show_factor)
        return s

    def __eq__(self, other):
        if self.atoms != other.atoms:
            return False
        elif not isinstance(other, Oofp):
            return False
        else:
            return True

    @property
    def near180(self):
        return self._near180

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


    def to_dict(self):
        d = {}
        d['type'] = Oofp.__name__ # 'Oofp'
        d['atoms'] = self.atoms # id to a tuple
        d['constraint'] = self.constraint
        d['range_min'] = self.range_min
        d['range_max'] = self.range_max
        d['near180'] = self.near180
        if self.has_ext_force:
            d['ext_force_str'] = self.ext_force.formula_string
        else:
            d['ext_force_str'] = None
        return d


    @classmethod
    def from_dict(cls, D):
        a = D['atoms'][0]
        b = D['atoms'][1]
        c = D['atoms'][2]
        d = D['atoms'][3]
        constraint = D.get('constraint', 'free')
        range_min = D.get('range_min', None)
        range_max = D.get('range_max', None)
        near180 = D.get('near180', 0)
        fstr = D.get('ext_force_str', None)
        if fstr is None:
            ext_force = None
        else:
            ext_force = string_math_fx(fstr)
        return cls(a, b, c, d, constraint, near180, range_min, range_max,
                   ext_force)


    def q(self, geom):
        """Compute torsion angle for geometry.

        Parameters
        ----------
        geom : ndarray
            (nat, 3) array of Cartesian coordinates [a0]

        Returns
        -------
        float
            Torsion angle [rad]

        """
        try:
            tau = v3d.oofp(geom[self.A], geom[self.B], geom[self.C], geom[self.D])
        except AlgError as error:
            raise

        # Extend domain of out-of-plane angles to beyond pi
        if self._near180 == -1 and tau > op.Params.fix_val_near_pi:
            return tau - 2.0 * math.pi
        elif self._near180 == +1 and tau < -1 * op.Params.fix_val_near_pi:
            return tau + 2.0 * math.pi
        else:
            return tau

    # out-of-plane is m-o-p-n
    # Assume angle phi_CBD is OK, or we couldn't calculate the value anyway.
    def DqDx(self, geom, dqdx, mini=False):

        #self.DqDx_sympy(geom, dqdx)
        #return
        
        eBA = geom[self.A] - geom[self.B]
        eBC = geom[self.C] - geom[self.B]
        eBD = geom[self.D] - geom[self.B]
        rBA = v3d.norm(eBA)
        rBC = v3d.norm(eBC)
        rBD = v3d.norm(eBD)
        eBA *= 1.0 / rBA
        eBC *= 1.0 / rBC
        eBD *= 1.0 / rBD

        # compute out-of-plane value, C-B-D angle
        val = self.q(geom)
        phi_CBD = v3d.angle(geom[self.C], geom[self.B], geom[self.D])

        # S vector for A
        tmp = v3d.cross(eBC, eBD)
        tmp /= math.cos(val) * math.sin(phi_CBD)
        tmp2 = math.tan(val) * eBA
        dqdx[3 * self.A:3 * self.A + 3] = self.neg * (tmp - tmp2) / rBA

        # S vector for C
        tmp = v3d.cross(eBD, eBA)
        tmp = tmp / (math.cos(val) * math.sin(phi_CBD))
        tmp2 = math.cos(phi_CBD) * eBD
        tmp3 = -1.0 * tmp2 + eBC
        tmp3 *= math.tan(val) / (math.sin(phi_CBD) * math.sin(phi_CBD))
        dqdx[3 * self.C:3 * self.C + 3] = self.neg * (tmp - tmp3) / rBC

        # S vector for D
        tmp = v3d.cross(eBA, eBC)
        tmp /= math.cos(val) * math.sin(phi_CBD)
        tmp2 = math.cos(phi_CBD) * eBC
        tmp3 = -1.0 * tmp2 + eBD
        tmp3 *= math.tan(val) / (math.sin(phi_CBD) * math.sin(phi_CBD))
        dqdx[3 * self.D:3 * self.D + 3] = self.neg * (tmp - tmp3) / rBD

        # S vector for B
        dqdx[3*self.B:3*self.B+3] = (self.neg * -1.0 * dqdx[3*self.A:3*self.A+3]
            - dqdx[3*self.C:3*self.C+3] - dqdx[3*self.D:3*self.D+3])

    def Dq2Dx2(self, geom, dqdx):
        raise AlgError('no derivative B matrices for out-of-plane angles')

    def diagonal_hessian_guess(self, geom, Z, connectivity, guess_type="SIMPLE"):
        """ Generates diagonal empirical Hessians in a.u. such as
          Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
          Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
        """
        logger = logging.getLogger(__name__)
        if guess_type == "SIMPLE":
            return 0.1

        elif guess_type == "SCHLEGEL":
            rval = 0.045 * self.q(geom)**4
            # RAK doesn't think it is optimal to be too small
            if rval < 0.001: rval = 0.001
            return rval

        elif guess_type == "FISCHER": # my A-B-C-D is their x-a-b-c
            Rax = v3d.dist(geom[self.B], geom[self.A])
            RCOVax = qcel.covalentradii.get(Z[self.B], missing=4.0) \
                   + qcel.covalentradii.get(Z[self.A], missing=4.0)
            RCOVab = qcel.covalentradii.get(Z[self.B], missing=4.0) \
                   + qcel.covalentradii.get(Z[self.C], missing=4.0)
            RCOVac = qcel.covalentradii.get(Z[self.B], missing=4.0) \
                   + qcel.covalentradii.get(Z[self.D], missing=4.0)
            a = 0.0025
            b = 0.0061
            c = 3.0
            d = 4.0
            e = 0.80
            val = self.q(geom)
            rval = a + b * (RCOVab*RCOVac)**e * math.cos(val)**d * math.exp(-c*(Rax-RCOVax))
            logger.debug('fisher: {:12.6e}'.format(rval))
            return rval

        else:
            logger.warning("Hessian guess encountered unknown coordinate type.\n")
            return 0.1

    #def q_sympy(self, geom):
    #    return self.symbolic_coord.q(geom)

    #def DqDx_sympy(self, geom, dqdx):
    #    dqdx_mat = self.symbolic_coord.dq_dx(geom)
    #    dqdx[self.A * 3: (3 * self.A) + 3] = dqdx_mat[0]
    #    dqdx[self.B * 3: (3 * self.B) + 3] = dqdx_mat[1]
    #    dqdx[self.C * 3: (3 * self.C) + 3] = dqdx_mat[2]
    #    dqdx[self.D * 3: (3 * self.D) + 3] = dqdx_mat[3]
        
