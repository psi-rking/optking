import physconst as pc

from simple import *
import v3d
import covRadii
from misc import delta, ZtoPeriod
import numpy as np

class STRE(SIMPLE):

    def __init__(self, a, b, frozen=False, fixedEqVal=None, inverse=False):

        self._inverse = inverse  # bool - is really 1/R coordinate?

        if a < b:  atoms = (a, b)
        else:      atoms = (b, a)

        SIMPLE.__init__(self, atoms, frozen, fixedEqVal)

    def __str__(self):
        if self.frozen: s = '*'
        else:           s = ' '

        if self.inverse: s += '1/R'
        else:            s += 'R'

        s += "(%d,%d)" % (self.A+1, self.B+1)
        if self.fixedEqVal:
            s += "[%.4f]" % self.fixedEqVal
        return s

    def __eq__(self, other):
        if self.atoms != other.atoms: return False
        elif not isinstance(other,STRE): return False
        elif self.inverse != other.inverse: return False
        else: return True

    @property
    def inverse(self):
        return self._inverse

    @inverse.setter
    def inverse(self, setval):
        self._inverse = bool(setval)

    def q(self, geom):
        return v3d.dist( geom[self.A], geom[self.B] )

    def qShow(self, geom):
        return self.showQFactor * self.q(geom)

    @property
    def qShowFactor(self):
        return pc.bohr2angstroms

    @property
    def fShowFactor(self):
        return pc.hartree2aJ/pc.bohr2angstroms 

    # If mini == False, dqdx is 2x(3*number of atoms in fragment).
    # if mini == True, dqdx is 2x6.
    def DqDx(self, geom, dqdx, mini=False):
        check, eAB = v3d.eAB(geom[self.A], geom[self.B]) # A->B
        if not check:
            raise INTCO_EXCEPT("STRE.DqDx: could not normalize s vector")

        if mini:
            startA = 0
            startB = 3
        else:
            startA = 3*self.A
            startB = 3*self.B

        dqdx[startA : startA+3] = -1 * eAB[0:3]
        dqdx[startB : startB+3] =      eAB[0:3]

        if (self._inverse):
            val = self.q(geom); 
            dqdx[startA : startA+3] *= -1.0*val*val # -(1/R)^2 * (dR/da)
            dqdx[startB : startB+3] *= -1.0*val*val 

        return

    # Return derivative B matrix elements.  Matrix is cart X cart and passed in.
    def Dq2Dx2(self, geom, dq2dx2):
        check, eAB = v3d.eAB(geom[self.A], geom[self.B]) # A->B
        if not check:
            raise INTCO_EXCEPT("STRE.Dq2Dx2: could not normalize s vector")

        if not self._inverse:
            length = self.q(geom)

            for a in range(2):
               for a_xyz in range(3):
                  for b in range(2):
                     for b_xyz in range(3):
                        tval = (eAB[a_xyz] * eAB[b_xyz] - delta(a_xyz,b_xyz))/length
                        if a == b: 
                           tval *= -1.0
                        dq2dx2[3*self.atoms[a]+a_xyz, \
                               3*self.atoms[b]+b_xyz] = tval

        else: # using 1/R
            val = self.q(geom)

            dqdx = np.zeros( (3*len(self.atoms)), float)
            self.DqDx(geom, dqdx, mini=True) # returned matrix is 1x6 for stre

            for a in range(a):
               for a_xyz in range(3):
                  for b in range(b):
                     for b_xyz in range(3):
                        dq2dx2[3*self.atoms[a]+a_xyz, 3*self.atoms[b]+b_xyz] \
                               = 2.0 / val * dqdx[3*a+a_xyz] * dqdx[3*b+b_xyz]

        return

    def diagonalHessianGuess(self, geom, Z, guess = "SIMPLE"):
        """ Generates diagonal empirical Hessians in a.u. such as 
          Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
          Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
        """
        if guess == "SIMPLE":
            return 0.5
	if guess == "SCHLEGEL":
	    Rcov = (covRadii.R[Z[self.A]]+covRadii.R[Z[self.B]])

	    PerA = ZtoPeriod(self.A)
	    PerB = ZtoPeriod(self.B)

	    A = 1.734
	    if   PerA == 1 and PerB == 1:
	        B = -0.244
	    elif PerA == 1 and PerB == 2:
		B = 0.352
	    elif PerA == 2 and PerB == 2:
		B = 1.085
	    elif PerA == 1 and PerB == 3:
		B = 0.660
	    elif PerA == 2 and PerB == 3:
		B = 1.522
	    elif PerA == 3 and PerB == 3:
		B = 2.068

	    F = A/((Rcov-B)*(Rcov-B)*(Rcov-B))

	    return F

#	if guess == "FISCHER":

#	if guess == "LINDH_SIMPLE":
   
        else:
            print "Warning: Hessian guess encountered unknown coordinate type."
            return 1.0

class HBOND(STRE):
    def __str__(self):
        if self.frozen: s = '*'
        else:           s = ' '

        if self.inverse: s += '1/H'
        else:            s += 'H'

        s += "(%d,%d)" % (self.A+1, self.B+1)
        if self.fixedEqVal:
            s += "[%.4f]" % self.fixedEqVal
        return s

    # overrides STRE eq in comparisons, regardless of order
    def __eq__(self, other):
        if self.atoms != other.atoms: return False
        elif not isinstance(other,HBOND): return False
        elif self.inverse != other.inverse: return False
        else: return True


    def diagonalHessianGuess(self, geom, Z, guess="SIMPLE"):
        """ Generates diagonal empirical Hessians in a.u. such as 
          Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
          Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
        """
        if guess == "SIMPLE":
            return 0.1
        else:
            print "Warning: Hessian guess encountered unknown coordinate type."
            return 1.0

