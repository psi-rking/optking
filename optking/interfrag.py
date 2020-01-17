import numpy as np
from math import sqrt

from optking import frag, stre, bend, tors
from optking import v3d
from math import fabs, acos # sin,  asin, fsum
import orient
import qcelemental as qcel

from optking import optparams as op
op.Params = op.OptParams({})

class Weight(object):
    def __init__(self, a, w):
        self._atom = a    # int:   index of atom in fragment
        self._weight = w  # float: weight of atom for reference point

    @property
    def atom(self):
        return self._atom

    @property
    def weight(self):
        return self._weight


class RefPoint(object):
    """ Collection of weights for a single reference point. """
    def __init__(self, atoms, coeff):
        self._weights = []
        if len(atoms) != len(coeff):
            raise OptError("Number of atoms and weights for ref. pt. differ")

        # Normalize the weights.
        norm = sqrt(sum(c*c for c in coeff))
        for i in range(len(atoms)):
            self._weights.append( Weight(atoms[i],coeff[i]/norm) )

    def __iter__(self):
        return (w for w in self._weights)

    def __len__(self):
        return len(self._weights)

    def __str__(self):
        s = "\t\t\t Atom            Coeff\n"
        for w in self:
             s+= "\t\t\t%5d     %15.10f\n" % (w.atom+1, w.weight)
        return s


class Interfrag(object):
    """ Set of (up to 6) coordinates between two distinct fragments.
    The fragments 'A' and 'B' have up to 3 reference atoms each (dA[3] and dB[3]).
    The reference atoms are defined in one of two ways:
    1. If interfrag_mode == FIXED, then fixed, linear combinations of atoms
          in A and B are used.
    2. (NOT YET IMPLEMENTED)
       If interfrag_mode == PRINCIPAL_AXES, then the references points are
        a. the center of mass
        b. a point a unit distance along the principal axis corresponding to the largest moment.
        c. a point a unit distance along the principal axis corresponding to the 2nd largest moment.
    #
    For simplicity, we sort the atoms in the reference point structure according to
    the assumed connectivity of the coordinates.  For A, this reverses the order relative to
    that in which the weights are provided to the constructor.
    #
    ref_geom[0] = dA[2];
    ref_geom[1] = dA[1];
    ref_geom[2] = dA[0];
    ref_geom[3] = dB[0];
    ref_geom[4] = dB[1];
    ref_geom[5] = dB[2];
    # 
    The six coordinates, if present, formed from the d{A-B}{0-2} sets are assumed to be the
    following in this canonical order:
    pos sym      type      atom-definition          present, if
    ---------------------------------------------------------------------------------
    0   RAB      distance  dA[0]-dB[0]              always
    1   theta_A  angle     dA[1]-dA[0]-dB[0]        A has > 1 atom
    2   theta_B  angle     dA[0]-dB[0]-dB[1]        B has > 1 atom
    3   tau      dihedral  dA[1]-dA[0]-dB[0]-dB[1]  A and B have > 1 atom
    4   phi_A    dihedral  dA[2]-dA[1]-dA[0]-dB[0]  A has > 2 atoms and is not linear
    5   phi_B    dihedral  dA[0]-dB[0]-dB[1]-dB[2]  B has > 2 atoms and is not linear
    #
    Parameters
    ----------
    A_idx : int
        index of fragment in molecule list
    A_atoms: list of (up to 3) lists of ints
        index of atoms used to define each reference point on A
    B_idx : int
        index of fragment in molecule list
    B_atoms: list of (up to 3) lists of ints
        index of atoms used to define each reference point on B
    A_weights (optional): list of (up to 3) lists of floats
        weights of atoms used to define each reference point of A
    B_weights (optional): list of (up to 3) lists of floats
        weights of atoms used to define each reference point of A
    A_lbl : string
        name for fragment A
    B_lbl : string
        name for fragment B
    The arguments are potentially confusing, so we'll do a lot of checking.
    """
    def __init__(self, A_frag_idx, A_atoms, B_frag_idx, B_atoms, A_weights=None, B_weights=None, \
                 A_lbl="A", B_lbl="B"):
        self._A_lbl = A_lbl
        self._B_lbl = B_lbl

        if type(A_atoms) != list:
            print(type(A_atoms))
            raise OptError("Atoms argument for frag A should be a list")
        for i,a in enumerate(A_atoms):
            if type(a) != list:
                raise OptError("Atoms argument for frag A, reference pt. %d should be a list"%(i+1))

        if type(B_atoms) != list:
            raise OptError("Atoms argument for frag B should be a list")
        for i,b in enumerate(B_atoms):
            if type(b) != list:
                raise OptError("Atoms argument for frag B, reference pt. %d should be a list"%(i+1))

        if A_weights == None:
            A_weights = []
            for i in len(range(A_atoms)):
                A_weights.append( len(A_atoms[i]) * [1.0] )
        else:
            if type(A_weights) == list:
                if len(A_weights) != len(A_atoms):
                    raise OptError("Number of reference atoms and weights on frag A are inconsistent")
                for i,w in enumerate(A_weights):
                    if type(w) != list:
                        raise OptError("Weight for frag A, reference pt. %d should be a list" %(i+1))
            else:
                raise OptError("Weights for reference atoms on frag A should be a list")

        if B_weights == None:
            B_weights = []
            for i in len(range(A_atoms)):
                B_weights.append( len(B_atoms[i]) * [1.0] )
        else:
            if type(B_weights) == list:
                if len(B_weights) != len(B_atoms):
                    raise OptError("Number of reference atoms and weights on frag B are inconsistent")
                for i,w in enumerate(B_weights):
                    if type(w) != list:
                        raise OptError("Weight for frag B, reference pt. %d should be a list" %(i+1))
            else:
                raise OptError("Weights for reference atoms on frag B should be a list")

        if len(A_atoms) > 3:
            raise OptError("Too many reference atoms for frag A provided")
        if len(B_atoms) > 3:
            raise OptError("Too many reference atoms for frag B provided")

        self._Arefs = []
        self._Brefs = []
        
        for i in range(len(A_atoms)):
            if len(A_atoms[i]) != len(A_weights[i]):
                raise OptError("Number of atoms and weights for frag A, reference pt. %d differ"%(i+1))
            if len(A_atoms[i]) != len(set(A_atoms[i])):
                raise OptError("Atom used more than once for frag A, reference pt. %d."%(i+1))
            self._Arefs.append( RefPoint(A_atoms[i], A_weights[i]) )
        for i in range(len(B_atoms)):
            if len(B_atoms[i]) != len(B_weights[i]):
                raise OptError("Number of atoms and weights for frag B, reference pt. %d differ"%(i+1))
            if len(B_atoms[i]) != len(set(B_atoms[i])):
                raise OptError("Atom used more than once for frag B, reference pt. %d."%(i+1))
            self._Brefs.append( RefPoint(B_atoms[i], B_weights[i]) )

        # Construct a pseudofragment that contains the (up to) 6 reference atoms
        Z = 6 * [1]        # not used, except maybe Hessian guess ?
        ref_geom = np.zeros((6, 3), float) # some rows may be unused
        masses = 6 * [0.0] # not used
        self._pseudo_frag = frag.Frag(Z, ref_geom, masses)

        # adds the coordinates connecting A2-A1-A0-B0-B1-B2
        # sets D_on to indicate which ones (of the 6) are unusued
        # turn all coordinates on ; turn off unused ones below
        self._D_on = 6*[True]
        one_stre  = None
        one_bend  = None
        one_bend2 = None
        one_tors  = None
        one_tors2 = None
        one_tors3 = None

        nA = len(self._Arefs) # Num. of reference points.
        nB = len(self._Brefs)
      
        if nA == 3 and nB == 3:
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            one_tors  = tors.Tors(1, 2, 3, 4)  # tau
            one_tors2 = tors.Tors(0, 1, 2, 3)  # phi_A
            one_tors3 = tors.Tors(2, 3, 4, 5)  # phi_B
        elif nA == 3 and nB == 2:
            self._D_on[5] = False            # no phi_B
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            one_tors  = tors.Tors(1, 2, 3, 4)  # tau
            one_tors2 = tors.Tors(0, 1, 2, 3)  # phi_A
        elif nA == 2 and nB == 3:
            self._D_on[4] = False            # no phi_A
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            one_tors  = tors.Tors(1, 2, 3, 4)  # tau
            one_tors3 = tors.Tors(2, 3, 4, 5)  # phi_B
        elif nA == 3 and nB == 1:
            self._D_on[2] = False
            self._D_on[3] = False
            self._D_on[5] = False            # no theta_B, tau, phi_B
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            one_tors2 = tors.Tors(0, 1, 2, 3)  # phi_A
        elif nA == 1 and nB == 3:
            self._D_on[1] = False
            self._D_on[3] = False
            self._D_on[4] = False            # no theta_A, tau, phi_A
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            one_tors3 = tors.Tors(2, 3, 4, 5)  # phi_B
        elif nA == 2 and nB == 2:
            self._D_on[4] = False
            self._D_on[5] = False            # no phi_A, phi_B
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            one_tors  = tors.Tors(1, 2, 3, 4)  # tau
        elif nA == 2 and nB == 1:
            self._D_on[2] = False
            self._D_on[4] = False 
            self._D_on[5] = False            # no theta_B, phi_A, phi_B
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            one_tors  = tors.Tors(1, 2, 3, 4)  # tau
        elif nA == 1 and nB == 2:
            self._D_on[1] = False
            self._D_on[4] = False
            self._D_on[5] = False  #          no theta_A, phi_A, phi_B
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            one_tors  = tors.Tors(1, 2, 3, 4)  # tau
        elif nA == 1 and nB == 1:
            self._D_on[1] = False
            self._D_on[2] = False
            self._D_on[3] = False
            self._D_on[4] = False
            self._D_on[5] = False 
            one_stre  = stre.Stre(2, 3)        # RAB
        else:
            raise OptError("No reference points present")

        if op.Params.interfrag_dist_inv:
            one_stre.inverse = True

        if one_stre is not None:
            self._pseudo_frag._intcos.append(one_stre)
        if one_bend is not None:
            self._pseudo_frag._intcos.append(one_bend)
        if one_bend2 is not None:
            self._pseudo_frag._intcos.append(one_bend2)
        if one_tors is not None:
            self._pseudo_frag._intcos.append(one_tors)
        if one_tors2 is not None:
            self._pseudo_frag._intcos.append(one_tors2)
        if one_tors3 is not None:
            self._pseudo_frag._intcos.append(one_tors3)

    def __str__(self):

        s =  "\tFragment %s\n" %  self._A_lbl
        for i,r in enumerate(self._Arefs):
            s += "\t\tReference point %d\n" % (i+1)
            s += r.__str__()
        s += "\n\tFragment %s\n" %  self._B_lbl
        for i,r in enumerate(self._Brefs):
            s += "\t\tReference point %d\n" % (i+1)
            s += r.__str__()

        s += self._pseudo_frag.__str__()
        return s

    @property
    def nArefs(self):  # number of reference points
        return len(self._Arefs)

    @property
    def nBrefs(self):
        return len(self._Brefs)

    def D_on(self, i):
        return self._D_on[i]

    def set_ref_geom(self, ArefGeom, BrefGeom): # for debugging
        self._pseudo_frag._geom[:] = 0.0
        for i, row in enumerate(ArefGeom):
            self._pseudo_frag._geom[2-i][:] = row
        for i, row in enumerate(BrefGeom):
            self._pseudo_frag._geom[3+i][:] = row
        return

    def q(self):
        return np.array( [i.q(self._pseudo_frag._geom) for i in self._pseudo_frag.intcos] )

    def qShow(self):
        return np.array( [i.qShow(self._pseudo_frag._geom) for i in self._pseudo_frag.intcos] )

    def update_reference_geometry(self, Ageom, Bgeom):
        self._pseudo_frag._geom[:] = 0.0
        for i, rp in enumerate(self._Arefs):  # First reference atom goes in 3rd row!
            for w in rp:
                self._pseudo_frag._geom[2-i][:] += w.weight * Ageom[w.atom]
        for i, rp in enumerate(self._Brefs):
            for w in rp:
                self._pseudo_frag._geom[3+i][:] += w.weight * Bgeom[w.atom]
        return

    def ARefGeom(self):
        x = np.zeros( (self.nArefs,3) )
        x[:] = self._pseudo_frag._geom[self.nArefs-1::-1]
        return x

    def BRefGeom(self):
        x = np.zeros( (self.nBrefs,3) )
        x[:] = self._pseudo_frag._geom[3:(3+self.nBrefs)]
        return x

    def activeLabels(self):
        lbls = []
        # to add later
        #  if (inter_frag->coords.simples[0]->is_inverse_stre()): #    lbl[0] += "1/R_AB"
        #  if (inter_frag->coords.simples[i]->is_frozen()) lbl[i] = "*";
        if self.D_on(0): lbls.append("R_AB")
        if self.D_on(1): lbls.append("theta_A")
        if self.D_on(2): lbls.append("theta_B")
        if self.D_on(3): lbls.append("tau")
        if self.D_on(4): lbls.append("phi_A")
        if self.D_on(5): lbls.append("phi_B")
        return lbls

    @property
    def Ncoord(self):
        return len(self._pseudo_frag._intcos)


    def orient_fragment(self, Ageom_in, Bgeom_in, q_target):
        """ orient_fragment() moves the geometry of fragment B so that the
            interfragment coordinates have the given values
 
        Parameters
        ----------
        q_target : numpy array float[6]
        ------------
       
        Returns:  new geometry for B
        """
        nArefs = self.nArefs # of ref pts on A to worry about
        nBrefs = self.nBrefs # of ref pts on B to worry about

        self.update_reference_geometry(Ageom_in, Bgeom_in)
        q_orig = self.q()
        if len(q_orig) != len(q_target):
            raise OptError("Unexpected number of target interfragment coordinates")
        dq_target = q_target - q_orig

        # Assign and identify needed variables.
        cnt = 0
        active_lbls = self.activeLabels()
        if self._D_on[0]:
            R_AB    = q_target[cnt]
            cnt += 1
        if self._D_on[1]:
            theta_A = q_target[cnt]
            cnt += 1
        if self._D_on[2]:
            theta_B = q_target[cnt]
            cnt += 1
        if self._D_on[3]:
            tau     = q_target[cnt]
            cnt += 1
        if self._D_on[4]:
            phi_A   = q_target[cnt]
            cnt += 1
        if self._D_on[5]:
            phi_B   = q_target[cnt]
            cnt += 1
    
        print("\t---Interfragment coordinates between fragments %s and %s" % (self._A_lbl, self._B_lbl))
        print("\t---Internal Coordinate Step in ANG or DEG, aJ/ANG or AJ/DEG ---")
        print("\t ----------------------------------------------------------------------")
        print("\t Coordinate             Previous     Change       Target")
        print("\t ----------             --------      -----       ------")

        for i in range(self.Ncoord):
            c = self._pseudo_frag._intcos[i].qShowFactor # for printing to Angstroms/degrees
            print("\t%-20s%12.5f%13.5f%13.5f" %
                  (active_lbls[i], c * q_orig[i], c * dq_target[i], c * q_target[i]))
        
        print("\t ----------------------------------------------------------------------")
      
        # From here on, for simplicity we include 3 reference atom rows, even if we don't
        # have 3 reference atoms.  So, stick SOMETHING non-linear/non-0 in for
        #  non-specified reference atoms so zmat function works.
        ref_A = np.zeros( (3,3) )
        ref_A[0:nArefs] = self.ARefGeom()
        print("ref_A:")
        print(ref_A)
      
        if nArefs < 3:  # pad ref_A with arbitrary entries
            for xyz in range(3):
                ref_A[2,xyz] = xyz+1
        if nArefs < 2:
            for xyz in range(3):
                ref_A[1,xyz] = xyz+2

        ref_B           = np.zeros( (3,3) )
        ref_B[0:nBrefs] = self.BRefGeom()

        ref_B_final = np.zeros( (nBrefs,3) )
      
        # compute B1-B2 distance, B2-B3 distance, and B1-B2-B3 angle
        if nBrefs>1:
            R_B1B2 = v3d.dist(ref_B[0], ref_B[1])
      
        if nBrefs>2:
            R_B2B3 = v3d.dist(ref_B[1], ref_B[2])
            B_angle = v3d.angle(ref_B[0], ref_B[1], ref_B[2])
      
        # Determine target location of reference pts for B in coordinate system of A
        ref_B_final[0][:] = orient.zmatPoint(
                    ref_A[2], ref_A[1], ref_A[0], R_AB, theta_A, phi_A)
        if nBrefs>1:
            ref_B_final[1][:] = orient.zmatPoint(
                    ref_A[1], ref_A[0], ref_B_final[0], R_B1B2, theta_B, tau)
        if nBrefs>2:
            ref_B_final[2][:] = orient.zmatPoint(
                    ref_A[0], ref_B_final[0], ref_B_final[1], R_B2B3, B_angle, phi_B)
      
        print("ref_B_final target:")
        print(ref_B_final)
        # Can use to test if target reference points give correct values.
        #self.set_ref_geom(ref_A, ref_B_final)
        #print(self._pseudo_frag)
        nBatoms = len(Bgeom_in)
        Bgeom = Bgeom_in.copy()

        self.update_reference_geometry(Ageom_in, Bgeom)
        ref_B[0:nBrefs] = self.BRefGeom()
        print("initial ref_B")
        print(ref_B)

        # 1) Translate B->geom to place B1 in correct location.
        for i in range(nBatoms):
            Bgeom[i] += ref_B_final[0] - ref_B[0]
      
        # recompute B reference points
        self.update_reference_geometry(Ageom_in, Bgeom)
        ref_B[0:nBrefs] = self.BRefGeom()
        print("ref_B after positioning B1:")
        print(ref_B)
      
        # 2) Move fragment B to place reference point B2 in correct location
        if nBrefs>1:
            # Determine rotational angle and axis
            e12  = v3d.eAB(ref_B[0], ref_B[1])       # normalized B1 -> B2
            e12b = v3d.eAB(ref_B[0], ref_B_final[1]) # normalized B1 -> B2target
            B_angle = acos(v3d.dot(e12b,e12))
      
            if fabs(B_angle) > 1.0e-7:
                erot = v3d.cross(e12,e12b)
      
                # Move B to put B1 at origin
                for i in range(nBatoms):
                    Bgeom[i] -= ref_B[0]
      
                # Rotate B
                orient.rotateVector(erot, B_angle, Bgeom)
      
                # Move B back to coordinate system of A
                for i in range(nBatoms):
                    Bgeom[i] += ref_B[0]
          
                # recompute current B reference points
                self.update_reference_geometry(Ageom_in, Bgeom)
                ref_B[0:nBrefs] = self.BRefGeom()
                print("ref_B after positioning B2:");
                print(ref_B)

        # 3) Move fragment B to place reference point B3 in correct location.
        if nBrefs==3:
            # Determine rotational angle and axis
            erot = v3d.eAB(ref_B[0], ref_B[1])  # B1 -> B2 is rotation axis
      
            # Calculate B3-B1-B2-B3' torsion angle
            B_angle = v3d.tors(ref_B[2], ref_B[0], ref_B[1], ref_B_final[2])
      
            #oprintf_out("B_angle: %15.10lf\n",B_angle);
            if fabs(B_angle) > 1.0e-10:
      
                # Move B to put B2 at origin
                for i in range(nBatoms):
                    Bgeom[i] -= ref_B[1]
    
                orient.rotateVector(erot, B_angle, Bgeom)
      
                # Translate B1 back to coordinate system of A
                for i in range(nBatoms):
                    Bgeom[i] += ref_B[1]
      
                self.update_reference_geometry(Ageom_in, Bgeom)
                ref_B[0:nBrefs] = self.BRefGeom()
                print("ref_B after positioning B3:");
                print(ref_B)
      
        # Check to see if desired reference points were obtained.
        tval = 0.0
        for i in range(nBrefs):
            tval += np.dot(ref_B[i] - ref_B_final[i], ref_B[i] - ref_B_final[i])
        tval = sqrt(tval)
        print("\tDifference from target, |x_target - x_achieved| = %.2e\n" % tval)
  
        return Bgeom
    # end def orient_fragment()
      

# Construct a water dimer interfragment coordinate
Xatoms = [[0],  [1],  [2]]
Xw     = [[1.0],[1.0],[1.0]]
Yatoms = [[1],  [0],  [2]]
Yw     = [[1.0],[1.0],[1.0]]

Itest = Interfrag(0, Xatoms, 1, Yatoms, Xw, Yw, "HOH-1", "HOH-2" )


Axyz = np.array( [[  0.282040,    -0.582562,    -1.151084],
                  [  0.354794,    -0.619936,    -0.171523],
                  [ -0.461895,    -0.173714,     0.096910]] )

Bxyz = np.array( [[ -0.398071,    -1.083028,    -3.470849],
                  [ -0.107030,    -0.374364,    -2.871543],
                  [  0.330163,     0.251712,    -3.474012]] )
Axyz /= qcel.constants.bohr2angstroms
Bxyz /= qcel.constants.bohr2angstroms

Itest.update_reference_geometry(Axyz, Bxyz)
print(Itest)
# Starting coordinates
#q_tar = np.array( [ 3.356432919134464, 2.973807349773256, 2.185812335680237,
#                   -1.865265973964125, -0.000000000000000, -2.657620919244832] )
# Test to displace to these final coordinates. 
q_tar = np.array( [ 3.36, 2.97, 2.19, -1.87, 0.01, -2.66] )

Bxyz_new = Itest.orient_fragment(Axyz, Bxyz, q_tar)

Itest.update_reference_geometry(Axyz, Bxyz_new)
print(Itest)

