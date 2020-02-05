import numpy as np
from math import sqrt

from optking import frag, stre, bend, tors
from optking import v3d
from optking import intcosMisc
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

        # Normalize the weights.  It is assumed that the weights are positive.
        #norm = sqrt(sum(c*c for c in coeff))
        norm = sum(c for c in coeff)
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


class DimerFrag(object):
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
    def __init__(self, A_idx, A_atoms, B_idx, B_atoms, A_weights=None, B_weights=None, \
                 A_lbl="A", B_lbl="B"):
        self._A_lbl = A_lbl
        self._B_lbl = B_lbl
        self._A_idx = A_idx
        self._B_idx = B_idx

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
            for i in range(len(A_atoms)):
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
            for i in range(len(B_atoms)):
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
        ref_geom = np.zeros((6, 3)) # some rows may be unused
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
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            one_tors  = tors.Tors(1, 2, 3, 4)  # tau
            one_tors2 = tors.Tors(0, 1, 2, 3)  # phi_A
            self._D_on[5] = False              # NO phi_B
        elif nA == 2 and nB == 3:
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            one_tors  = tors.Tors(1, 2, 3, 4)  # tau
            self._D_on[4] = False              # NO phi_A
            one_tors3 = tors.Tors(2, 3, 4, 5)  # phi_B
        elif nA == 3 and nB == 1:
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            self._D_on[2] = False              # NO theta_B
            self._D_on[3] = False              # NO tau
            one_tors2 = tors.Tors(0, 1, 2, 3)  # phi_A
            self._D_on[5] = False              # NO phi_B
        elif nA == 1 and nB == 3:
            one_stre  = stre.Stre(2, 3)        # RAB
            self._D_on[1] = False              # NO theta_A
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            self._D_on[3] = False              # NO tau
            self._D_on[4] = False              # NO phi_A
            one_tors3 = tors.Tors(2, 3, 4, 5)  # phi_B
        elif nA == 2 and nB == 2:
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            one_tors  = tors.Tors(1, 2, 3, 4)  # tau
            self._D_on[4] = False              # NO phi_A
            self._D_on[5] = False              # NO phi_B
        elif nA == 2 and nB == 1:
            one_stre  = stre.Stre(2, 3)        # RAB
            one_bend  = bend.Bend(1, 2, 3)     # theta_A
            self._D_on[2] = False              # NO theta_B
            self._D_on[3] = False              # NO tau
            self._D_on[4] = False              # NO phi_A
            self._D_on[5] = False              # NO phi_B
        elif nA == 1 and nB == 2:
            one_stre  = stre.Stre(2, 3)        # RAB
            self._D_on[1] = False              # NO phi_A
            one_bend2 = bend.Bend(2, 3, 4)     # theta_B
            self._D_on[3] = False              # NO tau
            self._D_on[4] = False              # NO phi_A
            self._D_on[5] = False              # NO phi_B
        elif nA == 1 and nB == 1:
            one_stre  = stre.Stre(2, 3)        # RAB
            self._D_on[1] = False
            self._D_on[2] = False
            self._D_on[3] = False
            self._D_on[4] = False
            self._D_on[5] = False 
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

    @property
    def A_idx(self):
        return self._A_idx

    @property
    def B_idx(self): 
        return self._B_idx

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
        return [i.q(self._pseudo_frag._geom) for i in self._pseudo_frag.intcos]

    def qShow(self):
        return [i.qShow(self._pseudo_frag._geom) for i in self._pseudo_frag.intcos]

    def update_reference_geometry(self, Ageom, Bgeom):
        self._pseudo_frag._geom[:] = 0.0
        for i, rp in enumerate(self._Arefs):  # First reference atom goes in 3rd row!
            for w in rp:
                self._pseudo_frag._geom[2-i][:] += w.weight * Ageom[w.atom]
        for i, rp in enumerate(self._Brefs):
            for w in rp:
                self._pseudo_frag._geom[3+i][:] += w.weight * Bgeom[w.atom]
        return

    def getRefGeom(self):
        return self._pseudo_frag._geom.copy()

    def ARefGeom(self):  # returns reference atoms in order dA1, dA2, dA3
        x = np.zeros( (self.nArefs,3) )
        for i in range(self.nArefs):
           x[i] = self._pseudo_frag._geom[2-i]
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
    def Nintcos(self):
        return len(self._pseudo_frag._intcos)


    def orient_fragment(self, Ageom_in, Bgeom_in, q_target, printCoords=False):
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

        # These values are arbitrary; used to determine ref. point locations
        # below only if a fragment doesn't have 3 of them.
        R_AB, theta_A, theta_B, tau, phi_A, phi_B = 1.0, 0.8, 0.8, 0.8, 0.8, 0.8
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
    
        if printCoords:
            print("\t---DimerFrag coordinates between fragments %s and %s" % (self._A_lbl, self._B_lbl))
            print("\t---Internal Coordinate Step in ANG or DEG, aJ/ANG or AJ/DEG ---")
            print("\t ----------------------------------------------------------------------")
            print("\t Coordinate             Previous     Change       Target")
            print("\t ----------             --------      -----       ------")
    
            for i in range(self.Nintcos):
                c = self._pseudo_frag._intcos[i].qShowFactor # for printing to Angstroms/degrees
                print("\t%-20s%12.5f%13.5f%13.5f" %
                      (active_lbls[i], c * q_orig[i], c * dq_target[i], c * q_target[i]))
            
            print("\t ----------------------------------------------------------------------")
      
        # From here on, for simplicity we include 3 reference atom rows, even if we don't
        # have 3 reference atoms.  So, stick SOMETHING non-linear/non-0 in for
        #  non-specified reference atoms so zmat function works.
        ref_A = np.zeros( (3,3) )
        ref_A[0:nArefs] = self.ARefGeom()
        #print("ref_A:")
        #print(ref_A)
      
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
      
        #print("ref_B_final target:")
        #print(ref_B_final)
        # Can use to test if target reference points give correct values.
        #self.set_ref_geom(ref_A, ref_B_final)
        #print(self._pseudo_frag)
        #quit()
        nBatoms = len(Bgeom_in)
        Bgeom = Bgeom_in.copy()

        self.update_reference_geometry(Ageom_in, Bgeom)
        ref_B[0:nBrefs] = self.BRefGeom()
        #print("initial ref_B")
        #print(ref_B)

        # 1) Translate B->geom to place B1 in correct location.
        for i in range(nBatoms):
            Bgeom[i] += ref_B_final[0] - ref_B[0]
      
        # recompute B reference points
        self.update_reference_geometry(Ageom_in, Bgeom)
        ref_B[0:nBrefs] = self.BRefGeom()
        #print("ref_B after positioning B1:")
        #print(ref_B)
      
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
                #print("ref_B after positioning B2:");
                #print(ref_B)

        # 3) Move fragment B to place reference point B3 in correct location.
        if nBrefs==3:
            # Determine rotational angle and axis
            erot = v3d.eAB(ref_B[0], ref_B[1])  # B1 -> B2 is rotation axis
      
            # Calculate B3-B1-B2-B3' torsion angle
            B_angle = v3d.tors(ref_B[2], ref_B[0], ref_B[1], ref_B_final[2])
      
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
                #print("ref_B after positioning B3:");
                #print(ref_B)
      
        # Check to see if desired reference points were obtained.
        #print("last ref_B:");
        #print(ref_B)
        tval = 0.0
        for i in range(nBrefs):
            tval += np.dot(ref_B[i] - ref_B_final[i], ref_B[i] - ref_B_final[i])
        tval = sqrt(tval)
        #print("orient_fragment: |x_target - x_achieved| = %.2e" % tval)
  
        return Bgeom
    # end def orient_fragment()


    def compute_B(self, A_geom, B_geom, Bmat_in, A_xyz_off=None, B_xyz_off=None):
        """ This function adds interfragment rows into an existing B matrix.
            B is (internals, Cartesians).  Often, 6 x 3*(Natoms).
        Parameters
        ----------
        A_geom : numpy array
            geometry of fragment A, array is (A atoms,3)
        B_geom : numpy array that is (B atoms,3)
            geometry of fragment B, array is (B atoms,3)
        Bmat_int : numpy array
            provided B matrix
        intco_off : int
            index of first row of Bmatrix to start writing the interfragment rows.
        A_off : int
            Column of B matrix at which the cartesian coordinates of atoms in fragment A begin.
            Needed since columns may span full molecular system.
        B_off : int
            Column of B matrix at which the cartesian coordinates of atoms in fragment B begin.
        If A_off and B_off are not given, then the minimal (dimer-only) B-matrix is returned.
        """
        print('compute_B:')
        print('A_geom:')
        print(A_geom)
        print('B_geom:')
        print(B_geom)
        print('Bmat in:')
        print(Bmat_in)
        print("A_xyz_off: %d" % A_xyz_off)
        print("B_xyz_off: %d" % B_xyz_off)
        print(Bmat_in)
    
        NatomA = len(A_geom)
        NatomB = len(B_geom)
        Ncart = 3*(NatomA+NatomB)

        if A_xyz_off == None:
            A_xyz_off = 0
        if B_xyz_off == None:
            B_xyz_off = 3*NatomA

        self.update_reference_geometry(A_geom, B_geom);
       
        # Compute B-matrix for reference points
        # Since the numbering of the atoms in the dimer coordinates (e.g. R(3,4)
        # is canonical, the reference-point B matrix always has 6*3=18 columns.
        B_ref = np.zeros( (self.Nintcos, 18) )
        for i, intco in enumerate(self._pseudo_frag._intcos):
            intco.DqDx( self.getRefGeom(), B_ref[i])
        print("Reference point B matrix:")
        print(B_ref)

        ## B_ref is derivative of interfragment D wrt reference point position
        cnt = 0

        if self.D_on(0):
            rf = [3*i for i in self._pseudo_frag._intcos[cnt].atoms]
            for xyz in range(3):
                # Add contributions to each atom included in reference pt definition. 
                for el in self._Arefs[0]:
                    Bmat_in[cnt,A_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[0]+xyz]
                for el in self._Brefs[0]:
                    Bmat_in[cnt,B_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[1]+xyz]
            cnt += 1
        
        if self.D_on(1):
            rf = [3*i for i in self._pseudo_frag._intcos[cnt].atoms]
            for xyz in range(3):
                for el in self._Arefs[1]:
                    Bmat_in[cnt,A_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[0]+xyz]
                for el in self._Arefs[0]:
                    Bmat_in[cnt,A_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[1]+xyz]
                for el in self._Brefs[0]:
                    Bmat_in[cnt,B_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[2]+xyz]
            cnt += 1
      
        if self.D_on(2):
            rf = [3*i for i in self._pseudo_frag._intcos[cnt].atoms]
            for xyz in range(3):
                for el in self._Arefs[0]:
                    Bmat_in[cnt,A_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[0]+xyz]
                for el in self._Brefs[0]:
                    Bmat_in[cnt,B_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[1]+xyz]
                for el in self._Brefs[1]:
                    Bmat_in[cnt,B_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[2]+xyz]
            cnt += 1
      
        if self.D_on(3):
            rf = [3*i for i in self._pseudo_frag._intcos[cnt].atoms]
            for xyz in range(3):
                for el in self._Arefs[1]:
                    Bmat_in[cnt,A_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[0]+xyz]
                for el in self._Arefs[0]:
                    Bmat_in[cnt,A_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[1]+xyz]
                for el in self._Brefs[0]:
                    Bmat_in[cnt,B_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[2]+xyz]
                for el in self._Brefs[1]:
                    Bmat_in[cnt,B_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[3]+xyz]
            cnt += 1
      
        if self.D_on(4):
            rf = [3*i for i in self._pseudo_frag._intcos[cnt].atoms]
            for xyz in range(3):
                for el in self._Arefs[2]:
                    Bmat_in[cnt,A_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[0]+xyz]
                for el in self._Arefs[1]:
                    Bmat_in[cnt,A_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[1]+xyz]
                for el in self._Arefs[0]:
                    Bmat_in[cnt,A_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[2]+xyz]
                for el in self._Brefs[0]:
                    Bmat_in[cnt,B_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[3]+xyz]
            cnt += 1
      
        if self.D_on(5):
            rf = [3*i for i in self._pseudo_frag._intcos[cnt].atoms]
            for xyz in range(3):
                for el in self._Arefs[0]:
                    Bmat_in[cnt,A_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[0]+xyz]
                for el in self._Brefs[0]:
                    Bmat_in[cnt,B_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[1]+xyz]
                for el in self._Brefs[1]:
                    Bmat_in[cnt,B_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[2]+xyz]
                for el in self._Brefs[2]:
                    Bmat_in[cnt,B_xyz_off + 3*el.atom + xyz] += el.weight * B_ref[cnt,rf[3]+xyz]
            cnt += 1

    def test_B(self, Axyz, Bxyz, printInfo=False):
        print("\tTesting B matrix")
        DISP_SIZE = 0.001
        MAX_ERROR = 1.0e-8
        NA = len(Axyz)
        Natoms = NA+len(Bxyz)
        B_analytic = np.zeros( (self.Nintcos, 3*Natoms) )
        Itest.compute_B(Axyz, Bxyz, B_analytic)
        if printInfo:
            print("Final B matrix")
            print(Bmat_in)
                
        B_fd = np.zeros((self.Nintcos, 3*Natoms))
        coord = np.concatenate( (Axyz,Bxyz) ).copy()
        #intcosMisc.updateDihedralOrientations(self._pseudo_frag._intcos, coord)
        #intcosMisc.fixBendAxes(self._pseudo_frag._intcos, coord)
                
        for atom in range(Natoms):
            for xyz in range(3):
                coord[atom, xyz] -= DISP_SIZE
                self.update_reference_geometry(coord[:NA], coord[NA:])
                q_m  = Itest.q()
                coord[atom, xyz] -= DISP_SIZE
                self.update_reference_geometry(coord[:NA], coord[NA:])
                q_m2 = Itest.q()
                coord[atom, xyz] += 3 * DISP_SIZE
                self.update_reference_geometry(coord[:NA], coord[NA:])
                q_p  = Itest.q()
                coord[atom, xyz] += DISP_SIZE
                self.update_reference_geometry(coord[:NA], coord[NA:])
                q_p2 = Itest.q()
                coord[atom, xyz] -= 2 * DISP_SIZE  # restore to original
                for i in range(self.Nintcos):
                    B_fd[i, 3 * atom + xyz] = (
                        q_m2[i] - 8 * q_m[i] + 8 * q_p[i] - q_p2[i]) / (12.0 * DISP_SIZE)
        
        if printInfo:
            print("Numerical B matrix in au, DISP_SIZE = %lf\n" % DISP_SIZE)
            print(B_fd)

        max_error = -1.0
        max_error_intco = -1
        for i in range(self.Nintcos):
            for j in range(3*Natoms):
                if fabs(B_analytic[i, j] - B_fd[i, j]) > max_error:
                    max_error = fabs(B_analytic[i][j] - B_fd[i][j])
                    max_error_intco = i
    
        print("\t\tMaximum difference is %.1e for internal coordinate %d." %
                  (max_error, max_error_intco + 1))
        print("\t\tThis coordinate is %s" % str(self._pseudo_frag._intcos[max_error_intco]))
    
        if max_error > MAX_ERROR:
            print("\tB-matrix could be in error. However, numerical tests may fail for\n"
                           + "\ttorsions at 180 degrees, and slightly for linear bond angles."
                           + "This is OK.\n")
            return False
        else:
            print("\t...Passed.")
            return True

    
def test_orient(NA, NB, printInfo=False) :
    """ Test the orient_fragment function to see if pre-determined target
    # coordinate values can be met.  Technically, this only tests consistency
    # within the class, i.e., whether computed values of the interfragment
    # coordinates match the target ones.  The point of this function is also
    # to ensure the code is robust for fragments with fewer than 3 reference atoms
    # (such as atoms, diatomics, linear molecules).
    #  - Chooses random geometries.
    #  - Chooses 1,2,or 3 random atoms to define reference points.
    #  Does not test a linear polyatomic.  Appears to fail at present only when
    # interior angles of torsions are too large or small (here by luck).
    """
    from random import random,shuffle,choice,sample
    NAref = min(NA,3)
    NBref = min(NB,3)
    Albl = "A-%d-atoms" % NA
    Blbl = "B-%d-atoms" % NB

    atom_list = list(range(NA))
    Aatoms = []
    Aweights = []
    n = 0
    while n < NAref:
        ref_length = min(NAref, choice([1,2,3])) # of atoms used to define ref. pt.
        l = sample(atom_list, ref_length)  # select random atoms in A
        l.sort()
        if l in Aatoms:
            continue
        n += 1
        Aatoms.append(l)
        Aweights.append( ref_length*[1.0] )
    if printInfo: print(Aatoms)
    if printInfo: print(Aweights)

    atom_list = list(range(NB))
    Batoms = []
    Bweights = []
    n = 0
    while n < NBref:
        ref_length = min(NBref, choice([1,2,3]))
        l = sample(atom_list, ref_length)
        l.sort()
        if l in Batoms:
            continue
        n += 1
        Batoms.append(l)
        Bweights.append( ref_length*[1.0])
    if printInfo: print(Batoms)
    if printInfo: print(Bweights)

    Itest = DimerFrag(0, Aatoms, 1, Batoms, Aweights, Bweights, Albl, Blbl)

    Axyz = np.zeros( (NA,3) )
    for i in range(NA): # insert random geometries
        Axyz[i][:] = 5.0+3.0*random(), 3.0*random(), 3.0*random()
    Bxyz = np.zeros( (NB,3) )
    for i in range(NB):
        Bxyz[i][:] = 3.0*random(), 3.0*random(), 3.0*random()
    Itest.update_reference_geometry(Axyz, Bxyz)
    #print(Itest)
    q_tar = Itest.Nintcos*[0.8]
    Bxyz_new = Itest.orient_fragment(Axyz, Bxyz, q_tar)
    Itest.update_reference_geometry(Axyz, Bxyz_new)
    print('Error in positioning dimer (%s/%s): %8.3e' %
         ( Albl, Blbl, sqrt(sum(((q_tar - Itest.q())**2)))))

# Test orient_fragment function on random molecules of various sizes
# for i in range(1,4):
#    for j in range(1,4):
#        test_orient(i,j)
## Example of how to use for water dimer
#Xatoms = [[0,    1],[1,     2],[2]]
#Xw     = [[1.0,0.1],[1.0, 0.1],[1.0]]
#Yatoms = [[1,  0],  [0],       [2, 1]]
#Yw     = [[1.0,0.1],[1.0],     [1.0, 0.1]]
#Itest = DimerFrag(0, Xatoms, 1, Yatoms, Xw, Yw, "HOH-1", "HOH-2" )
#Axyz = np.array( [[ 0.53297836, -1.10088263, -2.17523351],
#                  [ 0.67046349, -1.17150926, -0.32413149],
#                  [-0.87285505, -0.32827188,  0.18313336]], float )
#Bxyz = np.array( [[-0.75224517, -2.04662631, -6.55895403],
#                  [-0.20225739, -0.70744543, -5.42642983],
#                  [ 0.62391765,  0.47566674, -6.56493124]], float )
#
## Displace to target coordinates
#q_tar = [3.36,2.97,2.19,-1.66,0.01,-2.74]
#Bxyz_new = Itest.orient_fragment(Axyz, Bxyz, q_tar)
#Itest.update_reference_geometry(Axyz, Bxyz_new)
#print(Itest)
#q = Itest.q()
#print('Error in positioning water dimer: %8.3e'%sqrt(sum((q_tar - q)**2)))
#Itest.test_B(Axyz,Bxyz_new)
#
