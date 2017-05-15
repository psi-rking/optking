import numpy as np
import frag
from optParams import Params
from addIntcos import connectivityFromDistances
from printTools import printMat
import physconst as pc
import covRadii
import v3d

class MOLSYS():
    def __init__(self, fragments, fb_fragments=None, intcos=None):
        # ordinary fragments with internal structure
        self._fragments = []
        if fragments:
            self._fragments = fragments
        # fixed body fragments defined by Euler/rotation angles
        self._fb_fragments = []
        if fb_fragments:
            self._fb_fragments = fb_fragments

    def __str__(self):
        s = ''
        for iF, F in enumerate(self._fragments):
            s += "Fragment %d\n" % (iF + 1)
            s += F.__str__()
        for iB, B in enumerate(self._fb_fragments):
            s += "Fixed boxy Fragment %d\n" % (iB + 1)
            s += B.__str__()
        return s

    @classmethod
    def fromPsi4Molecule(cls, mol):

        NF = mol.nfragments()
        print "\t%d Fragments in PSI4 molecule object." % NF
        frags = []

        for iF in range(NF):
            fragMol = mol.extract_subsets(iF+1)

            fragNatom = fragMol.natom()
            print "\tCreating fragment %d with %d atoms" % (iF+1,fragNatom)

            fragGeom = np.zeros( (fragNatom,3), float)
            fragGeom[:] = fragMol.geometry()
         
            fragZ = np.zeros( fragNatom, int)
            for i in range(fragNatom):
                fragZ[i] = fragMol.Z(i)
         
            fragMasses = np.zeros( fragNatom, float)
            for i in range(fragNatom):
                fragMasses[i] = fragMol.mass(i)

            frags.append( frag.FRAG(fragZ, fragGeom, fragMasses) )
        return cls( frags )

    @property
    def Natom(self):
        s = 0
        for F in self._fragments:
            s += F.Natom 
        return s

    @property
    def Nfragments(self):
        return len(self._fragments) + len(self._fb_fragments)

    # Return overall index of first atom in fragment, beginning 0,1,...
    def frag_1st_atom(self, iF):
        if iF >= len(self._fragments):
            return ValueError()
        start = 0
        for i in range(0,iF):
            start += self._fragments[i].Natom
        return start

    @property
    def geom(self):
        geom = np.zeros( (self.Natom,3), float)
        for iF, F in enumerate(self._fragments):
            row   = self.frag_1st_atom(iF)
            geom[row:(row+F.Natom),:] = F.geom
        return geom

    @property
    def masses(self):
        m = np.zeros( self.Natom, float )
        for iF, F in enumerate(self._fragments):
            start = self.frag_1st_atom(iF)
            m[start:(start+F.Natom)] = F.masses
        return m

    @property
    def Z(self):
        z = np.zeros( self.Natom, float )
        for iF, F in enumerate(self._fragments):
            start = self.frag_1st_atom(iF)
            z[start:(start+F.Natom)] = F.Z
        return z

    @property
    def intcos(self):
        _intcos = []
        for F in self._fragments:
           _intcos += F.intcos
        return _intcos

    def printIntcos(self):
        for iF, F in enumerate(self._fragments):
            print "Fragment %d" % (iF+1)
            F.printIntcos()
        return

    def addIntcosFromConnectivity(self, C=None):
        for F in self._fragments:
            if C == None:
                C = F.connectivityFromDistances()
            F.addIntcosFromConnectivity(C)

    def addCartesianIntcos(self):
        for F in self._fragments:
            addIntcos.addCartesianIntcos(F._intcos, F._geom)

    def printGeom(self):
        for iF, F in enumerate(self._fragments):
            print "Fragment %d" % (iF+1)
            print F

    def showGeom(self):
        for iF, F in enumerate(self._fragments):
            print "Fragment %d" % (iF+1)
            print F

    def consolidateFragments(self):
        if self.Nfragments == 1:
            return
        print "Consolidating multiple fragments into one for optimization."
        consolidatedFrag = frag.FRAG(self.Z, self.geom, self.masses)
        del self._fragments[:]
        self._fragments.append( consolidatedFrag )

    # Split any fragment not connected by bond connectivity.
    def splitFragmentsByConnectivity(self):
        tempZ      = np.copy(self.Z)
        tempGeom   = np.copy(self.geom)
        tempMasses = np.copy(self.masses)

        newFragments = []
        for F in self._fragments:
            C = connectivityFromDistances(F.geom, F.Z)
            atomsToAllocate = list(reversed(range(F.Natom)))

            while atomsToAllocate:
                frag_atoms = [ atomsToAllocate.pop() ]

                more_found = True
                while more_found:
                    more_found = False
                    addAtoms = []
                    for A in frag_atoms:
                        for B in atomsToAllocate:
                            if C[A, B]: 
                                addAtoms.append(B)
                                more_found = True

                    for a in addAtoms:
                        frag_atoms.append(a)
                        atomsToAllocate.remove(a)

                frag_atoms.sort()
                subNatom = len(frag_atoms)
                subZ      = np.zeros( (subNatom), float)
                subGeom   = np.zeros( (subNatom,3), float)
                subMasses = np.zeros( (subNatom), float)
                for i, I in enumerate(frag_atoms):
                    subZ[i]        = tempZ[I]
                    subGeom[i,0:3] = tempGeom[I,0:3]
                    subMasses[i]   = tempMasses[I]
                newFragments.append( frag.FRAG(subZ, subGeom, subMasses) )

        del self._fragments[:]
        self._fragments = newFragments

    # bond separated fragments
    def augmentConnectivityToSingleFragment(self, C):
        if self.Nfragments:
            return
        print '\tAugmenting connectivity matrix to join fragments.'
        fragAtoms = []
        for iF, F in enumerate(fragments):
            fragAtoms.append(range(self.frag_1st_atom(iF), self.frag_1st_atom(iF) + F.Natom))

        # which fragments are connected?
        nF = self.Nfragments
        frag_connectivity = np.array( (nF,nF), bool)
        for iF in range(nF):
          frag_connectivity[iF][iF] = True

        Z = self.Z

        scale_dist = Params.interfragment_connect
        all_connected = False
        while not all_connected:
            for f2 in range(nF):
              for f1 in range(f2):
                  if frag_connectivity[f1][f2]:
                      continue # already connected
                  minVal = 1.0e12
      
                  # Find closest 2 atoms between fragments
                  for f1_atom in fragAtoms[f1]:
                    for f2_atom in fragAtoms[f2]:
                      tval = v3d.dist(geom[fragAtoms[f1][f1_atom]], geom[fragAtoms[f2][f2_atom]])
                      if tval < minVal:
                        minVal = tval
                        i = fragAtoms[f1][f1_atom]
                        j = fragAtoms[f2][f2_atom]
        
                  Rij = v3d.dist(geom[i], geom[j])
                  R_i = covRadii.R[ int(Z[i]) ] / pc.bohr2angstroms
                  R_j = covRadii.R[ int(Z[j]) ] / pc.bohr2angstroms
                  if (Rij > scale_dist * (R_i + R_j)):
                    continue  # ignore this as too far - for starters
      
                  print "\tConnecting fragments with atoms %d and %d" % (i+1, j+1)
                  C[i][j] = C[j][i] = True
                  frag_connectivity[f1][f2] = frag_connectivity[f2][f1] = True
                  # Now check for possibly symmetry-related atoms which are just as close
                  # We need them all to avoid symmetry breaking.
                  for f1_atom in fragAtoms[f1]:
                    for f2_atom in fragAtoms[f2]:
                      tval = v3d.dist(geom[fragAtoms[f1][f1_atom]], geom[fragAtoms[f2][f2_atom]])
                      if (fabs(tval - minVal) < 1.0e-14): 
                        i = fragAtoms[f1][f1_atom]
                        j = fragAtoms[f2][f2_atom]
                        print "\tConnecting fragments with atoms %d and %d" % (i+1, j+1)
                        C[i][j] = C[j][i] = True

            all_connected = True
"""
                  # Test whether all frags are connected using current distance threshold
                  set_label = np.zeros( nF, bool)
                  for i in range(nF):
                    set_label[i] = i
            
                  for i in range(nF):
                    for j in range(nF):
                      if frag_connectivity[i][j]:
                        ii = set_label[i]
                        jj = set_label[j]
                        if ii > jj:
                          set_label[i] = jj
                        else if jj > ii:
                          set_label[j] = ii

                  all_connected = True
                  for i in range(1,nF):
                    if set_label[i] != 0:
                      all_connected = False

            if not all_connected:
                scale_dist += 0.4
                print "\tIncreasing scaling to %6.3f to connect fragments." % (scale_dist)

"""

