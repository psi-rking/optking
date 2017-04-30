import numpy as np
#import misc
#import addIntcos
#import physconst
import frag

class MOLSYS():
    def __init__(self, fragments, fb_fragments=None, intcos=None):
        # ordinary fragments with internal structure
        self.fragments = []
        if fragments:
            self._fragments = list( fragments)
        # fixed body fragments defined by Euler/rotation angles
        self._fb_fragments = []
        if fb_fragments:
            self._fb_fragments = list( fb_fragments)

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
            print "\tCreating fragment %d" % (iF+1)
            fragMol = mol.extract_subsets(iF+1)
            fragGeom = np.array( fragMol.geometry() )
            fragNatom = fragMol.natom()
         
            fragZ = np.zeros( fragNatom, int)
            for i in range(fragNatom):
                fragZ[i] = fragMol.Z(i)
         
            fragMasses = np.zeros( fragNatom, float)
            for i in range(fragNatom):
                fragMasses[i] = fragMol.mass(i)

            frags.append( frag.FRAG(fragZ, fragGeom, fragMasses) )
        cls( frags )

    @property
    def Natom(self):
        s = 0
        for F in self._fragments:
            s += F.Natom 
        return s

    @property
    def Nfragments(self):
        return len(self._fragments) + len(self._fb_fragments)

    def frag_first_atom(self, iF):
        i = 0
        for F in self._fragments[0:iF]:
            i += F.Natom
        return i

    @property
    def geom(self):
        geom = np.zeros( self.Natom, float)
        for iF, F in enumerate(self._fragments):
            geom[self.frag_first_atom(iF),:] = F.geom

    @property
    def masses(self):
        m = []
        for F in self.fragments:
            m += F.masses 
        return m

    @property
    def intcos(self):
        _intcos = []
        for F in self._fragments:
           _intcos += F.intcos
        return _intcos

    def printIntcos(self):
        for iF, F in enumerate(fragments):
            print "Fragment %d" % (iF)
            print F.printIntcos()

    def addIntcosFromConnectivity(self):
        for F in fragments:
            C = addIntcos.connectivityFromDistances(F._geom, F._Z)
            addIntcos.addIntcosFromConnectivity(C, F._intcos, F._geom)

    def addCartesianIntcos(self):
        for F in self._fragments:
            addIntcos.addCartesianIntcos(F._intcos, F._geom)

    def printGeom(self):
        for iF, F in enumerate(fragments):
            print "Fragment %d" % (iF+1)
            print F

    def showGeom(self):
        for iF, F in enumerate(fragments):
            print "Fragment %d" % (iF+1)
            print F

