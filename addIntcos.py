from itertools import combinations,permutations
import numpy as np
import sys
import optParams as op

import physconst as pc
from intcosMisc import qValues
import molsys
import frag
import simple
import v3d
import covRadii
import stre
import bend
import tors
import cart
from printTools import printMat
from printTools import printArray

# returns connectivity matrix.  Matrix is 0 if i==j.
def connectivityFromDistances(geom, Z):
    C = np.zeros( (len(geom), len(geom)), bool )

    for i,j in combinations( range(len(geom)), 2):
        R = v3d.dist(geom[i], geom[j])
        Rcov = (covRadii.R[int(Z[i])] + covRadii.R[int(Z[j])])/pc.bohr2angstroms
        if R < op.Params.covalent_connect * Rcov:
            C[i,j] = C[j,i] = True

    return C

def addIntcosFromConnectivity(C, intcos, geom):
    addStreFromConnectivity(C, intcos)
    addBendFromConnectivity(C, intcos, geom)
    addTorsFromConnectivity(C, intcos, geom)
    return

# Add Stretches from connectivity.  Return number added.
def addStreFromConnectivity(C, intcos):
    Norig = len(intcos)
    for i,j in combinations( range(len(C)), 2):
        if C[i,j]:
            s = stre.STRE(i, j)
            if s not in intcos:
                intcos.append(s)
    return len(intcos) - Norig  # return number added

# Add Bends from connectivity.  Return number added.
def addBendFromConnectivity(C, intcos, geom):
    Norig = len(intcos)
    Natom = len(geom)
    for i,j in permutations( range(Natom), 2):
        if C[i,j]:
            for k in range(i+1,Natom):  # make i<k; the constructor checks too
                if C[j,k]:
                    (check, val) = v3d.angle(geom[i], geom[j], geom[k])
                    if not check: continue
                    if val < op.Params.linear_bend_threshold:
                        b = bend.BEND(i, j, k)
                        if b not in intcos:
                            intcos.append(b)
                    else:     # linear angle
                        b = bend.BEND(i, j, k, bendType="LINEAR")
                        if b not in intcos:
                            intcos.append(b)
        
                        b2 = bend.BEND(i, j, k, bendType="COMPLEMENT")
                        if b2 not in intcos:
                            intcos.append(b2)

    return len(intcos) - Norig

# Add torsions for all bonds present; return number added.
# Use prior existence of linear bends to determine linearity in this function.
def addTorsFromConnectivity(C, intcos, geom):
    Norig = len(intcos)
    Natom = len(geom)

    # Find i-j-k-l where i-j-k && j-k-l are NOT collinear.
    for i,j in permutations( range(Natom), 2):
        if C[i,j]:
            for k in range(Natom):
                if C[k,j] and k!=i:
      
                  # ensure i-j-k is not collinear; that a regular such bend exists
                  b = bend.BEND(i,j,k)
                  if b not in intcos:
                      continue
      
                  for l in range(i+1,Natom):
                      if C[l,k] and l!=j:
        
                          # ensure j-k-l is not collinear
                          b = bend.BEND(j,k,l)
                          if b not in intcos:
                              continue
          
                          t = tors.TORS(i,j,k,l)
                          if t not in intcos:
                              intcos.append(t)
      
    # Search for additional torsions around collinear segments.
    # Find collinear fragment j-m-k
    for j,m in permutations( range(Natom), 2):
       if C[j,m]:
          for k in range(j+1,Natom):
             if C[k,m]:
                # ignore if regular bend
                b = bend.BEND(j,m,k)
                if b in intcos:
                   continue
    
                # Found unique, collinear j-m-k
                # Count atoms bonded to m.
                nbonds = sum(C[m])
    
                if nbonds == 2: # Nothing else is bonded to m
    
                   # look for an 'I' for I-J-[m]-k-L such that I-J-K is not collinear
                   J = j
                   i = 0
                   while i < Natom:
                      if C[i,J] and i!=m:  # i!=J i!=m
                         b = bend.BEND(i,J,k,bendType='LINEAR')
                         if b in intcos:   # i,J,k is collinear
                            J = i
                            i = 0
                            continue
                         else:             # have I-J-[m]-k. Look for L.
                            I = i
                            K = k
                            l = 0
                            while l < Natom:
                               if C[l,K] and l!=m and l!=j and l!=i:
                                  b = bend.BEND(l,K,J,bendType='LINEAR')
                                  if b in intcos: # J-K-l is collinear
                                     K = l
                                     l = 0
                                     continue
                                  else: # Have found I-J-K-L.
                                     L = l
                                     check,val = v3d.tors(geom[I], geom[J], geom[K], geom[L])
                                     if check:
                                        t = tors.TORS(I,J,K,L)
                                        if t not in intcos:
                                           intcos.append(t)
                               l = l+1
                      i = i+1
    return len(intcos) - Norig


def addCartesianIntcos(intcos, geom):
    Norig = len(intcos)
    Natom = len(geom)

    for i in range(Natom):
        intcos.append( cart.CART(i, 'X') )
        intcos.append( cart.CART(i, 'Y') )
        intcos.append( cart.CART(i, 'Z') )

    return len(intcos) - Norig


# Identify linear angles and add them if necessary.
def linearBendCheck(intcos, geom, dq):
    linearBends = []

    # This will need generalized later for combination coordinates.
    q = qValues(intcos, geom)

    for i, intco in enumerate(intcos):
        if isinstance(intco, bend.BEND):
            newVal = intco.q(geom) + dq[i]
            A = intco.A
            B = intco.B
            C = intco.C

            # <ABC < 0.  A-C-B should be linear bends.
            if newVal < 0.0:
                linearBends.append( bend.BEND(A,C,B,bendType="LINEAR") )
                linearBends.append( bend.BEND(A,C,B,bendType="COMPLEMENT") )

            # <ABC~pi. Add A-B-C linear bends.
            elif newVal > op.Params.linear_bend_threshold:
                linearBends.append( bend.BEND(A,B,C,bendType="LINEAR") )
                linearBends.append( bend.BEND(A,B,C,bendType="COMPLEMENT") )

    linearBendsMissing = []
    for ib, b in enumerate(linearBends):
        if ib == 0:
            print "\tThe following linear bends should be present."
        print b

        if b in intcos:
            print ": already present."
        else:
            print ": missing."
            linearBendsMissing.append(b)

    return linearBendsMissing
####
## params: string of integers corresponding to internal coordinates
## returns: a list of integers correspoding to an atom
## removes spaces or non integer characters from string of internal coordinates to be frozen        
#### 
def parseFrozenString(frozenStr):
#    frozenStr = int(frozenStr)
    parsedList = frozenStr.encode('utf-8')
    parsedList = str.replace(parsedList, '(' , '')
    parsedList = str.replace(parsedList, ')' , '')
    parsedList = parsedList.split() #this is where string actually becomes a List
    for i in range (0, len(parsedList)):
        parsedList[i] = int(parsedList[i])
#    for i in range(0, len(parsedString)):
#        parsedString[i]= int(parsedString[i])
#    print (parsedString)
    return parsedList	
#####
## params: List of integers correspoding to atoms of distance to be frozen
##	       list of internal coordinates
## calls frozen_setter for stretches, bends, or dihedrals
####
## To-Do add exceptions and exception handling
def markDisAsFrozen(frozenIntcosList, Molsys, intcos):
    if(len(frozenIntcosList) % 2 == 0):
        for i in range (0, len(frozenIntcosList), 2):

#determine fragment of atom 1
#determine fragment of atom 2
#if different - raise exception for now.
#if same
#molsys._fragment[iF].intcos.append( )

            stretch = stre.STRE(frozenIntcosList[i] - 1, frozenIntcosList[i + 1] - 1)
            try:
                indexing = intcos.index(stretch)
                intcos[indexing].frozen = True
            except ValueError:
                checkAtomIndex(frozenIntcosList, stretch, i + 1, Molsys)
                checkFragment(frozenIntcosList, bendFroz, i, Molsys, identifier) 
                intcos.append(stretch)
                intcos[-1].frozen = True 
    else:
        print("Frozen Distance did not contain a suitable number of atoms, please freeze the distance in sets of two atoms. No distances frozen")
    return
#####
## params: List of integers correspoding to atoms of bend to be frozen
##	       list of internal coordinates
## calls frozen_setter for stretches, bends, or dihedrals
####
## To-Do add exceptions and exception handling
def markBendAsFrozen(frozenIntcosList, Molsys, intcos):
    if len(frozenIntcosList) % 3 == 0:
        for i in range (0, len(frozenIntcosList), 3):
            bendFroz = bend.BEND(frozenIntcosList[i] - 1, frozenIntcosList[i + 1] - 1, frozenIntcosList[i + 2] - 1)
            try:
                indexing = intcos.index(bendFroz)
                intcos[indexing].frozen = True 
            except ValueError: 
                checkAtomIndex(frozenIntcosList, bendFroz, i + 2, Molsys)
                identifier = "bendFroz"
                checkFragment(frozenIntcosList, bendFroz, i, Molsys, identifier)
                Molsys._fragments[atom2gragment_index(i)]._intcos.append(bendFroz)
    else:	
        print ("Frozen Bend did not contain suitable number of atoms, please freeze Bends in sets of three atoms. No Bends frozen")
	return
 
#####
## params: List of integers correspoding to atoms of dihedral to be frozen
##	       list of internal coordinates
## calls frozen_setter for stretches, bends, or dihedrals
####
## To-Do add exceptions and exception handling
def markTorsAsFrozen(frozenIntcosList, Molsys, intcos):
    if len(frozenIntcosList) % 4 == 0:
        for i in range (0, len(frozenIntcosList), 4):
            torsAngle = tors.TORS(frozenIntcosList[i] - 1, frozenIntcosList[i + 1] - 1, frozenIntcosList[i + 2] - 1, frozenIntcosList[i + 3] - 1)
            try:
                indexing = intcos.index(torsAngle)
                intcos[indexing].frozen = True 
            except ValueError:
                checkAtomIndex(frozenIntcosList, torsAngle, i+3, Molsys)
                identifier = "torsAngle"
                checkFragment(froenIntcosList, torsAngle, i, Molsys, identifier)
                Molsys._fragments[0]._intcos.append(torsAngle)
    else:
        print ("Frozen Dihedral angle does not contain suitable number of atoms, please freeze dihedrals in sets of four atoms. No Dihedrals frozen")
	return

def checkAtomIndex(frozenIntcosList, intcosType, index, Molsys):
    try:
        if frozenIntcosList[index] > Molsys.Natom:
            print ("=" * 90)
            print (intcosType),
            print ("contains atoms that do not exist in the molecule provided")
            print ("=" * 90)
            raise ValueError ("Qutting Program")
        else:
            return
    except ValueError as e:
        sys.exit(1)

def checkFragment(frozenIntcosList, intcosType, index, Molsys, identifier):
    fragmentIndex = MOlsys.atom2frag_index(i)
    fragmentRange = Molsys.frag_atom_range(fragmentIndex)
    try:
        if (identifier == "torsAngle"):
            lastIndex = i + 3
        elif (identifier == "bendFroz"):
            lastIndex = i + 2
        elif (identifier == "stretch"):
            lastIndex = i + 1

        if frozenIntcosList[lastIndex] > i + fragmentRange:
            print (intcosType),
            print ("contains atoms that are not in the same fragment"),
            print ("Optking cannot currently freeze interfragment coordinates")
            print ("Qutting program")
            raise ValueError
        else:
            return
    except ValueError as e:
            sys.exit(1)         
