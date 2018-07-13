from itertools import combinations, permutations

import numpy as np

import bend
import cart
import covRadii
import optExceptions
import optparams as op
import physconst as pc
import stre
import tors
import v3d
from intcosMisc import qValues
from printTools import print_opt


# returns connectivity matrix.  Matrix is 0 if i==j.
def connectivityFromDistances(geom, Z):
    C = np.zeros((len(geom), len(geom)), bool)

    for i, j in combinations(range(len(geom)), 2):
        R = v3d.dist(geom[i], geom[j])
        Rcov = (covRadii.R[int(Z[i])] + covRadii.R[int(Z[j])]) / pc.bohr2angstroms
        if R < op.Params.covalent_connect * Rcov:
            C[i, j] = C[j, i] = True

    return C


def addIntcosFromConnectivity(C, intcos, geom):
    addStreFromConnectivity(C, intcos)
    addBendFromConnectivity(C, intcos, geom)
    addTorsFromConnectivity(C, intcos, geom)
    return


# Add Stretches from connectivity.  Return number added.
def addStreFromConnectivity(C, intcos):
    Norig = len(intcos)
    for i, j in combinations(range(len(C)), 2):
        if C[i, j]:
            s = stre.Stre(i, j)
            if s not in intcos:
                intcos.append(s)
    return len(intcos) - Norig  # return number added


# Add Bends from connectivity.  Return number added.
def addBendFromConnectivity(C, intcos, geom):
    Norig = len(intcos)
    Natom = len(geom)
    for i, j in permutations(range(Natom), 2):
        if C[i, j]:
            for k in range(i + 1, Natom):  # make i<k; the constructor checks too
                if C[j, k]:
                    (check, val) = v3d.angle(geom[i], geom[j], geom[k])
                    if not check: continue
                    if val < op.Params.linear_bend_threshold:
                        b = bend.Bend(i, j, k)
                        if b not in intcos:
                            intcos.append(b)
                    else:  # linear angle
                        b = bend.Bend(i, j, k, bendType="LINEAR")
                        if b not in intcos:
                            intcos.append(b)

                        b2 = bend.Bend(i, j, k, bendType="COMPLEMENT")
                        if b2 not in intcos:
                            intcos.append(b2)

    return len(intcos) - Norig


# Add torsions for all bonds present; return number added.
# Use prior existence of linear bends to determine linearity in this function.
def addTorsFromConnectivity(C, intcos, geom):
    Norig = len(intcos)
    Natom = len(geom)

    # Find i-j-k-l where i-j-k && j-k-l are NOT collinear.
    for i, j in permutations(range(Natom), 2):
        if C[i, j]:
            for k in range(Natom):
                if C[k, j] and k != i:

                    # ensure i-j-k is not collinear; that a regular such bend exists
                    b = bend.Bend(i, j, k)
                    if b not in intcos:
                        continue

                    for l in range(i + 1, Natom):
                        if C[l, k] and l != j:

                            # ensure j-k-l is not collinear
                            b = bend.Bend(j, k, l)
                            if b not in intcos:
                                continue

                            t = tors.Tors(i, j, k, l)
                            if t not in intcos:
                                intcos.append(t)

    # Search for additional torsions around collinear segments.
    # Find collinear fragment j-m-k
    for j, m in permutations(range(Natom), 2):
        if C[j, m]:
            for k in range(j + 1, Natom):
                if C[k, m]:
                    # ignore if regular bend
                    b = bend.Bend(j, m, k)
                    if b in intcos:
                        continue

                    # Found unique, collinear j-m-k
                    # Count atoms bonded to m.
                    nbonds = sum(C[m])

                    if nbonds == 2:  # Nothing else is bonded to m

                        # look for an 'I' for I-J-[m]-k-L such that I-J-K is not collinear
                        J = j
                        i = 0
                        while i < Natom:
                            if C[i, J] and i != m:  # i!=J i!=m
                                b = bend.Bend(i, J, k, bendType='LINEAR')
                                if b in intcos:  # i,J,k is collinear
                                    J = i
                                    i = 0
                                    continue
                                else:  # have I-J-[m]-k. Look for L.
                                    I = i
                                    K = k
                                    l = 0
                                    while l < Natom:
                                        if C[l, K] and l != m and l != j and l != i:
                                            b = bend.Bend(l, K, J, bendType='LINEAR')
                                            if b in intcos:  # J-K-l is collinear
                                                K = l
                                                l = 0
                                                continue
                                            else:  # Have found I-J-K-L.
                                                L = l
                                                check, val = v3d.tors(
                                                    geom[I], geom[J], geom[K], geom[L])
                                                if check:
                                                    t = tors.Tors(I, J, K, L)
                                                    if t not in intcos:
                                                        intcos.append(t)
                                        l = l + 1
                            i = i + 1
    return len(intcos) - Norig


def addCartesianIntcos(intcos, geom):
    Norig = len(intcos)
    Natom = len(geom)

    for i in range(Natom):
        intcos.append(cart.Cart(i, 'X'))
        intcos.append(cart.Cart(i, 'Y'))
        intcos.append(cart.Cart(i, 'Z'))

    return len(intcos) - Norig


# Identify linear angles, return them.
def linearBendCheck(intcos, geom, dq):
    linearBends = []

    # This will need generalized later for combination coordinates.
    q = qValues(intcos, geom)

    for i, intco in enumerate(intcos):
        if isinstance(intco, bend.Bend):
            newVal = intco.q(geom) + dq[i]
            A = intco.A
            B = intco.B
            C = intco.C

            # <ABC < 0.  A-C-B should be linear bends.
            if newVal < 0.0:
                linearBends.append(bend.Bend(A, C, B, bendType="LINEAR"))
                linearBends.append(bend.Bend(A, C, B, bendType="COMPLEMENT"))

            # <ABC~pi. Add A-B-C linear bends.
            elif newVal > op.Params.linear_bend_threshold:
                linearBends.append(bend.Bend(A, B, C, bendType="LINEAR"))
                linearBends.append(bend.Bend(A, B, C, bendType="COMPLEMENT"))

    linearBendsMissing = []
    if linearBends:
        print_opt("\n\tThe following linear bends should be present:\n")
        for b in linearBends:
            print_opt('\t' + str(b))

            if b in intcos:
                print_opt(", already present.\n")
            else:
                print_opt(", missing.\n")
                linearBendsMissing.append(b)

    return linearBendsMissing


#####
## params: List of integers corresponding to atoms of distance to be frozen
##	       list of internal coordinates
####
def freezeStretchesFromInputAtomList(frozenStreList, oMolsys):
    if len(frozenStreList) % 2 != 0:
        raise optExceptions.OptFail(
            "Number of atoms in frozen stretch list not divisible by 2.")

    for i in range(0, len(frozenStreList), 2):
        stretch = stre.Stre(frozenStreList[i] - 1, frozenStreList[i + 1] - 1, frozen=True)
        f = checkFragment(stretch.atoms, oMolsys)
        try:
            I = oMolsys._fragments[f]._intcos.index(stretch)
            oMolsys._fragments[f]._intcos[I].frozen = True
        except ValueError:
            print_opt("Frozen stretch not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(stretch)
    return


#####
## params: List of integers corresponding to atoms of bend to be frozen
##	       list of internal coordinates
####
def freezeBendsFromInputAtomList(frozenBendList, oMolsys):
    if len(frozenBendList) % 3 != 0:
        raise optExceptions.OptFail(
            "Number of atoms in frozen bend list not divisible by 3.")

    for i in range(0, len(frozenBendList), 3):
        bendFroz = bend.Bend(
            frozenBendList[i] - 1,
            frozenBendList[i + 1] - 1,
            frozenBendList[i + 2] - 1,
            frozen=True)
        f = checkFragment(bendFroz.atoms, oMolsys)
        try:
            I = oMolsys._fragments[f]._intcos.index(bendFroz)
            oMolsys._fragments[f]._intcos[I].frozen = True
        except ValueError:
            print_opt("Frozen bend not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(bendFroz)
    return


#####
## params: List of integers corresponding to atoms of dihedral to be frozen
##	       list of internal coordinates
####
def freezeTorsionsFromInputAtomList(frozenTorsList, oMolsys):
    if len(frozenTorsList) % 4 != 0:
        raise optExceptions.OptFail(
            "Number of atoms in frozen torsion list not divisible by 4.")

    for i in range(0, len(frozenTorsList), 4):
        torsAngle = tors.Tors(
            frozenTorsList[i] - 1,
            frozenTorsList[i + 1] - 1,
            frozenTorsList[i + 2] - 1,
            frozenTorsList[i + 3] - 1,
            frozen=True)
        f = checkFragment(torsAngle.atoms, oMolsys)
        try:
            I = oMolsys._fragments[f]._intcos.index(torsAngle)
            oMolsys._fragments[f]._intcos[I].frozen = True
        except ValueError:
            print_opt("Frozen dihedral not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(torsAngle)
    return


#####
## params: List of integers indicating atoms, and then 'x' or 'xy', etc.
## indicating cartesians to be frozen
####
def freeze_cartesians_from_input_list(frozen_cart_list, oMolsys):

    for i in range(0, len(frozen_cart_list), 2):
        at = frozen_cart_list[i] - 1
        f = oMolsys.atom2frag_index(at) # get frag #
        for xyz in frozen_cart_list[i+1]:
            newCart = cart.Cart(at, xyz, frozen=True)
            try:
                I = oMolsys._fragments[f]._intcos.index(newCart)
                oMolsys._fragments[f]._intcos[I].frozen = True
            except ValueError:
                print_opt("\tFrozen cartesian not present, so adding it.\n")
                oMolsys._fragments[f]._intcos.append(newCart)
    return


# Check if a group of atoms are in the same fragment (or not).
# Implicitly this function also returns a ValueError for too high atom indices.
# Raise error if different, return fragment if same.
def checkFragment(atomList, oMolsys):
    fragList = oMolsys.atomList2uniqueFragList(atomList)
    if len(fragList) != 1:
        print_opt(
            "Coordinate contains atoms in different fragments. Not currently supported.\n"
        )
        raise optExceptions.OptFail("Atom list contains multiple fragments.")
    return fragList[0]


# Length mod 3 should be checked in optParams
def fixStretchesFromInputList(fixedStreList, oMolsys):
    for i in range(0, len(fixedStreList), 3):  # loop over fixed stretches
        stretch = stre.Stre(fixedStreList[i] - 1, fixedStreList[i + 1] - 1)
        val = fixedStreList[i + 2] / stretch.qShowFactor
        stretch.fixedEqVal = val
        f = checkFragment(stretch.atoms, oMolsys)
        try:
            I = oMolsys._fragments[f]._intcos.index(stretch)
            oMolsys._fragments[f]._intcos[I].fixedEqVal = val
        except ValueError:
            print_opt("Fixed stretch not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(stretch)
    return


def fixBendsFromInputList(fixedBendList, oMolsys):
    for i in range(0, len(fixedBendList), 4):  # loop over fixed bends
        one_bend = bend.Bend(fixedBendList[i] - 1, fixedBendList[i + 1] - 1,
                             fixedBendList[i + 2] - 1)
        val = fixedBendList[i + 3] / one_bend.qShowFactor
        one_bend.fixedEqVal = val
        f = checkFragment(one_bend.atoms, oMolsys)
        try:
            I = oMolsys._fragments[f]._intcos.index(one_bend)
            oMolsys._fragments[f]._intcos[I].fixedEqVal = val
        except ValueError:
            print_opt("Fixed bend not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(one_bend)
    return


def fixTorsionsFromInputList(fixedTorsList, oMolsys):
    for i in range(0, len(fixedTorsList), 5):  # loop over fixed dihedrals
        one_tors = tors.Tors(fixedTorsList[i] - 1, fixedTorsList[i + 1] - 1,
                             fixedTorsList[i + 2] - 1, fixedTorsList[i + 3] - 1)
        val = fixedTorsList[i + 4] / one_tors.qShowFactor
        one_tors.fixedEqVal = val
        f = checkFragment(one_tors.atoms, oMolsys)
        try:
            I = oMolsys._fragments[f]._intcos.index(one_tors)
            oMolsys._fragments[f]._intcos[I].fixedEqVal = val
        except ValueError:
            print_opt("Fixed torsion not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(one_tors)
    return


def addFrozenAndFixedIntcos(oMolsys):
    if op.Params.frozen_distance:
        freezeStretchesFromInputAtomList(op.Params.frozen_distance, oMolsys)
    if op.Params.frozen_bend:
        freezeBendsFromInputAtomList(op.Params.frozen_bend, oMolsys)
    if op.Params.frozen_dihedral:
        freezeTorsionsFromInputAtomList(op.Params.frozen_dihedral, oMolsys)
    if op.Params.frozen_cartesian:
        freeze_cartesians_from_input_list(op.Params.frozen_cartesian, oMolsys)

    if op.Params.fixed_distance:
        fixStretchesFromInputList(op.Params.fixed_distance, oMolsys)
    if op.Params.fixed_bend:
        fixBendsFromInputList(op.Params.fixed_bend, oMolsys)
    if op.Params.fixed_dihedral:
        fixTorsionsFromInputList(op.Params.fixed_dihedral, oMolsys)
    return
