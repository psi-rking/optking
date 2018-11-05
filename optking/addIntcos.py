from itertools import combinations, permutations
import logging

import numpy as np
import qcelemental as qcel

from . import bend
from . import cart
from . import optExceptions
from . import optparams as op
from . import stre
from . import tors
from . import v3d
from .intcosMisc import qValues


def connectivityFromDistances(geom, Z):
    """
    Creates a matrix (1 or 0) to describe molecular connectivity based on
    nuclear distances
    Parameters
    ----------
    geom : ndarray
        (nat, 3) cartesian geometry
    Z : list
        (nat) list of atomic numbers

    Returns
    -------
    C : ndarray
        (nat, nat)

    """
    nat = geom.shape[0]
    C = np.zeros((len(geom), len(geom)), bool)
    # logger = logging.getLogger(__name__)
    for i, j in combinations(range(nat), 2):
        R = v3d.dist(geom[i], geom[j])
        Rcov = qcel.covalentradii.get(Z[i], missing=4.0) + qcel.covalentradii.get(Z[j], missing=4.0)
        # logger.debug("Trying to connect atoms " + str(i) + ' and ' + str(j) + " distance is: " +
        #            str(qcel.covalentradii.get(Z[i], missing=4.0) + qcel.covalentradii.get(Z[j], missing=4.0)))
        if R < op.Params.covalent_connect * Rcov:
            C[i, j] = C[j, i] = True

    return C


def addIntcosFromConnectivity(C, intcos, geom):
    """
    Calls add_x_FromConnectivity for each internal coordinate type
    Parameters
    ----------
    C : ndarray
        (nat, nat) matrix desribing connectivity
        see intcosMisc.connectivityFromDistances()
    intcos : list
            (nat) list of current internal coordiantes (Stre, Bend, Tors)
    geom : ndarray
        (nat, 3) cartesian geometry

    """
    addStreFromConnectivity(C, intcos)
    addBendFromConnectivity(C, intcos, geom)
    addTorsFromConnectivity(C, intcos, geom)


def addStreFromConnectivity(C, intcos):
    """
    Adds stretches from connectivity

    Parameters
    ----------
    C : ndarray
        (nat, nat)
    intcos : list
        (nat)
    Returns
    -------
    int
        number of stretches added

    """

    # Norig = len(intcos)
    for i, j in combinations(range(len(C)), 2):
        if C[i, j]:
            s = stre.Stre(i, j)
            if s not in intcos:
                intcos.append(s)
    # return len(intcos) - Norig  # return number added


def addBendFromConnectivity(C, intcos, geom):
    """
    Adds Bends from connectivity

    Paramters
    ---------
    C : ndarray
        (nat, nat) unitary connectivity matrix
    intcos : list
        (nat) list of internal coordinates
    geom : ndarray
        (nat, 3) cartesian geometry
    Returns
    -------
    float
        number of bends added

    """

    # Norig = len(intcos)
    nat = len(geom)
    for i, j in permutations(range(nat), 2):
        if C[i, j]:
            for k in range(i + 1, nat):  # make i<k; the constructor checks too
                if C[j, k]:
                    try:
                        val = v3d.angle(geom[i], geom[j], geom[k])
                    except optExceptions.AlgFail:
                        pass
                    else:
                        if val > op.Params.linear_bend_threshold:
                            b = bend.Bend(i, j, k, bendType="LINEAR")
                            if b not in intcos:
                                intcos.append(b)

                            b2 = bend.Bend(i, j, k, bendType="COMPLEMENT")
                            if b2 not in intcos:
                                intcos.append(b2)
                        else:
                            b = bend.Bend(i, j, k)
                            if b not in intcos:
                                intcos.append(b)
    # return len(intcos) - Norig


def addTorsFromConnectivity(C, intcos, geom):
    """
    Add torisions for all bonds present and determine linearity from existance of
    linear bends

    Parameters
    ----------
    C : ndarray
        (nat, nat) connectivity matrix
    intcos : list
        (nat) list of stretches, bends, etc...
    geom : ndarray
        (nat, 3) cartesian geometry
    Returns
    -------
    float
        number of torsions added
    """

    # Norig = len(intcos)
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
                                                try:
                                                    val = v3d.tors(
                                                        geom[I], geom[J], geom[K], geom[L])
                                                except optExceptions.AlgFail:
                                                    pass
                                                else:
                                                    t = tors.Tors(I, J, K, L)
                                                    if t not in intcos:
                                                        intcos.append(t)
                                        l += 1
                            i += 1
    # return len(intcos) - Norig


def addCartesianIntcos(intcos, geom):
    """
    Add cartesian coordiantes to intcos (takes place of internal coordiantes)
    Parameters
    ----------
    intcos : list
        (nat) list of coordinates
    geom : ndarray
        (nat, 3) cartesian geometry
    Returns
    -------
    float
        number of coordinates added
    """

    # Norig = len(intcos)
    Natom = len(geom)

    for i in range(Natom):
        intcos.append(cart.Cart(i, 'X'))
        intcos.append(cart.Cart(i, 'Y'))
        intcos.append(cart.Cart(i, 'Z'))

    # return len(intcos) - Norig


def linearBendCheck(intcos, geom, dq):
    """
    Identifies near linear angles
    Paramters
    ---------
    intcos : list
        (nat) list of stretches, bends, etc
    geom : ndarray
        (nat, 3) cartesian geometry
    dq : ndarray

    Returns
    -------
    list
        missing linear bends
    """

    logger = logging.getLogger(__name__)
    linearBends = []

    # TODO This will need generalized later for combination coordinates.
    # q = qValues(intcos, geom)

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
        linear_bend_string = ("\n\tThe following linear bends should be present:\n")
        for b in linearBends:
            linear_bend_string += '\t' + str(b)

            if b in intcos:
                linear_bend_string += (", already present.\n")
            else:
                linear_bend_string += (", missing.\n")
                linearBendsMissing.append(b)
        logger.warning(linearBendsMissing)
    return linearBendsMissing


def freezeStretchesFromInputAtomList(frozenStreList, oMolsys):
    """
    Freezes stretch coordinate between atoms
    Parameters
    ----------
    frozenStreList : list
        (2*x) list of pairs of atoms. 1 indexed
    oMolsys : object
        optking molecular system
    """

    logger = logging.getLogger(__name__)
    if len(frozenStreList) % 2 != 0:
        raise optExceptions.OptFail(
            "Number of atoms in frozen stretch list not divisible by 2.")

    for i in range(0, len(frozenStreList), 2):
        stretch = stre.Stre(frozenStreList[i] - 1, frozenStreList[i + 1] - 1, frozen=True)
        f = checkFragment(stretch.atoms, oMolsys)
        try:
            frozen_stretch = oMolsys._fragments[f]._intcos.index(stretch)
            oMolsys._fragments[f]._intcos[frozen_stretch].frozen = True
        except ValueError:
            logger.info("Stretch to be frozen not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(stretch)


def freezeBendsFromInputAtomList(frozenBendList, oMolsys):
    """
    Freezes bend coordinates
    Parameters
    ----------
    frozenBendList : list
        (3*x) list of triplets of atoms
    oMolsys : object
        optking molecular system
    """
    logger = logging.getLogger(__name__)
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
            freezing_bend = oMolsys._fragments[f]._intcos.index(bendFroz)
            oMolsys._fragments[f]._intcos[freezing_bend].frozen = True
        except ValueError:
            logger.info("Frozen bend not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(bendFroz)


def freezeTorsionsFromInputAtomList(frozenTorsList, oMolsys):
    """
    Freezes dihedral angles
    Paramters
    ---------
    frozenTorsList : list
        (4*x) list of atoms in sets of four
    oMolsys: object
        optking molecular system
    """
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
            freezing_tors = oMolsys._fragments[f]._intcos.index(torsAngle)
            oMolsys._fragments[f]._intcos[freezing_tors].frozen = True
        except ValueError:
            logging.info("Frozen dihedral not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(torsAngle)


def freeze_cartesians_from_input_list(frozen_cart_list, oMolsys):
    """ params: List of integers indicating atoms, and then 'x' or 'xy', etc.
    indicating cartesians to be frozen

    Parameters
    ----------
    frozen_cart_list : string
        uses 1 indexing for atoms. number xy or z or any combo
        add example
    """
    logger = logging.getLogger(__name__)
    for i in range(0, len(frozen_cart_list), 2):
        at = frozen_cart_list[i] - 1
        f = oMolsys.atom2frag_index(at)  # get frag
        for xyz in frozen_cart_list[i+1]:
            newCart = cart.Cart(at, xyz, frozen=True)
            try:
                freezing_cart = oMolsys._fragments[f]._intcos.index(newCart)
                oMolsys._fragments[f]._intcos[freezing_cart].frozen = True
            except ValueError:
                logger.info("\tFrozen cartesian not present, so adding it.\n")
                oMolsys._fragments[f]._intcos.append(newCart)


def checkFragment(atomList, oMolsys):
    """Check if a group of atoms are in the same fragment (or not).
    Implicitly this function also returns a ValueError for too high atom indices.
    Raise error if different, return fragment if same.
    """
    logger = logging.getLogger(__name__)
    fragList = oMolsys.atomList2uniqueFragList(atomList)
    if len(fragList) != 1:
        logger.error(
            "Coordinate contains atoms in different fragments. Not currently supported.\n"
        )
        raise optExceptions.OptFail("Atom list contains multiple fragments.")
    return fragList[0]


# TODO Length mod 3 should be checked in optParams
def fixStretchesFromInputList(fixedStreList, oMolsys):
    logger = logging.getLogger(__name__)
    for i in range(0, len(fixedStreList), 3):  # loop over fixed stretches
        stretch = stre.Stre(fixedStreList[i] - 1, fixedStreList[i + 1] - 1)
        val = fixedStreList[i + 2] / stretch.qShowFactor
        stretch.fixedEqVal = val
        f = checkFragment(stretch.atoms, oMolsys)
        try:
            fixing_stretch = oMolsys._fragments[f]._intcos.index(stretch)
            oMolsys._fragments[f]._intcos[fixing_stretch].fixedEqVal = val
        except ValueError:
            logger.info("Fixed stretch not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(stretch)


def fixBendsFromInputList(fixedBendList, oMolsys):
    logger = logging.getLogger(__name__)
    for i in range(0, len(fixedBendList), 4):  # loop over fixed bends
        one_bend = bend.Bend(fixedBendList[i] - 1, fixedBendList[i + 1] - 1,
                             fixedBendList[i + 2] - 1)
        val = fixedBendList[i + 3] / one_bend.qShowFactor
        one_bend.fixedEqVal = val
        f = checkFragment(one_bend.atoms, oMolsys)
        try:
            fixing_bend = oMolsys._fragments[f]._intcos.index(one_bend)
            oMolsys._fragments[f]._intcos[fixing_bend].fixedEqVal = val
        except ValueError:
            logger.info("Fixed bend not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(one_bend)


def fixTorsionsFromInputList(fixedTorsList, oMolsys):
    logger = logging.getLogger(__name__)
    for i in range(0, len(fixedTorsList), 5):  # loop over fixed dihedrals
        one_tors = tors.Tors(fixedTorsList[i] - 1, fixedTorsList[i + 1] - 1,
                             fixedTorsList[i + 2] - 1, fixedTorsList[i + 3] - 1)
        val = fixedTorsList[i + 4] / one_tors.qShowFactor
        one_tors.fixedEqVal = val
        f = checkFragment(one_tors.atoms, oMolsys)
        try:
            fixing_tors = oMolsys._fragments[f]._intcos.index(one_tors)
            oMolsys._fragments[f]._intcos[fixing_tors].fixedEqVal = val
        except ValueError:
            logger.info("Fixed torsion not present, so adding it.\n")
            oMolsys._fragments[f]._intcos.append(one_tors)


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
