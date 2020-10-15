from itertools import combinations, permutations
import logging
import numpy as np
import qcelemental as qcel
from copy import deepcopy

from .exceptions import AlgError, OptError
from . import optparams as op
from . import v3d
from . import stre
from . import tors
from . import bend
from . import cart
from . import dimerfrag
from .v3d import are_collinear

# Functions related to freezing, fixing, determining, and
#    adding coordinates.


def connectivity_from_distances(geom, Z):
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


def add_intcos_from_connectivity(C, intcos, geom):
    """
    Calls add_x_FromConnectivity for each internal coordinate type
    Parameters
    ----------
    C : ndarray
        (nat, nat) matrix desribing connectivity
        see intcosMisc.connectivity_from_distances()
    intcos : list
            (nat) list of current internal coordinates (Stre, Bend, Tors)
    geom : ndarray
        (nat, 3) cartesian geometry

    """
    add_stre_from_connectivity(C, intcos)
    add_bend_from_connectivity(C, intcos, geom)
    add_tors_from_connectivity(C, intcos, geom)


def add_stre_from_connectivity(C, intcos):
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

    """

    # Norig = len(intcos)
    for i, j in combinations(range(len(C)), 2):
        if C[i, j]:
            s = stre.Stre(i, j)
            if s not in intcos:
                intcos.append(s)
    # return len(intcos) - Norig  # return number added


def add_h_bonds(geom, zs: list, num_atoms):
    """ Add Hydrogen bonds to a fragments coordinate list
    Parameters
    ----------
    geom : np.ndarray
    zs : list
    num_atoms : int
    Returns
    -------
    list[stre.HBond]
    Notes
    -----
    Look for electronegative atoms.
    Find hydrogen atoms between covalent radii test and 2.3 Angstroms
    Check these hydrogen atoms are already bonded to an electronegative atom
    Check bond angle >= 90 degrees
    """
    logger = logging.getLogger(__name__)
    logger.warning("This method should be adjusted after dimer fragments")

    # N, O, F, P, S, Cl as proposed by Bakken and Helgaker
    electroneg_zs = [7, 8, 9, 15, 16, 17]
    # Get atom indices (within a fragment) for the electronegative atoms present and the
    # hydrogen atoms present also get
    electronegs_present = [index for index, z in enumerate(zs) if z in electroneg_zs]
    hydrogens = [index for index, i in enumerate(zs) if i == 1]

    # Some shortcuts
    min_factor = op.Params.covalent_connect
    limit = op.Params.h_bond_connect
    cov = qcel.covalentradii.get
    h_bonds = []

    for index_i, i in enumerate(electronegs_present):
        for j in hydrogens:
            if j < i:
                # do only i < j
                break
            distance = v3d.dist(geom[i], geom[j])
            covalent_thresh = min_factor * (cov(zs[index_i], missing=4.0) + cov(1, missing=4.0))
            if limit > distance > covalent_thresh:
                for k in range(num_atoms):
                    # grab k part in electronegs_present.
                    if k in electronegs_present:
                        test_angle = v3d.angle(geom[k], geom[j], geom[i])
                        if test_angle >= (np.pi / 2):
                            h_bonds.append(stre.HBond(i, j))

                            break  # Add hydrogen bond if 1 appropriate angle in connected atoms
    return h_bonds


def add_bend_from_connectivity(C, intcos, geom):
    """
    Adds Bends from connectivity

    Parameters
    ---------
    C : ndarray
        (nat, nat) unitary connectivity matrix
    intcos : list
        (nat) list of internal coordinates
    geom : ndarray
        (nat, 3) cartesian geometry
    Returns
    -------

    """

    # Norig = len(intcos)
    nat = len(geom)
    for i, j in permutations(range(nat), 2):
        if C[i, j]:
            for k in range(i + 1, nat):  # make i<k; the constructor checks too
                if C[j, k]:
                    try:
                        val = v3d.angle(geom[i], geom[j], geom[k])
                    except AlgError:
                        pass
                    else:
                        if val > op.Params.linear_bend_threshold:
                            b = bend.Bend(i, j, k, bend_type="LINEAR")
                            if b not in intcos:
                                intcos.append(b)

                            b2 = bend.Bend(i, j, k, bend_type="COMPLEMENT")
                            if b2 not in intcos:
                                intcos.append(b2)
                        else:
                            b = bend.Bend(i, j, k)
                            if b not in intcos:
                                intcos.append(b)
    # return len(intcos) - Norig


def add_tors_from_connectivity(C, intcos, geom):
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
                                b = bend.Bend(i, J, k, bend_type='LINEAR')
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
                                            b = bend.Bend(l, K, J, bend_type='LINEAR')
                                            if b in intcos:  # J-K-l is collinear
                                                K = l
                                                l = 0
                                                continue
                                            else:  # Have found I-J-K-L.
                                                L = l
                                                try:
                                                    val = v3d.tors(
                                                        geom[I], geom[J], geom[K], geom[L])
                                                except AlgError:
                                                    pass
                                                else:
                                                    t = tors.Tors(I, J, K, L)
                                                    if t not in intcos:
                                                        intcos.append(t)
                                        l += 1
                            i += 1
    # return len(intcos) - Norig


def add_cartesian_intcos(intcos, geom):
    """
    Add cartesian coordinates to intcos (takes place of internal coordinates)
    Parameters
    ----------
    intcos : list
        (nat) list of coordinates
    geom : ndarray
        (nat, 3) cartesian geometry
    Returns
    -------
    """

    # Norig = len(intcos)
    Natom = len(geom)

    for i in range(Natom):
        intcos.append(cart.Cart(i, 'X'))
        intcos.append(cart.Cart(i, 'Y'))
        intcos.append(cart.Cart(i, 'Z'))

    # return len(intcos) - Norig


def linear_bend_check(oMolsys, dq):
    """
    Searches fragments to identify bends which are quasi-linear but not
    previously identified as "linear bends".
    Parameters
    ---------
    oMolsys : MOLSYS class
    dq : ndarray

    Returns
    -------
    list
        missing linear bends
    """

    logger = logging.getLogger(__name__)
    linearBends = []

    for iF, F in enumerate(oMolsys.fragments):
        for i, intco in enumerate(F.intcos):
            if isinstance(intco, bend.Bend):
                newVal = intco.q(F.geom) + dq[oMolsys.frag_1st_intco(iF)+i]
                A,B,C = intco.A, intco.B, intco.C
    
                # <ABC < 0.  A-C-B should be linear bends.
                if newVal < 0.0:
                    linearBends.append(bend.Bend(A, C, B, bend_type="LINEAR"))
                    linearBends.append(bend.Bend(A, C, B, bend_type="COMPLEMENT"))
    
                # <ABC~pi. Add A-B-C linear bends.
                elif newVal > op.Params.linear_bend_threshold:
                    linearBends.append(bend.Bend(A, B, C, bend_type="LINEAR"))
                    linearBends.append(bend.Bend(A, B, C, bend_type="COMPLEMENT"))
    
        linearBendsMissing = []
        if linearBends:
            linear_bend_string = "\n\tThe following linear bends should be present:\n"
            for b in linearBends:
                linear_bend_string += '\t' + str(b)
    
                if b in F.intcos:
                    linear_bend_string += ", already present.\n"
                else:
                    linear_bend_string += ", missing.\n"
                    linearBendsMissing.append(b)
            logger.warning(linearBendsMissing)

    return linearBendsMissing


def frozen_stre_from_input(frozenStreList, oMolsys):
    """
    Freezes stretch coordinate between atoms
    Parameters
    ----------
    frozenStreList : list
        each entry is a list of 2 atoms, indexed from 1
    oMolsys : molsys.Molsys
        optking molecular system
    """
    logger = logging.getLogger(__name__)
    for S in frozenStreList:
        if len(S) != 2:
            raise OptError("Num. of atoms in frozen stretch should be 2.")

        stretch = stre.Stre(S[0] - 1, S[1] - 1, constraint='frozen')
        f = check_fragment(stretch.atoms, oMolsys)
        try:
            frozen_stretch = oMolsys.fragments[f].intcos.index(stretch)
            oMolsys.fragments[f].intcos[frozen_stretch].freeze()
        except ValueError:
            logger.info("Stretch to be frozen not present, so adding it.\n")
            oMolsys.fragments[f].intcos.append(stretch)


def ranged_stre_from_input(rangedStreList, oMolsys):
    """
    Creates ranged stretch coordinate between atoms
    Parameters
    ----------
    rangedStreList : list
        each entry is a list of 2 atoms (indexed from 1) and 2 floats
    oMolsys : molsys.Molsys
        optking molecular system
    """
    for S in rangedStreList:
        if len(S) != 4:
            raise OptError("Num. of entries in ranged stretch should be 4.")

        stretch = stre.Stre(S[0]-1, S[1]-1)
        qmin = S[2] / stretch.q_show_factor
        qmax = S[3] / stretch.q_show_factor

        f = check_fragment(stretch.atoms, oMolsys)
        try:
            I = oMolsys.fragments[f].intcos.index(stretch)
            oMolsys.fragments[f].intcos[I].set_range(qmin,qmax)
        except ValueError:
            logger = logging.getLogger(__name__)
            logger.info("Stretch to be ranged not present, so adding it.\n")
            stretch.set_range(qmin,qmax)
            oMolsys.fragments[f].intcos.append(stretch)


def frozen_bend_from_input(frozenBendList, oMolsys):
    """
    Freezes bend coordinates
    Parameters
    ----------
    frozenBendList : list
        each entry is a list of 3 atoms numbers, indexed from 1
    oMolsys : molsys.Molsys
        optking molecular system
    """
    logger = logging.getLogger(__name__)
    for B in frozenBendList:
        if len(B) != 3:
            raise OptError("Num. of atoms in frozen bend should be 3.")

        bendFroz = bend.Bend(B[0]-1, B[1]-1, B[2]-1, constraint='frozen')
        f = check_fragment(bendFroz.atoms, oMolsys)
        try:
            freezing_bend = oMolsys.fragments[f].intcos.index(bendFroz)
            oMolsys.fragments[f].intcos[freezing_bend].freeze()
        except ValueError:
            logger.info("Frozen bend not present, so adding it.\n")
            oMolsys.fragments[f].intcos.append(bendFroz)

def ranged_bend_from_input(rangedBendList, oMolsys):
    """
    Creates ranged bend coordinates
    Parameters
    ----------
    frozenBendList : list
        each entry is a list of 3 atoms, followed by 2 floats
    oMolsys : molsys.Molsys
        optking molecular system
    """
    logger = logging.getLogger(__name__)
    for B in rangedBendList:
        if len(B) != 5:
            raise OptError("Num. of entries in ranged bend should be 5.")

        Rbend = bend.Bend(B[0]-1, B[1]-1, B[2]-1)
        qmin = B[3] / Rbend.q_show_factor
        qmax = B[4] / Rbend.q_show_factor
        f = check_fragment(Rbend.atoms, oMolsys)
        try:
            I = oMolsys.fragments[f].intcos.index(Rbend)
            oMolsys.fragments[f].intcos[I].set_range(qmin,qmax)
        except ValueError:
            logger.info("Frozen bend not present, so adding it.\n")
            Rbend.set_range(qmin,qmax)
            oMolsys.fragments[f].intcos.append(Rbend)


def frozen_tors_from_input(frozenTorsList, oMolsys):
    """
    Freezes dihedral angles
    Parameters
    ---------
    frozenTorsList : list
        each entry is list with 4 atoms (indexed from 1)
    oMolsys: object
        optking molecular system
    """
    for T in frozenTorsList:
        if len(T) != 4:
            raise OptError("Num. of atoms in frozen torsion should be 4.")

        torsAngle = tors.Tors(T[0]-1,T[1]-1,T[2]-1,T[3]-1,constraint='frozen')
        f = check_fragment(torsAngle.atoms, oMolsys)
        try:
            I = oMolsys.fragments[f].intcos.index(torsAngle)
            oMolsys.fragments[f].intcos[I].freeze()
        except ValueError:
            logging.info("Frozen dihedral not present, so adding it.\n")
            oMolsys.fragments[f].intcos.append(torsAngle)


def ranged_tors_from_input(rangedTorsList, oMolsys):
    """
    Creates ranged dihedral angles from input
    Parameters
    ---------
    frozenTorsList : list
        each entry is list with 4 atoms plus 2 floats
    oMolsys: object
        optking molecular system
    """
    for T in rangedTorsList:
        if len(T) != 6:
            raise OptError("Num. of entries in ranged dihedral should be 6.")

        torsAngle = tors.Tors(T[0]-1,T[1]-1,T[2]-1,T[3]-1)
        qmin = T[4] / torsAngle.q_show_factor
        qmax = T[5] / torsAngle.q_show_factor
        f = check_fragment(torsAngle.atoms, oMolsys)
        try:
            I = oMolsys.fragments[f].intcos.index(torsAngle)
            oMolsys.fragments[f].intcos[I].set_range(qmin,qmax)
        except ValueError:
            logging.info("Frozen dihedral not present, so adding it.\n")
            torsAngle.set_range(qmin,qmax)
            oMolsys.fragments[f].intcos.append(torsAngle)


def frozen_oofp_from_input(frozenOofpList, oMolsys):
    """
    Freezes out-of-plane angles
    Parameters
    ---------
    frozenOofpList : list
        each entry is list with 4 atoms (indexed from 1)
    oMolsys: object
        optking molecular system
    """
    for T in frozenOofpList:
        if len(T) != 4:
            raise OptError("Num. of atoms in frozen out-of-plane should be 4.")

        oofpAngle = oofp.Oofp(T[0]-1,T[1]-1,T[2]-1,T[3]-1,constraint='frozen')
        f = check_fragment(oofpAngle.atoms, oMolsys)
        try:
            I = oMolsys.fragments[f].intcos.index(oofpAngle)
            oMolsys.fragments[f].intcos[I].freeze()
        except ValueError:
            logging.info("Frozen out-of-plane not present, so adding it.\n")
            oMolsys.fragments[f].intcos.append(oofpAngle)


def ranged_oofp_from_input(rangedOofpList, oMolsys):
    """
    Creates ranged out-of-plane angles from input
    Parameters
    ---------
    frozenOofpList : list
        each entry is list with 4 atoms plus 2 floats
    oMolsys: object
        optking molecular system
    """
    for T in rangedOofpList:
        if len(T) != 6:
            raise OptError("Num. of entries in ranged out-of-plane should be 6.")

        oofpAngle = oofp.Oofp(T[0]-1,T[1]-1,T[2]-1,T[3]-1)
        qmin = T[4] / oofpAngle.q_show_factor
        qmax = T[5] / oofpAngle.q_show_factor
        f = check_fragment(oofpAngle.atoms, oMolsys)
        try:
            I = oMolsys.fragments[f].intcos.index(oofpAngle)
            oMolsys.fragments[f].intcos[I].set_range(qmin,qmax)
        except ValueError:
            logging.info("Frozen out-of-plane not present, so adding it.\n")
            oofpAngle.set_range(qmin,qmax)
            oMolsys.fragments[f].intcos.append(oofpAngle)


def frozen_cart_from_input(frozen_cart_list, oMolsys):
    """
    Creates frozen cartesian coordinates from input
    Parameters
    ----------
    frozen_cart_list : list
        each entry is list with atom number, then list of 'x','y',or'z'
    oMolsys : molsys.Molsys
    """
    logger = logging.getLogger(__name__)
    for C in frozen_cart_list:
        if len(C) != 2:
            raise OptError("Num. of entries in frozen cart should be 2.")
        at = C[0]-1
        f = oMolsys.atom2frag_index(at)  # get frag
        for xyz in C[1]:
            newCart = cart.Cart(at, xyz, constraint='frozen')
            try:
                freezing_cart = oMolsys.fragments[f].intcos.index(newCart)
                oMolsys.fragments[f].intcos[freezing_cart].freeze()
            except ValueError:
                logger.info("\tFrozen cartesian not present, so adding it.\n")
                oMolsys.fragments[f].intcos.append(newCart)

def ranged_cart_from_input(ranged_cart_list, oMolsys):
    """
    Creates ranged cartesian coordinates from input
    Parameters
    ----------
    frozen_cart_list : list
        each entry is list with atom number, then list with only 1 of
        'x','y', or 'z', then 2 floats, min and max
        so if user wants to range x, y and z coordinates, then user should
        enter three separate entries with their ranges
    oMolsys : molsys.Molsys
    """
    logger = logging.getLogger(__name__)
    for C in ranged_cart_list:
        if len(C) != 4:
            raise OptError("Num. of entries in frozen cart should be 4.")
        atom = C[0]-1
        f = oMolsys.atom2frag_index(atom)  # get frag
        if len(C[1]) != 1:
            raise OptError("Ranged cartesian only takes 1 of x, y, or z.")
        newCart = cart.Cart(atom, C[1][0])
        qmin = C[2] / newCart.q_show_factor
        qmax = C[3] / newCart.q_show_factor
        try:
            I = oMolsys.fragments[f].intcos.index(newCart)
            oMolsys.fragments[f].intcos[I].set_range(qmin,qmax)
        except ValueError:
            logger.info("\tFrozen cartesian not present, so adding it.\n")
            newCart.set_range(qmin,qmax)
            oMolsys.fragments[f].intcos.append(newCart)


def check_fragment(atomList, oMolsys):
    """Check if a group of atoms are in the same fragment (or not).
    Implicitly this function also returns a ValueError for too high atom indices.
    Raise error if different, return fragment if same.
    """
    logger = logging.getLogger(__name__)
    fragList = oMolsys.atom_list2unique_frag_list(atomList)
    if len(fragList) != 1:
        logger.error(
            "Coordinate contains atoms in different fragments. Not currently supported.\n"
        )
        raise OptError("Atom list contains multiple fragments.")
    return fragList[0]


# TODO Length mod 3 should be checked in OptParams
def fix_stretches_from_input(fixedStreList, oMolsys):
    logger = logging.getLogger(__name__)
    for i in range(0, len(fixedStreList), 3):  # loop over fixed stretches
        stretch = stre.Stre(fixedStreList[i] - 1, fixedStreList[i + 1] - 1)
        val = fixedStreList[i + 2] / stretch.q_show_factor
        stretch.fixedEqVal = val
        f = check_fragment(stretch.atoms, oMolsys)
        try:
            fixing_stretch = oMolsys.fragments[f].intcos.index(stretch)
            oMolsys.fragments[f].intcos[fixing_stretch].fixed_eq_val = val
        except ValueError:
            logger.info("Fixed stretch not present, so adding it.\n")
            oMolsys.fragments[f].intcos.append(stretch)


def fix_bends_from_input(fixedBendList, oMolsys):
    logger = logging.getLogger(__name__)
    for i in range(0, len(fixedBendList), 4):  # loop over fixed bends
        one_bend = bend.Bend(fixedBendList[i] - 1, fixedBendList[i + 1] - 1,
                             fixedBendList[i + 2] - 1)
        val = fixedBendList[i + 3] / one_bend.q_show_factor
        one_bend.fixedEqVal = val
        f = check_fragment(one_bend.atoms, oMolsys)
        try:
            fixing_bend = oMolsys.fragments[f].intcos.index(one_bend)
            oMolsys.fragments[f].intcos[fixing_bend].fixed_eq_val = val
        except ValueError:
            logger.info("Fixed bend not present, so adding it.\n")
            oMolsys.fragments[f].intcos.append(one_bend)


def fix_torsions_from_input(fixedTorsList, oMolsys):
    logger = logging.getLogger(__name__)
    for i in range(0, len(fixedTorsList), 5):  # loop over fixed dihedrals
        one_tors = tors.Tors(fixedTorsList[i] - 1, fixedTorsList[i + 1] - 1,
                             fixedTorsList[i + 2] - 1, fixedTorsList[i + 3] - 1)
        val = fixedTorsList[i + 4] / one_tors.q_show_factor
        one_tors.fixedEqVal = val
        f = check_fragment(one_tors.atoms, oMolsys)
        try:
            fixing_tors = oMolsys.fragments[f].intcos.index(one_tors)
            oMolsys.fragments[f].intcos[fixing_tors].fixed_eq_val = val
        except ValueError:
            logger.info("Fixed torsion not present, so adding it.\n")
            oMolsys.fragments[f].intcos.append(one_tors)


def freeze_intrafrag(oMolsys):
    if oMolsys.nfragments < 2:
        raise OptError('Fragments are to be frozen, but there is only one of them')
    for F in oMolsys.fragments:
        F.freeze()


def add_frozen_and_fixed_intcos(oMolsys):
    if op.Params.frozen_distance:
        frozen_stre_from_input(op.Params.frozen_distance, oMolsys)
    if op.Params.frozen_bend:
        frozen_bend_from_input(op.Params.frozen_bend, oMolsys)
    if op.Params.frozen_dihedral:
        frozen_tors_from_input(op.Params.frozen_dihedral, oMolsys)
    if op.Params.frozen_oofp:
        frozen_oofp_from_input(op.Params.frozen_oofp, oMolsys)
    if op.Params.frozen_cartesian:
        frozen_cart_from_input(op.Params.frozen_cartesian, oMolsys)

    if op.Params.ranged_distance:
        ranged_stre_from_input(op.Params.ranged_distance, oMolsys)
    if op.Params.ranged_bend:
        ranged_bend_from_input(op.Params.ranged_bend, oMolsys)
    if op.Params.ranged_dihedral:
        ranged_tors_from_input(op.Params.ranged_dihedral, oMolsys)
    if op.Params.ranged_oofp:
        ranged_oofp_from_input(op.Params.ranged_oofp, oMolsys)
    if op.Params.ranged_cartesian:
        ranged_cart_from_input(op.Params.ranged_cartesian, oMolsys)

    if op.Params.fixed_distance:
        fix_stretches_from_input(op.Params.fixed_distance, oMolsys)
    if op.Params.fixed_bend:
        fix_bends_from_input(op.Params.fixed_bend, oMolsys)
    if op.Params.fixed_dihedral:
        fix_torsions_from_input(op.Params.fixed_dihedral, oMolsys)
    if op.Params.freeze_intrafrag:
        freeze_intrafrag(oMolsys)


def add_dimer_frag_intcos(oMolsys):
    # Look for coordinates in the following order:
    # 1. Check for 1 or list of dicts for 'interfrag_coords' in params
    # TODO: test non-equal weights
    # 2. Check 'frag_ref_atoms' keyword.  It is less flexible than 1.
    # and lower level.  We may want to remove it in the future.
    # 3. Auto-generate reference atoms.
    # TODO: move into a molsys class function?

    if op.Params.interfrag_coords != None:
        if type(op.Params.interfrag_coords) in [list, tuple]:
            for coord in op.Params.interfrag_coords:
                c = eval(coord)
                df = dimerfrag.DimerFrag.fromUserDict(c)
                df.update_reference_geometry(oMolsys.frag_geom(df.A_idx), oMolsys.frag_geom(df.B_idx))
                oMolsys.dimer_intcos.append(df)
        else:
            c = eval(op.Params.interfrag_coords)
            df = dimerfrag.DimerFrag.fromUserDict(c)
            df.update_reference_geometry(oMolsys.frag_geom(df.A_idx), oMolsys.frag_geom(df.B_idx))
            oMolsys.dimer_intcos.append(df)
   
    elif op.Params.frag_ref_atoms is not None:
        # User-defined ref atoms starting from 1. Decrement here.
        # Assuming that for trimers+, the same reference atoms are
        # desired for each coordinate involving that fragment.
        frag_ref_atoms = deepcopy(op.Params.frag_ref_atoms)
        for iF, F in enumerate(frag_ref_atoms): # fragments
            for iRP, RP in enumerate(F):        # reference points
                for iAT in range(len(RP)):      # atoms
                    frag_ref_atoms[iF][iRP][iAT] -= 1

        for A, B in combinations(range(oMolsys.nfragments),r=2):
            df = dimerfrag.DimerFrag(A, frag_ref_atoms[A], B, frag_ref_atoms[B])
            df.update_reference_geometry(oMolsys.frag_geom(A), oMolsys.frag_geom(B))
            oMolsys.dimer_intcos.append(df)

    else: # autogenerate interfragment coordinates
        for A, B in combinations(range(oMolsys.nfragments),r=2):
            xyzA = oMolsys.frag_geom(A)
            xyzB = oMolsys.frag_geom(B)
            # Choose closest two atoms for 1st reference pt.
            (refA1, refB1) = oMolsys.closest_atoms_between_2_frags(A, B)
            frag_ref_atomsA = [ [refA1] ]
            frag_ref_atomsB = [ [refB1] ]
            # Find ref. pt. 2 on A.
            if not oMolsys.fragments[A].is_atom():
                for i in range(oMolsys.fragments[A].natom):
                    if i == refA1 or are_collinear(xyzA[i],xyzA[refA1],xyzB[refB1]):
                        continue
                    refA2 = i
                    frag_ref_atomsA.append([refA2])
                    break
                else:
                    raise OptError('could not find 2nd atom on fragment {:d}'.format(A+1))
            # Find ref. pt. 2 on B.
            if not oMolsys.fragments[B].is_atom():
                for i in range(oMolsys.fragments[B].natom):
                    if i == refB1 or are_collinear(xyzB[i],xyzB[refB1],xyzA[refA1]):
                        continue
                    refB2 = i
                    frag_ref_atomsB.append([refB2])
                    break
                else:
                    raise OptError('could not find 2nd atom on fragment {:d}'.format(B+1))
            # Find ref. pt. 3 on A.
            if oMolsys.fragments[A].natom > 2 and not oMolsys.fragments[A].is_linear():
                for i in range(oMolsys.fragments[A].natom):
                    if i in [refA1,refA2] or are_collinear(xyzA[i],xyzA[refA2],xyzA[refA1]):
                        continue
                    frag_ref_atomsA.append([i])
                    break
                else:
                    raise OptError('could not find 3rd atom on fragment {:d}'.format(A+1))
            # Find ref. pt. 3 on B.
            if oMolsys.fragments[B].natom > 2 and not oMolsys.fragments[B].is_linear():
                for i in range(oMolsys.fragments[B].natom):
                    if i in [refB1,refB2] or are_collinear(xyzB[i],xyzB[refB2],xyzB[refB1]):
                        continue
                    frag_ref_atomsB.append([i])
                    break
                else:
                    raise OptError('could not find 3rd atom on fragment {:d}'.format(A+1))

            df = dimerfrag.DimerFrag(A, frag_ref_atomsA, B, frag_ref_atomsB)
            df.update_reference_geometry(oMolsys.frag_geom(A), oMolsys.frag_geom(B))
            oMolsys.dimer_intcos.append(df)

        #print('end of add_dimer_frag_intcos')
        #print(oMolsys)
    return

