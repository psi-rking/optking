import json
import logging
from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import qcelemental as qcel

from . import bend, cart, dimerfrag, oofp
from . import stre, tors, v3d
from .exceptions import AlgError, OptError
from .v3d import are_collinear
from . import log_name
from . import op
from . import misc

# Functions related to freezing, fixing, determining, and
#    adding coordinates.

logger = logging.getLogger(f"{log_name}{__name__}")


def connectivity_from_distances(geom, Z):
    """
    Creates a matrix (1 or 0) to describe molecular connectivity based on
    nuclear distances
    Parameters
    ----------
    geom : ndarray
        (nat, 3) cartesian geometry
    Z : list[int]
        (nat) list of atomic numbers

    Returns
    -------
    C : ndarray
        (nat, nat)

    """
    nat = geom.shape[0]
    C = np.zeros((len(geom), len(geom)), bool)
    for i, j in combinations(range(nat), 2):
        R = v3d.dist(geom[i], geom[j])
        Rcov = qcel.covalentradii.get(Z[i], missing=4.0) + qcel.covalentradii.get(Z[j], missing=4.0)
        #logger.debug("Checking atoms %d (Z=%d) and %d (Z=%d); R: %.3f; Rcov: %.3f; %s" %(i,
        #             Z[i],j,Z[j],R,Rcov, 'Y' if (R<op.Params.covalent_connect*Rcov) else 'N'))
        if R < op.Params.covalent_connect * Rcov:
            C[i, j] = C[j, i] = True

    return C


def add_auxiliary_bonds(connectivity, intcos, geom, Z):
    """
    Adds "auxiliary" or pseudo-bond stretches to support 1-5 carbon motions.
    Parameters
    ----------
    connectivity : ndarray
        (nat, nat) bond connectivity matrix
    intcos : list[simple.Simple]
            (nat) list of current internal coordinates (Stre, Bend, Tors)
    geom : ndarray
        (nat, 3) cartesian geometry
    Z : list[int]
        (nat) list of atomic numbers

    Returns
    -------
    Nadded : int
        number of auxiliary bonds added on to intcos list

    """
    radii = qcel.covalentradii  # these are in bohr
    Natom = len(geom)  # also in bohr
    Nadded=0

    for a, b in combinations(range(Natom), 2):
        # No auxiliary bonds involving H atoms
        if Z[a] == 1 or Z[b] == 1:
            continue

        R = v3d.dist(geom[a], geom[b])
        Rcov = radii.get(Z[a], missing=4.0) + radii.get(Z[b], missing=4.0)

        if R > Rcov * op.Params.auxiliary_bond_factor:
            continue

        omit = False
        # Omit auxiliary bonds between a and b, if a-c-b
        for c in range(Natom):
            if c not in [a,b]:
                if connectivity[a][c] and connectivity[b][c]:
                    omit = True
                    break
        if omit: continue

        # Omit auxiliary bonds between a and b, if a-c-d-b
        for c in range(Natom):
            if c not in [a,b]:
                if connectivity[c][a]:
                    for d in range(Natom):
                        if d not in [a,b,c]:
                             if connectivity[d][c] and connectivity[d][b]:
                                 omit = True
                                 break
                    if omit: break
        if omit: continue

        s = stre.Stre(a, b)
        if s not in intcos:
            logger.info("Adding auxiliary bond %d - %d" % (a+1,b+1))
            logger.info("Rcov = %10.5f; R = %10.5f; R/Rcov = %10.5f" % (Rcov, R, R/Rcov))
            intcos.append(s)
            Nadded += 1

    return Nadded


def add_intcos_from_connectivity(C, intcos, geom):
    """
    Calls add_x_FromConnectivity for each internal coordinate type
    Parameters
    ----------
    C : ndarray
        (nat, nat) matrix desribing connectivity
        see intcosMisc.connectivity_from_distances()
    intcos : list[simple.Simple]
            (nat) list of current internal coordinates (Stre, Bend, Tors)
    geom : ndarray
        (nat, 3) cartesian geometry

    """
    add_stre_from_connectivity(C, intcos)
    add_bend_from_connectivity(C, intcos, geom)
    add_tors_from_connectivity(C, intcos, geom)
    if op.Params.include_oofp or check_if_oofp_needed(C, intcos, geom):
        add_oofp_from_connectivity(C, intcos, geom)


def add_stre_from_connectivity(C, intcos):
    """
    Adds stretches from connectivity

    Parameters
    ----------
    C : ndarray
        (nat, nat)
    intcos : list[simple.Simple]
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
    """Add Hydrogen bonds to a fragments coordinate list
    Parameters
    ----------
    geom : np.ndarray
    zs : list
    num_atoms : int
    Returns
    -------
    list[stre.HBond]
    -----
    Look for electronegative atoms.
    Find hydrogen atoms between covalent radii test and 2.3 Angstrom
    Notess
    Check these hydrogen atoms are already bonded to an electronegative atom
    Check bond angle >= 90 degrees
    """

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
    intcos : list[simple.Simple]
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
    intcos : list[simple.Simple]
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
                                b = bend.Bend(i, J, k, bend_type="LINEAR")
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
                                            b = bend.Bend(l, K, J, bend_type="LINEAR")
                                            if b in intcos:  # J-K-l is collinear
                                                K = l
                                                l = 0
                                                continue
                                            else:  # Have found I-J-K-L.
                                                L = l
                                                try:
                                                    val = v3d.tors(
                                                        geom[I],
                                                        geom[J],
                                                        geom[K],
                                                        geom[L],
                                                    )
                                                except AlgError:
                                                    pass
                                                else:
                                                    t = tors.Tors(I, J, K, L)
                                                    if t not in intcos:
                                                        intcos.append(t)
                                        l += 1
                            i += 1
    # return len(intcos) - Norig


# Function to determine if out-of-plane angles are needed.
def check_if_oofp_needed(C, intcos, geom):
    # At present, only checks for molecules with least 4 atoms,
    # and for which a single atom is connected to all others.
    # This catches cases like BF3, and CH4.
    Natom = len(C)
    maxNneighbors = max([sum(C[i]) for i in range(Natom)])
    if maxNneighbors == Natom - 1 and maxNneighbors > 2:
        logger.debug("check_if_oofp_needed() is turning oofp ON")
        return True
    else:
        return False


def add_oofp_from_connectivity(C, intcos, geom):
    # Look for:  (terminal atom)-connected to-(tertiary atom)
    Nneighbors = [sum(C[i]) for i in range(len(C))]
    terminal_atoms = [i for i in range(len(Nneighbors)) if Nneighbors[i] == 1]
    errors = []

    # Find adjacent atoms
    vertex_atoms = []
    for T in terminal_atoms:
        vertex_atoms.append(np.where(C[T])[0][0])

    for (T, V) in zip(terminal_atoms, vertex_atoms):
        if Nneighbors[V] < 3:
            pass
        # Find at least 2 other/side atoms
        side = []
        for N in np.where(C[V])[0]:
            if N == T:
                pass
            else:
                side.append(N)

        if len(side) >= 2:
            for side1, side2 in combinations(side, 2):
                try:
                    _ = v3d.oofp(geom[T], geom[V], geom[side1], geom[side2])
                except AlgError:
                    logger.warning("Skipping OOFP (%d, %d, %d, %d)", T, V, side1, side2)
                    errors.append([T, V, side1, side2])
                    continue
                else:
                    oneOofp = oofp.Oofp(T, V, side1, side2)
                    if oneOofp not in intcos:
                        intcos.append(oneOofp)

    # Check all torsions that could not be added. If one or more OOFPs were not added for that
    # central atom, then place add an improper torsion. Not a fully redundant set but an improper
    # torsion in addition to the linear bends should be sufficient
    if errors:
        covered = False
        for coord in intcos:
            if isinstance(coord, oofp.Oofp):
                if V == coord.atoms[1]:
                    covered = True
                if not covered:
                    try:
                        im_tors = improper_torsion_around_oofp(
                            coord.atoms[1],
                            coord.atoms[0],
                            coord.atoms[2],
                            coord.atoms[3]
                        )
                        intcos.append(im_tors)
                    except AlgError:
                        raise AlgError("Tried to add out-of-plane angles but couldn't evaluate all of them.", oofp_failures=oofp.Oofp(T, V, side1, side2))

    return


def add_cartesian_intcos(intcos, geom):
    """
    Add cartesian coordinates to intcos (takes place of internal coordinates)
    Parameters
    ----------
    intcos : list[simple.Simple]
        (nat) list of coordinates
    geom : ndarray
        (nat, 3) cartesian geometry
    Returns
    -------
    """

    # Norig = len(intcos)
    Natom = len(geom)

    for i in range(Natom):
        intcos.append(cart.Cart(i, "X"))
        intcos.append(cart.Cart(i, "Y"))
        intcos.append(cart.Cart(i, "Z"))

    # return len(intcos) - Norig


def linear_bend_check(o_molsys):
    """
    Searches fragments to identify bends which are quasi-linear but not
    previously identified as "linear bends". Called in displace after fragments are adjusted (post backtransform)
    Parameters
    ---------
    o_molsys : MOLSYS class
    dq : ndarray

    Returns
    -------
    list
        missing linear bends
    """

    linear_bends = []
    missing_bends = []

    for frag_index, frag in enumerate(o_molsys.fragments):
        for i, intco in enumerate(frag.intcos):
            if isinstance(intco, bend.Bend):
                new_val = intco.q(frag.geom)
                A, B, C = intco.A, intco.B, intco.C

                # <ABC < 0.  A-C-B should be linear bends.
                if new_val < 0.0:
                    linear_bends.append(bend.Bend(A, C, B, bend_type="LINEAR"))
                    linear_bends.append(bend.Bend(A, C, B, bend_type="COMPLEMENT"))

                # <ABC~pi. Add A-B-C linear bends.
                elif new_val > op.Params.linear_bend_threshold:
                    linear_bends.append(bend.Bend(A, B, C, bend_type="LINEAR"))
                    linear_bends.append(bend.Bend(A, B, C, bend_type="COMPLEMENT"))

        missing_bends = [b for b in linear_bends if b not in frag.intcos]
        bend_report = [f"{b}, already present.\n" if b not in missing_bends else f"{b}, missing.\n" for b in linear_bends]

        if missing_bends:
            logger.warning("\n\tThe following linear bends should be present:\n %s", "\t".join(bend_report))
        # Need to reset this or linear bends will be rechecked for alternate fragments
        linear_bends = []

    return missing_bends


def frozen_stre_from_input(frozen_stre_list, o_molsys):
    """
    Freezes stretch coordinate between atoms
    Parameters
    ----------
    frozen_stre_list : list
        each entry is a list of 2 atoms, indexed from 1
    o_molsys : molsys.Molsys
        optking molecular system
    """
    for S in frozen_stre_list:
        if len(S) != 2:
            raise OptError("Num. of atoms in frozen stretch should be 2.")

        stretch = stre.Stre(S[0] - 1, S[1] - 1, constraint="frozen")
        f = check_fragment(stretch.atoms, o_molsys)
        try:
            frozen_stretch = o_molsys.fragments[f].intcos.index(stretch)
            o_molsys.fragments[f].intcos[frozen_stretch].freeze()
        except ValueError:
            logger.info("Stretch to be frozen not present, so adding it.\n")
            o_molsys.fragments[f].intcos.append(stretch)


def ranged_stre_from_input(ranged_stre_list, o_molsys):
    """
    Creates ranged stretch coordinate between atoms
    Parameters
    ----------
    ranged_stre_list : list
        each entry is a list of 2 atoms (indexed from 1) and 2 floats
    o_molsys : molsys.Molsys
        optking molecular system
    """
    for S in ranged_stre_list:
        if len(S) != 4:
            raise OptError("Num. of entries in ranged stretch should be 4.")

        stretch = stre.Stre(S[0] - 1, S[1] - 1)
        qmin = S[2] / stretch.q_show_factor
        qmax = S[3] / stretch.q_show_factor

        f = check_fragment(stretch.atoms, o_molsys)
        try:
            I = o_molsys.fragments[f].intcos.index(stretch)
            o_molsys.fragments[f].intcos[I].set_range(qmin, qmax)
        except ValueError:
            logger.info("Stretch to be ranged not present, so adding it.\n")
            stretch.set_range(qmin, qmax)
            o_molsys.fragments[f].intcos.append(stretch)


def ext_force_stre_from_input(ext_force_stre_list, o_molsys):
    """
    Creates distance coordinate with external force
    Parameters
    ----------
    ext_force_stre_list : list
        each entry is a list of 2 atoms (indexed from 1), followed by a formula
    o_molsys : molsys.Molsys
        optking molecular system
    """
    for S in ext_force_stre_list:
        if len(S) != 3:
            raise OptError("Num. of entries in ext. force stretch should be 3.")
        stretch = stre.Stre(S[0] - 1, S[1] - 1)
        f = check_fragment(stretch.atoms, o_molsys)
        try:
            I = o_molsys.fragments[f].intcos.index(stretch)
            o_molsys.fragments[f].intcos[I].ext_force = S[2]
        except ValueError:
            logger.info("External force stretch not present, so adding it.\n")
            stretch.ext_force = S[2]
            o_molsys.fragments[f].intcos.append(stretch)


def frozen_bend_from_input(frozen_bend_list, o_molsys):
    """
    Freezes bend coordinates
    Parameters
    ----------
    frozen_bend_list : list
        each entry is a list of 3 atoms numbers, indexed from 1
    o_molsys : molsys.Molsys
        optking molecular system
    """
    for B in frozen_bend_list:
        if len(B) != 3:
            raise OptError("Num. of atoms in frozen bend should be 3.")

        bendFroz = bend.Bend(B[0] - 1, B[1] - 1, B[2] - 1, constraint="frozen")
        f = check_fragment(bendFroz.atoms, o_molsys)
        try:
            freezing_bend = o_molsys.fragments[f].intcos.index(bendFroz)
            o_molsys.fragments[f].intcos[freezing_bend].freeze()
        except ValueError:
            logger.info("Frozen bend not present, so adding it.\n")
            o_molsys.fragments[f].intcos.append(bendFroz)


def ranged_bend_from_input(ranged_bend_list, o_molsys):
    """
    Creates ranged bend coordinates
    Parameters
    ----------
    ranged_bend_list : list
        each entry is a list of 3 atoms, followed by 2 floats
    o_molsys : molsys.Molsys
        optking molecular system
    """
    for B in ranged_bend_list:
        if len(B) != 5:
            raise OptError("Num. of entries in ranged bend should be 5.")

        Rbend = bend.Bend(B[0] - 1, B[1] - 1, B[2] - 1)
        qmin = B[3] / Rbend.q_show_factor
        qmax = B[4] / Rbend.q_show_factor
        f = check_fragment(Rbend.atoms, o_molsys)
        try:
            I = o_molsys.fragments[f].intcos.index(Rbend)
            o_molsys.fragments[f].intcos[I].set_range(qmin, qmax)
        except ValueError:
            logger.info("Frozen bend not present, so adding it.\n")
            Rbend.set_range(qmin, qmax)
            o_molsys.fragments[f].intcos.append(Rbend)


def ext_force_bend_from_input(ext_force_bend_list, o_molsys):
    """
    Creates bend coordinate with external force
    Parameters
    ----------
    ext_force_bend_list : list
        each entry is a list of 3 atoms (indexed from 1), followed by a formula
    o_molsys : molsys.Molsys
        optking molecular system
    """
    for B in ext_force_bend_list:
        if len(B) != 4:
            raise OptError("Num. of entries in ext. force bend should be 4.")
        eBend = bend.Bend(B[0] - 1, B[1] - 1, B[2] - 1)
        f = check_fragment(eBend.atoms, o_molsys)
        try:
            I = o_molsys.fragments[f].intcos.index(eBend)
            o_molsys.fragments[f].intcos[I].ext_force = B[3]
        except ValueError:
            logger.info("External force bend not present, so adding it.\n")
            eBend.ext_force = B[3]
            o_molsys.fragments[f].intcos.append(eBend)


def frozen_tors_from_input(frozen_tors_list, o_molsys):
    """
    Freezes dihedral angles
    Parameters
    ---------
    frozen_tors_list : list
        each entry is list with 4 atoms (indexed from 1)
    o_molsys: molsys.Molsys
        optking molecular system
    """
    for T in frozen_tors_list:
        if len(T) != 4:
            raise OptError("Num. of atoms in frozen torsion should be 4.")

        torsAngle = tors.Tors(T[0] - 1, T[1] - 1, T[2] - 1, T[3] - 1, constraint="frozen")
        f = check_fragment(torsAngle.atoms, o_molsys)
        try:
            I = o_molsys.fragments[f].intcos.index(torsAngle)
            o_molsys.fragments[f].intcos[I].freeze()
        except ValueError:
            logger.info("Frozen dihedral not present, so adding it.\n")
            o_molsys.fragments[f].intcos.append(torsAngle)


def ranged_tors_from_input(ranged_tors_list, o_molsys):
    """
    Creates ranged dihedral angles from input
    Parameters
    ---------
    ranged_tors_list : list
        each entry is list with 4 atoms plus 2 floats
    o_molsys: molsys.Molsys
        optking molecular system
    """
    for T in ranged_tors_list:
        if len(T) != 6:
            raise OptError("Num. of entries in ranged dihedral should be 6.")

        torsAngle = tors.Tors(T[0] - 1, T[1] - 1, T[2] - 1, T[3] - 1)
        qmin = T[4] / torsAngle.q_show_factor
        qmax = T[5] / torsAngle.q_show_factor
        f = check_fragment(torsAngle.atoms, o_molsys)
        try:
            I = o_molsys.fragments[f].intcos.index(torsAngle)
            o_molsys.fragments[f].intcos[I].set_range(qmin, qmax)
        except ValueError:
            logger.info("Frozen dihedral not present, so adding it.\n")
            torsAngle.set_range(qmin, qmax)
            o_molsys.fragments[f].intcos.append(torsAngle)


def ext_force_tors_from_input(extForceTorsList, o_molsys):
    """
    Creates tors coordinate with external force
    Parameters
    ----------
    extForceTorsList : list
        each entry is a list of 4 atoms (indexed from 1), followed by a formula
    o_molsys : molsys.Molsys
        optking molecular system
    """
    for T in extForceTorsList:
        if len(T) != 5:
            raise OptError("Num. of entries in ext. force dihedral should be 5.")
        torsAngle = tors.Tors(T[0] - 1, T[1] - 1, T[2] - 1, T[3] - 1)
        f = check_fragment(torsAngle.atoms, o_molsys)
        try:
            I = o_molsys.fragments[f].intcos.index(torsAngle)
            o_molsys.fragments[f].intcos[I].ext_force = T[4]
        except ValueError:
            logger.info("External force dihedral not present, so adding it.\n")
            torsAngle.ext_force = T[4]
            o_molsys.fragments[f].intcos.append(torsAngle)


def frozen_oofp_from_input(frozenOofpList, o_molsys):
    """
    Freezes out-of-plane angles
    Parameters
    ---------
    frozenOofpList : list
        each entry is list with 4 atoms (indexed from 1)
    o_molsys: molsys.Molsys
        optking molecular system
    """
    for T in frozenOofpList:
        if len(T) != 4:
            raise OptError("Num. of atoms in frozen out-of-plane should be 4.")

        oofpAngle = oofp.Oofp(T[0] - 1, T[1] - 1, T[2] - 1, T[3] - 1, constraint="frozen")
        f = check_fragment(oofpAngle.atoms, o_molsys)
        try:
            I = o_molsys.fragments[f].intcos.index(oofpAngle)
            o_molsys.fragments[f].intcos[I].freeze()
        except ValueError:
            logger.info("Frozen out-of-plane not present, so adding it.\n")
            o_molsys.fragments[f].intcos.append(oofpAngle)


def ranged_oofp_from_input(ranged_oofp_list, o_molsys):
    """
    Creates ranged out-of-plane angles from input
    Parameters
    ---------
    ranged_oofp_list : list
        each entry is list with 4 atoms plus 2 floats
    o_molsys: molsys.Molsys
        optking molecular system
    """
    for T in ranged_oofp_list:
        if len(T) != 6:
            raise OptError("Num. of entries in ranged out-of-plane should be 6.")

        oofpAngle = oofp.Oofp(T[0] - 1, T[1] - 1, T[2] - 1, T[3] - 1)
        qmin = T[4] / oofpAngle.q_show_factor
        qmax = T[5] / oofpAngle.q_show_factor
        f = check_fragment(oofpAngle.atoms, o_molsys)
        try:
            I = o_molsys.fragments[f].intcos.index(oofpAngle)
            o_molsys.fragments[f].intcos[I].set_range(qmin, qmax)
        except ValueError:
            logger.info("Frozen out-of-plane not present, so adding it.\n")
            oofpAngle.set_range(qmin, qmax)
            o_molsys.fragments[f].intcos.append(oofpAngle)


def ext_force_oofp_from_input(ext_force_oofp_list, o_molsys):
    """
    Creates out-of-plane coordinate with external force
    Parameters
    ----------
    ext_force_oofp_list : list
        each entry is a list of 4 atoms (indexed from 1), followed by a formula
    o_molsys : molsys.Molsys
        optking molecular system
    """
    for T in ext_force_oofp_list:
        if len(T) != 5:
            raise OptError("Num. of entries in ext. force out-of-plane should be 5.")
        oofpAngle = oofp.Oofp(T[0] - 1, T[1] - 1, T[2] - 1, T[3] - 1)
        f = check_fragment(oofpAngle.atoms, o_molsys)
        try:
            I = o_molsys.fragments[f].intcos.index(oofpAngle)
            o_molsys.fragments[f].intcos[I].ext_force = T[4]
        except ValueError:
            logger.info("External force out-of-plane not present, so adding it.")
            oofpAngle.ext_force = T[4]
            o_molsys.fragments[f].intcos.append(oofpAngle)


def frozen_cart_from_input(frozen_cart_list, o_molsys):
    """
    Creates frozen cartesian coordinates from input
    Parameters
    ----------
    frozen_cart_list : list
        each entry is list with atom number, then list of 'x','y',or'z'
    o_molsys : molsys.Molsys
    """
    for C in frozen_cart_list:
        if len(C) != 2:
            raise OptError("Num. of entries in frozen cart should be 2.")
        at = C[0] - 1
        f = o_molsys.atom2frag_index(at)  # get frag
        for xyz in C[1]:
            newCart = cart.Cart(at, xyz, constraint="frozen")
            try:
                freezing_cart = o_molsys.fragments[f].intcos.index(newCart)
                o_molsys.fragments[f].intcos[freezing_cart].freeze()
            except ValueError:
                logger.info("\tFrozen cartesian not present, so adding it.\n")
                o_molsys.fragments[f].intcos.append(newCart)


def ranged_cart_from_input(ranged_cart_list, o_molsys):
    """
    Creates ranged cartesian coordinates from input
    Parameters
    ----------
    ranged_cart_list : list
        each entry is list with atom number, then list with only 1 of
        'x','y', or 'z', then 2 floats, min and max
        so if user wants to range x, y and z coordinates, then user should
        enter three separate entries with their ranges
    o_molsys : molsys.Molsys
    """
    for C in ranged_cart_list:
        if len(C) != 4:
            raise OptError("Num. of entries in ranged cart should be 4.")
        atom = C[0] - 1
        f = o_molsys.atom2frag_index(atom)  # get frag
        if len(C[1]) != 1:
            raise OptError("Ranged cartesian only takes 1 of x, y, or z.")
        newCart = cart.Cart(atom, C[1][0])
        qmin = C[2] / newCart.q_show_factor
        qmax = C[3] / newCart.q_show_factor
        try:
            I = o_molsys.fragments[f].intcos.index(newCart)
            o_molsys.fragments[f].intcos[I].set_range(qmin, qmax)
        except ValueError:
            logger.info("\tRanged cartesian not present, so adding it.\n")
            newCart.set_range(qmin, qmax)
            o_molsys.fragments[f].intcos.append(newCart)


def ext_force_cart_from_input(ext_force_cart_list, o_molsys):
    """
    Creates cartesian coordinate with external force
    Parameters
    ----------
    ext_force_cart_list : list
        each entry is list with atom number, then list with only 1 of
        'x','y', or 'z', then formula for external force
    o_molsys : molsys.Molsys
    """
    for C in ext_force_cart_list:
        if len(C) != 3:
            raise OptError("Num. of entries in ext. force Cartesian should be 3.")

        atom = C[0] - 1
        if len(C[1]) != 1:
            raise OptError("External force Cartesian takes only 1 of x/y/z.")
        newCart = cart.Cart(atom, C[1][0])
        f = o_molsys.atom2frag_index(atom)
        try:
            I = o_molsys.fragments[f].intcos.index(newCart)
            o_molsys.fragments[f].intcos[I].ext_force = C[2]
        except ValueError:
            logger.info("External force Cartesian not present, so adding it.")
            newCart.ext_force = C[2]
            o_molsys.fragments[f].intcos.append(newCart)


def check_fragment(atom_list, o_molsys):
    """Check if a group of atoms are in the same fragment (or not).
    Implicitly this function also returns a ValueError for too high atom indices.
    Raise error if different, return fragment if same.
    """
    fragList = o_molsys.atom_list2unique_frag_list(atom_list)
    if len(fragList) != 1:
        logger.error("Coordinate contains atoms in different fragments. Not currently supported.\n")
        raise OptError("Atom list contains multiple fragments.")
    return fragList[0]


def freeze_intrafrag(o_molsys):
    if o_molsys.nfragments < 2:
        raise OptError("Fragments are to be frozen, but there is only one of them")
    for F in o_molsys.fragments:
        F.freeze()


def add_constrained_intcos(o_molsys, params):

    # Frozen coordinates
    def frozen(input, natom, cart=False):
        if cart:
            return misc.int_xyz_float_list(
                misc.tokenize_input_string(input),
                Nint=natom,
                Nxyz=1,
                Nfloat=0
            )
        return misc.int_list(misc.tokenize_input_string(input), natom)

    def ranged(input, natom, cart=False):
        if cart:
            return misc.int_xyz_float_list(
                misc.tokenize_input_string(input),
                Nint=1,
                Nxyz=1,
                Nfloat=2
            )
        else:
            return misc.int_float_list(misc.tokenize_input_string(input), Nint=natom, Nfloat=2)

    def ext_force(input, natom, cart=False):
        if cart:
            return misc.int_xyz_fx_string(input, Nint=1)
        else:
            return misc.int_fx_string(input, natom)

    # encode how to treat each constraint option
    # natoms, contstraint_type (corresponds to method above), specific method to call
    constraints = {
        "frozen_distance": (2, frozen, frozen_stre_from_input),
        "frozen_bend": (3, frozen, frozen_bend_from_input),
        "frozen_dihedral": (4, frozen, frozen_tors_from_input),
        "frozen_oofp": (4, frozen, frozen_oofp_from_input),
        "frozen_cartesian": (1, frozen, frozen_cart_from_input),
        "ranged_distance": (2, ranged, ranged_stre_from_input),
        "ranged_bend": (3, ranged, ranged_bend_from_input),
        "ranged_dihedral": (4, ranged, ranged_tors_from_input),
        "ranged_oofp": (4, ranged, ranged_oofp_from_input),
        "ranged_cartesian": (1, ranged, ranged_cart_from_input),
        "ext_force_distance": (2, ext_force, ext_force_stre_from_input),
        "ext_force_bend": (3, ext_force, ext_force_bend_from_input),
        "ext_force_dihedral": (4, ext_force, ext_force_tors_from_input),
        "ext_force_oofp": (4, ext_force, ext_force_oofp_from_input),
        "ext_force_cartesian": (1, ext_force, ext_force_cart_from_input),
    }

    for key, val in constraints.items():
        option = params.to_dict(by_alias=False).get(key)  # lookup option
        if option:
            # parser converts string to list. constrainer adds constrained coords to molsys
            natom, parser, constrainer = val
            cart = "cartesian" in key
            constrainer(parser(option, natom, cart), o_molsys)

    if op.Params.freeze_intrafrag:
        freeze_intrafrag(o_molsys)

    if op.Params.freeze_all_dihedrals:
        torsions = frozen(params.unfreeze_dihedrals, 4)
        freeze_all_torsions(o_molsys, torsions)

def freeze_all_torsions(molsys, skipped_torsions=[]):
    """ Freeze all intrafragment torsions.
    Parameters
    ----------
    molsys: molsys.Molsys
    skipped_torsions: list[list[int]]
        each set of integers should be four integers denoting a dihedral angle / torsion
    """

    for frag in molsys.fragments:
        for intco in frag.intcos:
            if isinstance(intco, tors.Tors):
                intco.freeze()

    # Seems better to just unfreeze at end rather than check whether each torsion is in this list
    for index_set in skipped_torsions:
        atoms = [index - 1 for index in index_set]
        f = check_fragment(atoms, molsys)
        new_tors = tors.Tors(*atoms)

        try:
            index = molsys.fragments[f].intcos.index(new_tors)
            frag.intcos[index].unfreeze()
        except ValueError:
            logger.info(
                "dihedral angle %s was unfrozen but was not present - adding it.",
                new_tors
            )
            frag.intcos.append(new_tors)

def add_dimer_frag_intcos(o_molsys, params):
    # Look for coordinates in the following order:
    # 1. Check for 1 or list of dicts for 'interfrag_coords' in params
    # TODO: test non-equal weights
    # 2. Check 'frag_ref_atoms' keyword.  It is less flexible than 1.
    # and lower level.  We may want to remove it in the future.
    # 3. Auto-generate reference atoms.
    # TODO: move into a molsys class function?

    input = params.interfrag_coords

    if input:

        # optparams now ensures that type is list[dict], required keys are present, and types a
        # sensible
        for val in input:
            df = dimerfrag.DimerFrag.from_user_dict(val)
            df.update_reference_geometry(o_molsys.frag_geom(df.A_idx), o_molsys.frag_geom(df.B_idx))
            o_molsys.dimer_intcos.append(df)

    elif params.frag_ref_atoms:
        # User-defined ref atoms starting from 1. Decrement here.
        # Assuming that for trimers+, the same reference atoms are
        # desired for each coordinate involving that fragment.
        frag_ref_atoms = deepcopy(params.frag_ref_atoms)
        for iF, F in enumerate(frag_ref_atoms):  # fragments
            frag_1st_atom = o_molsys.frag_1st_atom(iF)
            for iRP, RP in enumerate(F):  # reference points
                for iAT in range(len(RP)):  # atoms
                    frag_ref_atoms[iF][iRP][iAT] -= 1 + frag_1st_atom

        for A, B in combinations(range(o_molsys.nfragments), r=2):
            df = dimerfrag.DimerFrag(A, frag_ref_atoms[A], B, frag_ref_atoms[B])
            df.update_reference_geometry(o_molsys.frag_geom(A), o_molsys.frag_geom(B))
            o_molsys.dimer_intcos.append(df)

    else:  # autogenerate interfswap_min_maxragment coordinates
        # Tolerance for collinearity of ref points. Can be mad smaller, but its
        # riskier to start wth ref points the make very large angles
        col_tol = params.interfrag_collinear_tol
        for A, B in combinations(range(o_molsys.nfragments), r=2):
            xyzA = o_molsys.frag_geom(A)
            xyzB = o_molsys.frag_geom(B)
            # Choose closest two atoms for 1st reference pt.
            (refA1, refB1) = o_molsys.closest_atoms_between_2_frags(A, B)
            frag_ref_atomsA = [[refA1]]
            frag_ref_atomsB = [[refB1]]
            # Find ref. pt. 2 on A.
            if not o_molsys.fragments[A].is_atom():
                for i in range(o_molsys.fragments[A].natom):
                    if i == refA1 or are_collinear(xyzA[i], xyzA[refA1], xyzB[refB1], col_tol):
                        continue
                    refA2 = i
                    frag_ref_atomsA.append([refA2])
                    break
                else:
                    raise OptError("could not find 2nd atom on fragment {:d}".format(A + 1))
            # Find ref. pt. 2 on B.
            if not o_molsys.fragments[B].is_atom():
                for i in range(o_molsys.fragments[B].natom):
                    if i == refB1 or are_collinear(xyzB[i], xyzB[refB1], xyzA[refA1], col_tol):
                        continue
                    refB2 = i
                    frag_ref_atomsB.append([refB2])
                    break
                else:
                    raise OptError("could not find 2nd atom on fragment {:d}".format(B + 1))
            # Find ref. pt. 3 on A.
            if o_molsys.fragments[A].natom > 2 and not o_molsys.fragments[A].is_linear():
                for i in range(o_molsys.fragments[A].natom):
                    if i in [refA1, refA2] or are_collinear(xyzA[i], xyzA[refA2], xyzA[refA1], col_tol):
                        continue
                    frag_ref_atomsA.append([i])
                    break
                else:
                    raise OptError("could not find 3rd atom on fragment {:d}".format(A + 1))
            # Find ref. pt. 3 on B.
            if o_molsys.fragments[B].natom > 2 and not o_molsys.fragments[B].is_linear():
                for i in range(o_molsys.fragments[B].natom):
                    if i in [refB1, refB2] or are_collinear(xyzB[i], xyzB[refB2], xyzB[refB1], col_tol):
                        continue
                    frag_ref_atomsB.append([i])
                    break
                else:
                    raise OptError("could not find 3rd atom on fragment {:d}".format(A + 1))

            df = dimerfrag.DimerFrag(A, frag_ref_atomsA, B, frag_ref_atomsB)
            df.update_reference_geometry(o_molsys.frag_geom(A), o_molsys.frag_geom(B))
            o_molsys.dimer_intcos.append(df)

        # print('end of add_dimer_frag_intcos')
        # print(o_molsys)
    return


def improper_torsion_around_oofp(center, a, b, c, geom):
    """ To help compensate for missing the oofp. Create an improper torsion which goes from
    T1-T2-C-T3 where T denotes terminal atoms, C denotes the OOFP center, and T1-C-T3 is linear """

    if v3d.are_collinear(geom[center], geom[a], geom[b]):
        return tors.Tors(a, c, center, b)
    elif v3d.are_collinear(geom[center], geom[b], geom[c]):
        return tors.Tors(b, a, center, c)
    else:
        return tors.Tors(a, b, center, c)
