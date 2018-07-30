
from math import sqrt
import numpy as np
import logging

import oofp
import optparams as op
import bend
import tors

from linearAlgebra import symmMatInv
from printTools import printMatString, printArrayString

# Simple operations on internal :148
# coordinate sets.  For example,
# return values, Bmatrix or fix orientation.


# q    -> qValues
# DqDx -> Bmat
def qValues(intcos, geom):
    """
    Calculates internal coordinates
    Paramters
    ---------
    intcos : list
        (nat) list of stretches, bends, etc...
    geom : ndarray
        (nat, 3) cartesian geometry
    Returns
    -------
    ndarray
        internal coordinate values
    """

    q = np.array([i.q(geom) for i in intcos])
    return q


def qShowValues(intcos, geom):
    """
    Scales internal coordiante values by coordinate factor
    Parameters
    ----------
    intcos : list
        (nat) list of stretches, bends, etc
    geom : ndarray
        (nat, 3) cartesian geometry
    Returns
    -------
    ndarray
        scaled internal coordinate values
    """
    q = np.array([i.qShow(geom) for i in intcos])
    return q


def updateDihedralOrientations(intcos, geom):
    """ wrapper for updating orientation of dihedrals
    calls updateOrientation for each tors coordinate
    """
    for intco in intcos:
        if isinstance(intco, tors.Tors) or isinstance(intco, oofp.OOFP):
            intco.updateOrientation(geom)


def fixBendAxes(intcos, geom):
    for intco in intcos:
        if isinstance(intco, bend.Bend):
            intco.fixBendAxes(geom)


def unfixBendAxes(intcos):
    for intco in intcos:
        if isinstance(intco, bend.Bend):
            intco.unfixBendAxes()


def Bmat(intcos, geom, masses=None):
    logger = logging.getLogger(__name__)
    Nint = len(intcos)
    Ncart = geom.size

    B = np.zeros((Nint, Ncart), float)
    for i, intco in enumerate(intcos):
        intco.DqDx(geom, B[i])

    # mass-weight & solve
    #   sqrtmmm = np.repeat(np.sqrt(mass), 3)
    #   sqrtmmminv = np.divide(1.0, sqrtmmm)
    #   mwhess = np.einsum('i,ij,j->ij', sqrtmmminv, nmwhess, sqrtmmminv)
    if masses:
        logger.debug("mass weighting B matrix\n")
        B = inv_mass_weighted(B, intcos, masses)

    return B


def Gmat(intcos, geom, masses=None):
    """ Calculates BuB^T (calculates B matrix)

    Parameters
    ----------
    intcos : list
        list of internal coordinates
    geom : ndarray
        (nat, 3) cartesian geometry

    """
    B = Bmat(intcos, geom, masses)

    if masses:
        B = inv_mass_weighted(B, intcos, masses)

    return np.dot(B, B.T)


def Gmat_B(B, intcos, masses=None):
    """ Calculate G matrix (BuB^T) if B matrix has already been calculated """
    if masses:
        B = inv_mass_weighted(B, intcos, masses)

    return np.dot(B, B.T)


def inv_mass_weighted(matrix, intcos, masses):
    """ Mass weight a B or G matrix for dqi/dxj mass weight wrt atom j

    Parameters
    ----------
    matrix : ndarray
        (len(intcos), 3nat)

    masses: list

    Returns
    -------
    ndarray
        mass weighted matrix
    """

    sqrtm = np.array([np.repeat(np.sqrt(masses), 3)]*len(intcos))
    return np.divide(matrix, sqrtm)


def qForces(intcos, geom, gradient_x):
    """
    Transforms forces from cartesian to internal coordinates
    fq = (BuB^T)^(-1)*B*f_x
    Parameters
    ----------
    intcos : list
        stretches, bends, etc
    geom : ndarray
        (nat, 3) cartesian geometry
    gradient _x :
        (nat, 3) cartesian gradient
    Returns
    -------
    ndarray
        (1, intcos) internal coordinate forces (-1) * gradient
    """
    if len(intcos) == 0 or len(geom) == 0:
        return np.zeros(0, float)
    B = Bmat(intcos, geom)
    fx = -1.0 * gradient_x  # gradient -> forces
    temp_arr = np.dot(B, fx.T)
    G = np.dot(B, B.T)
    Ginv = symmMatInv(G, redundant=True)
    fq = np.dot(Ginv, temp_arr.T)
    return fq


def qShowForces(intcos, forces):
    """
    returns scaled forces (does not recompute)
    This may not be tested.
    """
    # qaJ = np.copy(forces)
    factors = np.asarray([intco.fShowFactor for intco in intcos])
    qaJ = forces * factors
    # for i, intco in enumerate(intcos):
    #    qaJ[i] *= intco.fShowFactor
    return qaJ


def constraint_matrix(intcos):
    if not any([coord.frozen for coord in intcos]):
        return None
    C = np.zeros((len(intcos), len(intcos)), float)
    for i, coord in enumerate(intcos):
        if coord.frozen:
            C[i, i] = 1.0
    return C
    # frozen_coords = np.asarray([int(coord.frozen) for coord in intcos])
    # if np.any(frozen_coords):
    #    return np.diagflat(frozen_coords)
    # else:
    #    return None


def projectRedundanciesAndConstraints(intcos, geom, fq, H):
    """Project redundancies and constraints out of forces and Hessian"""
    logger = logging.getLogger(__name__)
    # dim = len(intcos)
    # compute projection matrix = G G^-1
    G = Gmat(intcos, geom)
    G_inv = symmMatInv(G, redundant=True)
    Pprime = np.dot(G, G_inv)
    if op.Params.print_lvl >= 3:
        logger.debug("\tProjection matrix for redundancies.\n\n" + printMatString(Pprime))
    # Add constraints to projection matrix
    C = constraint_matrix(intcos)

    if np.any(C):
        logger.info("Adding constraints for projection.\n" + printMatString(C))
        # print_opt(np.dot(C, np.dot(Pprime, C)))
        CPC = np.zeros((len(intcos), len(intcos)), float)
        CPC[:, :] = np.dot(C, np.dot(Pprime, C))
        CPCInv = symmMatInv(CPC, redundant=True)
        P = np.zeros((len(intcos), len(intcos)), float)
        P[:, :] = Pprime - np.dot(Pprime, np.dot(C, np.dot(CPCInv, np.dot(C, Pprime))))
    else:
        P = Pprime

    # Project redundancies out of forces.
    # fq~ = P fq
    fq[:] = np.dot(P, fq.T)

    if op.Params.print_lvl >= 3:
        logger.info("\tInternal forces in au, after projection of redundancies"
                    + "and constraints.\n" + printArrayString(fq))
    # Project redundancies out of Hessian matrix.
    # Peng, Ayala, Schlegel, JCC 1996 give H -> PHP + 1000(1-P)
    # The second term appears unnecessary and sometimes messes up Hessian updating.
    tempMat = np.dot(H, P)
    H[:, :] = np.dot(P, tempMat)
    # for i in range(dim)
    #    H[i,i] += 1000 * (1.0 - P[i,i])
    # for i in range(dim)
    #    for j in range(i):
    #        H[j,i] = H[i,j] = H[i,j] + 1000 * (1.0 - P[i,j])
    if op.Params.print_lvl >= 3:
        logger.debug("Projected (PHP) Hessian matrix\n" + printMatString(H))


def applyFixedForces(oMolsys, fq, H, stepNumber):
    logger = logging.getLogger(__name__)
    x = oMolsys.geom
    for iF, F in enumerate(oMolsys._fragments):
        for i, intco in enumerate(F.intcos):
            if intco.fixed:
                # TODO we may need to add iF to the location to get unique locations
                # for each fragment
                location = oMolsys.frag_1st_intco(iF) + i
                val = intco.q(x)
                eqVal = intco.fixedEqVal

                # Increase force constant by 5% of initial value per iteration
                k = (1 + 0.05 * stepNumber) * op.Params.fixed_coord_force_constant
                force = k * (eqVal - val)
                H[location][location] = k
                fq[location] = force
                fix_forces_report = ("\n\tAdding user-defined constraint:"
                                     + "Fragment %d; Coordinate %d:\n" % (iF + 1, i + 1))
                fix_forces_report += ("\t\tValue = %12.6f; Fixed value    = %12.6f"
                                      % (val, eqVal))
                fix_forces_report += ("\t\tForce = %12.6f; Force constant = %12.6f"
                                      % (force, k))
                logger.info(fix_forces_report)

                # Delete coupling between this coordinate and others.
                logger.info("\t\tRemoving off-diagonal coupling between coordinate"
                            + "%d and others." % (location + 1))
                for j in range(len(H)):  # gives first dimension length
                    if j != location:
                        H[j][location] = H[location][j] = 0.0


# """
# def massWeightedUMatrixCart(masses):
#    atom = 1
#    masses = [15.9994, 1.00794, 1.00794]
#    U = np.zeros((3 * nAtom, 3 * nAtom), float)
#    for i in range (0, (3 * nAtom)):
#        U[i][i] = 1 / sqrt(masses[atom - 1])
#        if (i % 3 == 0):
#            nAtom += 1
#    return U
# """


def convertHessianToInternals(H, intcos, geom, masses=None, g_x=None):
    logger = logging.getLogger(__name__)
    logger.info("Converting Hessian from cartesians to internals.\n")

    G = Gmat(intcos, geom, masses)
    Ginv = symmMatInv(G, redundant=True)
    B = Bmat(intcos, geom, masses)
    Atranspose = np.dot(Ginv, B)

    Hworking = H.copy()
    if g_x is None:  # A^t Hxy A
        logger.info("Neglecting force/B-matrix derivative term, only correct at"
                    + "stationary points.\n")
    else:  # A^t (Hxy - Kxy) A;    K_xy = sum_q ( grad_q[I] d^2(q_I)/(dx dy) )
        logger.info("Including force/B-matrix derivative term.\n")

        g_q = np.dot(Atranspose, g_x)
        Ncart = 3 * len(geom)
        dq2dx2 = np.zeros((Ncart, Ncart), float)  # should be cart x cart for fragment ?

        for I, q in enumerate(intcos):
            dq2dx2[:] = 0
            q.Dq2Dx2(geom, dq2dx2)  # d^2(q_I)/ dx_i dx_j

            for a in range(Ncart):
                for b in range(Ncart):
                    # adjust indices for multiple fragments
                    Hworking[a, b] -= g_q[I] * dq2dx2[a, b] 

    Hq = np.dot(Atranspose, np.dot(Hworking, Atranspose.T))
    return Hq


def convertHessianToCartesians(Hint, intcos, geom, masses=None, g_q=None):
    logger = logging.getLogger(__name__)
    logger.info("Converting Hessian from internals to cartesians.\n")

    B = Bmat(intcos, geom, masses)
    Hxy = np.dot(B.T, np.dot(Hint, B))

    if g_q is None:  # Hxy =  B^t Hij B
        logger.info("Neglecting force/B-matrix derivative term, only correct at"
                    + "stationary points.\n")
    else:  # Hxy += dE/dq_I d2(q_I)/dxdy
        logger.info("Including force/B-matrix derivative term.\n")
        Ncart = 3 * len(geom)

        dq2dx2 = np.zeros((Ncart, Ncart), float)  # should be cart x cart for fragment ?
        for I, q in enumerate(intcos):
            dq2dx2[:] = 0
            q.Dq2Dx2(geom, dq2dx2)

            for a in range(Ncart):
                for b in range(Ncart):
                    Hxy[a, b] += g_q[I] * dq2dx2[a, b]

    return Hxy




def torsContainsBend(b, t):
    return (b.atoms in [t.atoms[0:3],
                        list(reversed(t.atoms[0:3])),
                        t.atoms[1:4],
                        list(reversed(t.atoms[1:4]))])


def removeOldNowLinearBend(atoms, intcos):
    """ For given bend [A,B,C], remove any regular bends as well as any torsions
    which contain it 
    """
    logger = logging.getLogger(__name__)
    b = bend.Bend(atoms[0], atoms[1], atoms[2])
    logger.info("Removing Old Linear Bend")
    logger.info(str(b) + '\n')
    intcos[:] = [I for I in intcos if I != b]
    intcos[:] = [
        I for I in intcos if not (isinstance(I, tors.Tors) and torsContainsBend(b, I))
    ]
    #    if b == Coord:
    #        del intcos[iCoord]
    #    if isinstance(Coord, tors.Tors):
    #        if (atoms in [Coord.atoms[0:3], list(reversed(Coord.atoms[0:3])),
    #                      Coord.atoms[1:4], list(reversed(Coord.atoms[1:4]))]):
    #            del intcos[iCoord]
