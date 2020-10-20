from math import sqrt
import numpy as np
import logging

from . import optparams as op
from . import oofp
from . import bend
from . import tors
from .linearAlgebra import symm_mat_inv, symm_mat_root
from .printTools import print_mat_string, print_array_string

# Some of these functions act on an arbitrary list of simple internals,
# geometry etc. that may or may not be in a molecular system.
# Also, a few complicated function that act on molecular system
# forces and Hessians.


def q_values(intcos, geom):
    # available for simple intco lists
    vals = [intco.q(geom) for intco in intcos]
    return np.asarray( vals )


def Bmat(intcos, geom, masses=None):
    # Allocate memory for full system.
    # Returns mass-weighted Bmatrix if masses are supplied.
    # available for simple intco lists
    Nint  = len(intcos)
    B = np.zeros( (Nint, 3*len(geom)) )

    for i, intco in enumerate(intcos):
        intco.DqDx(geom, B[i])

    if type(masses) is np.ndarray:
        sqrtm = np.array([np.repeat(np.sqrt(masses), 3)]*Nint, float)
        B[:] = np.divide(B, sqrtm)

    return B


def q_forces(intcos, geom, gradient_x, B=None):
    """Transforms cartesian gradient to internals

    Parameters
    ----------
    intcos : list
        stretches, bends, etc
    geom : ndarray
        (nat, 3) cartesian geometry
    gradient_x :
        (3nat, 1) cartesian gradient
    B matrix: (optional)

    Returns
    -------
    ndarray
        forces in internal coordinates (-1 * gradient)
    Notes
    -----
    fq = (BuB^T)^(-1)*B*f_x

    """
    if not intcos or not len(geom):
        return np.zeros(0)

    if B is None:
        B = Bmat(intcos, geom)

    fx = np.multiply(-1.0, gradient_x)  # gradient -> forces
    G = np.dot(B, B.T)
    Ginv = symm_mat_inv(G, redundant=True)
    fq = np.dot(np.dot(Ginv, B), fx)
    return fq


def project_redundancies_and_constraints(oMolsys, fq, H):
    """Project redundancies and constraints out of forces and Hessian"""
    logger = logging.getLogger(__name__)
    Nint = oMolsys.num_intcos
    # compute projection matrix = G G^-1
    G = oMolsys.compute_g_mat()
    G_inv = symm_mat_inv(G, redundant=True)
    Pprime = np.dot(G, G_inv)
    # logger.debug("\tProjection matrix for redundancies.\n\n" + print_mat_string(Pprime))
    # Add constraints to projection matrix
    C = oMolsys.constraint_matrix(fq) # returns None, if aren't any
    # fq is passed to Supplement matrix with ranged variables that are at their limit

    if C is not None:
        logger.debug("Adding constraints for projection.\n" + print_mat_string(C))
        CPC = np.zeros( (Nint, Nint))
        CPC[:, :] = np.dot(C, np.dot(Pprime, C))
        CPCInv = symm_mat_inv(CPC, redundant=True)
        P = np.zeros( (Nint, Nint) )
        P[:, :] = Pprime - np.dot(Pprime, np.dot(C, np.dot(CPCInv, np.dot(C, Pprime))))
    else:
        P = Pprime

    # Project redundancies out of forces.
    # fq~ = P fq
    fq[:] = np.dot(P, fq.T)

    #if op.Params.print_lvl >= 3:
    logger.debug("\n\tInternal forces in au, after projection of redundancies"
                 + " and constraints.\n" + print_array_string(fq))
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
        logger.debug("Projected (PHP) Hessian matrix\n" + print_mat_string(H))


def apply_external_forces(oMolsys, fq, H, stepNumber):
    logger = logging.getLogger(__name__)
    report = ''
    for iF, F in enumerate(oMolsys._fragments):
        for i, intco in enumerate(F.intcos):
            if intco.has_ext_force:
                if report == '': report = 'Adding external forces\n'
                # TODO we may need to add iF to the location to get unique locations
                # for each fragment
                val = intco.q_show(oMolsys.geom)
                ext_force = intco.ext_force_val(oMolsys.geom)

                location = oMolsys.frag_1st_intco(iF) + i
                fq[location] += ext_force
                report += 'Frag {:d}, Coord {:d}, Value {:10.5f}, Force {:12.6f}\n'.format(
                           iF+1, i+1, val, ext_force)
                # modify Hessian later ?
                # H[location][location] = k
                # Delete coupling between this coordinate and others.
                #logger.info("\t\tRemoving off-diagonal coupling between coordinate"
                #            + "%d and others." % (location + 1))
                #for j in range(len(H)):  # gives first dimension length
                #    if j != location:
                #        H[j][location] = H[location][j] = 0.0

    logger.info(report)


def hessian_to_internals(H, oMolsys, g_x=None):
    """ converts the hessian from cartesian coordinates into internal coordinates 
    
    Parameters
    ----------
    H : ndarray
        Hessian in cartesians
    oMolsys : molecular system
    Returns
    -------
    Hq : ndarray
    """
    logger = logging.getLogger(__name__)
    logger.info("Converting Hessian from cartesians to internals.\n")
    # B = compute_b_mat(intcos, geom)
    B = oMolsys.compute_b_mat()
    G = np.dot(B, B.T)
    geom = oMolsys.geom

    Ginv = symm_mat_inv(G, redundant=True)
    Atranspose = np.dot(Ginv, B)

    Hworking = H.copy()
    if g_x is None:  # A^t Hxy A
        logger.info("Neglecting force/B-matrix derivative term, only correct at"
                    + "stationary points.\n")
    else:  # A^t (Hxy - Kxy) A;    K_xy = sum_q ( grad_q[I] d^2(q_I)/(dx dy) )
        logger.info("Including force/B-matrix derivative term.\n")

        g_q = np.dot(Atranspose, g_x)
        Ncart = 3 * oMolsys.natom
        dq2dx2 = np.zeros((Ncart, Ncart))  # should be cart x cart for fragment ?

        for I, q in enumerate(oMolsys.intcos):
            dq2dx2[:] = 0
            q.Dq2Dx2(geom, dq2dx2)  # d^2(q_I)/ dx_i dx_j

            for a in range(Ncart):
                for b in range(Ncart):
                    # adjust indices for multiple fragments
                    Hworking[a, b] -= g_q[I] * dq2dx2[a, b] 

    Hq = np.dot(Atranspose, np.dot(Hworking, Atranspose.T))
    return Hq

"""
def massWeightHessianInternals(Hq, intcos, geom, masses):
    Mass-weights the internal coordinate hessian
    
    Parameters
    ----------
    Hq : ndarray
        hessian in internal coordinates
    intcos : list 
        internal coordinates (stretches, bends, etc...)
    geom : ndarray
        Cartesian geometry
    massses : ndarray

    Returns
    -------
    Hq : ndarray

    Notes
    -----
    Mass-weights the hessian by (G^1/2 Hq G^1/2.T) where G is mass-weighted.
    
    
    logger = logging.getLogger(__name__)

    GM = compute_g_mat(intcos, geom, masses)
    GM_root = symm_mat_root(GM)
    HqM = np.dot(np.dot(GM_root, Hq), GM_root.T)
    
    return HqM
"""


def convert_hessian_to_cartesians(Hint, intcos, geom, masses=None, g_q=None):
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

        dq2dx2 = np.zeros((Ncart, Ncart) )  # should be cart x cart for fragment ?
        for I, q in enumerate(intcos):
            dq2dx2[:] = 0
            q.Dq2Dx2(geom, dq2dx2)

            for a in range(Ncart):
                for b in range(Ncart):
                    Hxy[a, b] += g_q[I] * dq2dx2[a, b]

    return Hxy


def tors_contains_bend(b, t):
    return (b.atoms in [t.atoms[0:3],
                        list(reversed(t.atoms[0:3])),
                        t.atoms[1:4],
                        list(reversed(t.atoms[1:4]))])


def remove_old_now_linear_bend(atoms, intcos):
    """ For given bend [A,B,C], remove any regular bends as well as any torsions
    which contain it 
    """
    logger = logging.getLogger(__name__)
    b = bend.Bend(atoms[0], atoms[1], atoms[2])
    logger.info("Removing Old Linear Bend")
    logger.info(str(b) + '\n')
    intcos[:] = [I for I in intcos if I != b]
    intcos[:] = [
        I for I in intcos if not (isinstance(I, tors.Tors) and tors_contains_bend(b, I))
    ]
