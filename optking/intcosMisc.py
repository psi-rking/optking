from math import sqrt
import numpy as np
import logging
import warnings

from . import optparams as op
from . import oofp
from . import bend
from . import tors
from .linearAlgebra import symm_mat_inv, symm_mat_root
from .printTools import print_mat_string, print_array_string

# Some of these functions act on an arbitrary list of simple internals,
# geometry etc. that may or may not be in a molecular system.
# Also, a few complicated functions that act on molecular system
# forces and Hessians.


def q_values(intcos, geom):
    """ available for simple intco lists

    Parameters
    ----------
    intcos : list
    geom : np.ndarray

    Returns
    -------
    np.ndarray

    """
    warnings.warn("""Method - intcosMisc.q_values - has moved to the molecular system. 
                  self.q should be used instead""")
    vals = [intco.q(geom) for intco in intcos]
    return np.asarray(vals)


def Bmat(intcos, geom, masses=None):
    """ Returns mass-weighted Bmatrix if masses are supplied.
    available for simple intco lists
    """
    # Allocate memory for full system.
    warnings.warn(
        """Method - intcosMisc.Bmat - has moved to the molecular system - self.wilson_b_mat. """)
    warnings.warn("Only test B should need to call this")
    Nint = len(intcos)
    B = np.zeros((Nint, 3*len(geom)))

    for i, intco in enumerate(intcos):
        intco.dqdx(geom, B[i])

    if type(masses) is np.ndarray:
        sqrtm = np.array([np.repeat(np.sqrt(masses), 3)]*Nint, float)
        B[:] = np.divide(B, sqrtm)

    return B


def q_forces(gradient_x, oMolsys, B=None):
    """Transforms cartesian gradient to internals

    Parameters
    ----------
    oMolsys
    gradient_x :
        (3nat, 1) cartesian gradient
    B : (optional)

    Returns
    -------
    ndarray
        forces in internal coordinates (-1 * gradient)
    Notes
    -----
    fq = (BuB^T)^(-1)*atom_b*f_x

    """
    logger = logging.getLogger(__name__)
    warnings.warn("""method has moved to molsys. Should not be used here""")
    logger.warning("Method was changed. This code may not be tested.")
    if not oMolsys.intcos or not oMolsys.geom:
        return np.zeros(0)

    if B is None:
        B = oMolsys.wilson_b_mat()

    fx = np.multiply(-1.0, gradient_x)  # gradient -> forces
    G = np.dot(B, B.T)
    Ginv = np.linalg.pinv(G)
    fq = np.dot(np.dot(Ginv, B), fx)
    return fq


def project_redundancies_and_constraints(oMolsys, fq, H):
    """Project redundancies and constraints out of forces and Hessian"""
    # def project_redundancies_and_constraints(intcos, geom, fq, H):
    logger = logging.getLogger(__name__)
    logger.critical("Using modified optking")
    n_int = oMolsys.num_intcos
    # compute projection matrix = G G^-1
    G = oMolsys.wilson_g_mat()
    G_inv = symm_mat_inv(G, redundant=True)

    logger.debug(print_mat_string(G, title="G matrix\n"))
    logger.debug(print_mat_string(G_inv, title="Inverse G matrix\n"))

    P_prime = np.dot(G, G_inv)
    # logger.debug("\tProjection matrix for redundancies.\n\n" + printMatString(Pprime))
    # Add constraints to projection matrix
    C = oMolsys.constraint_matrix  # returns None, if aren't any

    if C is not None:
        logger.debug("Adding constraints for projection.\n" + print_mat_string(C))
        CPC = np.zeros((n_int, n_int))
        CPC[:, :] = np.dot(C, np.dot(P_prime, C))
        CPCInv = symm_mat_inv(CPC, redundant=True)
        P = np.zeros((n_int, n_int))
        P[:, :] = P_prime - np.dot(P_prime, np.dot(C, np.dot(CPCInv, np.dot(C, P_prime))))
    else:
        P = P_prime

    logger.debug(print_mat_string(P, title="Projection Matrix"))

    # Project redundancies out of forces.
    # fq~ = self fq
    projected_fq = P.dot(fq.T)

    # if op.Params.print_lvl >= 3:
    logger.debug("\n\tInternal forces in au, after projection of redundancies"
                 + " and constraints.\n" + print_array_string(fq))
    # Project redundancies out of Hessian matrix.
    # Peng, Ayala, Schlegel, JCC 1996 give H -> PHP + 1000(1-self)
    # The second term appears unnecessary and sometimes messes up Hessian updating.

    projected_hess = P.dot(H).dot(P)
    # tempMat = np.dot(H, P)
    # H[:, :] = np.dot(P, tempMat)
    # for i in range(dim)
    #    H[i,i] += 1000 * (1.0 - self[i,i])
    # for i in range(dim)
    #    for j in range(i):
    #        H[j,i] = H[i,j] = H[i,j] + 1000 * (1.0 - self[i,j])
    logger.debug("Projected (PHP) Hessian matrix\n" + print_mat_string(projected_hess))
    return projected_fq, projected_hess


def apply_fixed_forces(oMolsys, fq, H, stepNumber):
    logger = logging.getLogger(__name__)
    x = oMolsys.geom
    for iF, F in enumerate(oMolsys.fragments):
        for i, intco in enumerate(F.intcos):
            if intco.fixed:
                # TODO we may need to add iF to the location to get unique locations
                # for each fragment
                location = oMolsys.frag_1st_intco(iF) + i
                val = intco.q(x)
                eqVal = intco.fixed_eq_val

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
#    U = np.zeros((3 * nAtom, 3 * nAtom) )
#    for i in range (0, (3 * nAtom)):
#        U[i][i] = 1 / sqrt(masses[atom - 1])
#        if (i % 3 == 0):
#            nAtom += 1
#    return U
# """


def hessian_to_internals(H, oMolsys, g_x=None):
    """ converts the hessian from cartesian coordinates into internal coordinates 
    
    Parameters
    ----------
    H : ndarray
        Hessian in cartesians
    oMolsys : molecular system
    g_x : cartesian gradient
        Not working. 2nd derivative terms not implemented for dimer coordinates
    Returns
    -------
    Hq : ndarray
    """
    logger = logging.getLogger(__name__)
    logger.info("Converting Hessian from cartesians to internals.\n")
    B = oMolsys.wilson_b_mat()
    G = np.dot(B, B.T)
    # geom = oMolsys.geom

    Ginv = symm_mat_inv(G, redundant=True)
    Atranspose = np.dot(Ginv, B)

    Hworking = H.copy()
    if g_x is None:  # atom_a^t Hxy atom_a
        logger.info("Neglecting force/atom_b-matrix derivative term, only correct at"
                    + "stationary points.\n")
    """
    else:  # atom_a^t (Hxy - Kxy) atom_a;    K_xy = sum_q ( grad_q[I] d^2(q_I)/(dx dy) )
        logger.info("Including force/atom_b-matrix derivative term.\n")

        g_q = np.dot(Atranspose, g_x)
        Ncart = 3 * oMolsys.natom
        dq2dx2 = np.zeros((Ncart, Ncart))  # should be cart x cart for fragment ?

        for I, q in enumerate(oMolsys.intcos):
            dq2dx2[:] = 0
            q.dq2dx2(geom, dq2dx2)  # d^2(q_I)/ dx_i dx_j

            for a in range(Ncart):
                for b in range(Ncart):
                    # adjust indices for multiple fragments
                    Hworking[a, b] -= g_q[I] * dq2dx2[a, b] 
    """

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

    GM = wilson_g_mat(intcos, geom, masses)
    GM_root = symm_mat_root(GM)
    HqM = np.dot(np.dot(GM_root, Hq), GM_root.T)
    
    return HqM
"""


def hessian_to_cartesians(Hint, oMolsys, masses=None, g_q=None):
    """

    Parameters
    ----------
    Hint
    oMolsys : molsys.oMolsys
    masses
    g_q

    Returns
    -------

    """
    logger = logging.getLogger(__name__)
    logger.info("Converting Hessian from internals to cartesians.\n")

    B = oMolsys.wilson_b_mat(masses=masses)
    Hxy = np.dot(B.T, np.dot(Hint, B))

    if g_q is None:  # Hxy =  atom_b^t Hij atom_b
        logger.info("Neglecting force/atom_b-matrix derivative term, only correct at"
                    + "stationary points.\n")
    else:  # Hxy += dE/dq_I d2(q_I)/dxdy
        logger.info("Including force/atom_b-matrix derivative term.\n")
        Ncart = 3 * len(oMolsys.geom)

        dq2dx2 = np.zeros((Ncart, Ncart))  # should be cart x cart for fragment ?
        for I, q in enumerate(oMolsys.intcos):
            dq2dx2[:] = 0
            q.dq2dx2(oMolsys.geom, dq2dx2)

            for a in range(Ncart):
                for b in range(Ncart):
                    Hxy[a, b] += g_q[I] * dq2dx2[a, b]

    return Hxy


def tors_contains_bend(b, t):
    """

    Parameters
    ----------
    b: bend.Bend
    t tors.Tors

    Returns
    -------
    bool?
    """
    return (b.atoms in [t.atoms[0:3], list(reversed(t.atoms[0:3])), t.atoms[1:4],
                        list(reversed(t.atoms[1:4]))])


def remove_old_now_linear_bend(atoms, intcos):
    """ For given bend [atom_a,atom_b,connectivity_mat], remove any regular bends as well as any
    torsions which contain it
    """
    logger = logging.getLogger(__name__)
    b = bend.Bend(atoms[0], atoms[1], atoms[2])
    logger.info("Removing Old Linear Bend")
    logger.info(str(b) + '\n')
    intcos[:] = [I for I in intcos if I != b]
    intcos[:] = [
        I for I in intcos if not (isinstance(I, tors.Tors) and tors_contains_bend(b, I))
    ]
