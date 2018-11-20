# Functions to compute properties of 3d vectors, including angles,
# torsions, out-of-plane angles.  Several return False if the operation
# cannot be completed numerically, as for example a torsion in which 3
# points are collinear.
import logging
import numpy as np
from math import fabs, sin, acos, asin, fsum

from . import optparams as op
from .exceptions import AlgError, OptError

# a couple of obscure parameters used in torsion computation:
#  phi_lim = op.Params.v3d_tors_angle_lim
#  tors_cos_tol = op.Params.v3d_tors_cos_tol

DOT_PARALLEL_LIMIT = 1.e-10


def norm(v):
    return np.linalg.norm(v)
    # return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def dot(v1, v2, length=None):
    """
    Asks numpy to perform dot prodct, with the option (not used?) to return part of
    the resulting vector
    """
    if length is None:
        return np.dot(v1, v2)
        # return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    else:
        return fsum([v1[i] * v2[i] for i in range(length)])
        # I'll have to look closer before removing this


def dist(v1, v2):
    return np.linalg.norm(v1-v2)
    # return sqrt((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2 + (v2[2] - v1[2])**2)


def normalize(v1, Rmin=1.0e-8, Rmax=1.0e15):
    """
    Normalize vector in place.  If norm exceeds thresholds, don't normalize and return False..
    """
    n = norm(v1)
    if n < Rmin or n > Rmax:
        raise AlgError("Could not normalize vector. Vector norm beyond tolerance")
    else:
        v1 /= n


#def axpy(a, X, Y):
#    Z = np.zeros(Y.shape, float)
#    Z = a * X + Y
#    return Z


# Compute and return normalized vector from point p1 to point p2.
# If norm is too small, don't normalize and return check as False.
def eAB(p1, p2):
    eAB = p2 - p1
    normalize(eAB)
    return eAB


# Compute and return cross-product.
def cross(u, v):
    # X = np.zeros(3, float)
    X = np.cross(u, v)
    # X[0] = u[1] * v[2] - u[2] * v[1]
    # X[1] = -u[0] * v[2] + u[2] * v[0]
    # X[2] = u[0] * v[1] - u[1] * v[0]
    return X


def are_parallel(u, v):
    """ Determines if two vectors are parallel within tolerance (1e-10)"""
    if fabs(dot(u, v) - 1.0e0) < DOT_PARALLEL_LIMIT:
        return True
    else:
        return False


def are_antiparallel(u, v):
    """ Determines if two vectors are antiparallel within tolerance (1e-10)"""
    if fabs(dot(u, v) + 1.0e0) < DOT_PARALLEL_LIMIT:
        return True
    else:
        return False


def are_parallel_or_antiparallel(u, v):
    """
    Determines if two vectors are parallel and or antiparallal

    Returns
    -------

    boolean
        if vectors are either parallel or antiparallel
    """

    return are_parallel(u, v) or are_antiparallel(u, v)


def angle(A, B, C, tol=1.0e-14):
    """ Compute and return angle in radians A-B-C (between vector B->A and vector B->C)
    If points are absurdly close or far apart, returns False

    Parameters
    ----------
    A : int
        number of atom in fragment system. uses 1 indexing
    B : int
    C : int

    Returns
    -------
    float
        angle in radians
    """
    logger = logging.getLogger(__name__)
    try:
        eBA = eAB(B, A)
    except AlgError as error:
        logger.warning("Could not normalize eBA in angle()\n")
        raise optExcpetions.AlgError from error

    try:
        eBC = eAB(B, C)
    except AlgError as error:
        logger.warning("Could not normalize eBC in angle()\n")
        raise AlgError from error

    return _calc_angle(eBA, eBC, tol)


def _calc_angle(vec_1, vec_2, tol=1.0e-14):
    """
    Computes and returns angle in radians A-B_B (between vector B->A and vector B->C

    Should only be called by tors or angle. Error checking and vector creation
    is performed in angle() or tors() previously

    Paramters
    ---------
    vec_1 : ndarray
        first vector of an angle
    vec_2 : ndarray
        second vector on an angle
    tol : float
        nearness of cos to 1/-1 to set angle 0/pi.
    """

    dotprod = dot(vec_1, vec_2)

    if dotprod > 1.0 - tol:
        phi = 0.0
    elif dotprod < -1.0 + tol:
        phi = acos(-1.0)
    else:
        phi = acos(dotprod)

    return phi


def tors(A, B, C, D):
    """ Compute and return angle in dihedral angle in radians A-B-C-D
    Raises AlgError exception if bond angles are too large for good torsion definition
    """
    logger = logging.getLogger(__name__)
    phi_lim = op.Params.v3d_tors_angle_lim
    tors_cos_tol = op.Params.v3d_tors_cos_tol

    # Form e vectors
    try:
        EBA = eAB(B, A)
        EAB = -1 * EBA
    except AlgError as error:
        logger.warning("Could not normalize %d, %d vector in tors()\n" % (str(A), str(B)))
        raise
    try:
        EBC = eAB(B, C)
    except AlgError as error:
        logger.warning("Could not normalize %d, %d vector in tors()\n" % (str(B), str(C)))
        raise
    try:
        ECB = eAB(C, B)
        EBC = -1 * ECB
    except AlgError as error:
        logger.warning("Could not normalize %d, %d vector in tors()\n" % (str(C), str(D)))
        raise
    try:
        ECD = eAB(C, D)
    except AlgError as error:
        logger.warning("Could not normalize %d, %d vector in tors()\n" % (str(C), str(D)))
        raise

    # Compute bond angles
    phi_123 = _calc_angle(EBA, EBC)
    phi_234 = _calc_angle(ECB, ECD)

    up_lim = acos(-1) - phi_lim

    if phi_123 < phi_lim or phi_123 > up_lim or phi_234 < phi_lim or phi_234 > up_lim:
        raise AlgError("Tors angle for %d, %d, %d, %d is too large for good "
                                    + "definition" % (str(A), str(B), str(C), str(D)))

    tmp = cross(EAB, EBC)
    tmp2 = cross(EBC, ECD)
    tval = dot(tmp, tmp2) / (sin(phi_123) * sin(phi_234))

    if tval >= 1.0 - tors_cos_tol:  # accounts for numerical leaking out of range
        tau = 0.0
    elif tval <= -1.0 + tors_cos_tol:
        tau = acos(-1)
    else:
        tau = acos(tval)

    # determine sign of torsion ; this convention matches Wilson, Decius and Cross
    if tau != acos(-1):  # no torsion will get value of -pi; Range is (-pi,pi].
        tmp = cross(EBC, ECD)
        tval = dot(EAB, tmp)
        if tval < 0:
            tau *= -1

    return tau


def oofp(A, B, C, D):
    """ Compute and return angle in dihedral angle in radians A-B-C-D
    returns false if bond angles are too large for good torsion definition
    """
    logger = logging.getLogger(__name__)
    try:
        eBA = eAB(B, A)
    except AlgError as error:
        logger.warning("Could not normalize %d, %d vector in tors()\n" % (str(B), str(A)))
        raise
    try:
        eBC = eAB(B, C)
    except AlgError as error:
        logger.warning("Could not normalize %d, %d vector in tors()\n" % (str(B), str(C)))
        raise
    try:
        eBD = eAB(B, D)
    except AlgError as error:
        logger.warning("Could not normalize %d, %d vector in tors()\n" % (str(B), str(D)))
        raise

    phi_CBD = _calc_angle(eBC, eBD)

    # This shouldn't happen unless angle B-C-D -> 0,
    if sin(phi_CBD) < op.Params.v3d_tors_cos_tol:  # reusing parameter
        raise AlgError("Angle: %d, %d, %d is to close to zero in oofp\n"
                                    % (str(C), str(B), str(D)))

    dotprod = dot(cross(eBC, eBD), eBA) / sin(phi_CBD)

    if dotprod > 1.0:
        tau = acos(-1)
    elif dotprod < -1.0:
        tau = -1 * acos(-1)
    else:
        tau = asin(dotprod)
    return tau
