import logging
from math import fabs, sqrt

import numpy as np
from numpy.linalg import LinAlgError

from .exceptions import OptError
from .printTools import print_array_string
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")
#  Linear algebra routines. #


def norm(V):
    return np.linalg.norm(V)


def abs_max(V):
    return max(abs(elem) for elem in V)


def abs_min(V):
    return min(abs(elem) for elem in V)


def rms(V):
    return np.sqrt(np.mean(V ** 2))


def sign_of_double(d):
    if d > 0:
        return 1
    elif d < 0:
        return -1
    else:
        return 0


# Returns eigenvectors as rows?
def symm_mat_eig(mat):
    try:
        evals, evects = np.linalg.eigh(mat)
        if abs(min(evects[:, 0])) > abs(max(evects[:, 0])):
            evects[:, 0] *= -1.0
    except:
        raise OptError("symm_mat_eig: could not compute eigenvectors")
        # could be ALG_FAIL ?
    evects = evects.T
    return evals, evects


def lowest_eigenvector_symm_mat(mat):
    """Returns eigenvector with lowest eigenvalues; makes the largest
        magnitude element positive.

    Parameters
    ----------
    mat:  np.ndarray

    Returns
    -------
    np.ndarray
        eigenvector for lowest eigenvalue

    """

    try:
        evals, evects = np.linalg.eigh(mat)
        if abs(min(evects[:, 0])) > abs(max(evects[:, 0])):
            evects[:, 0] *= -1.0
    except:
        raise OptError("symm_mat_eig: could not compute eigenvectors")
    return evects[:, 0]


def asymm_mat_eig(mat):
    """Compute the eigenvalues and right eigenvectors of a square array.
    Wraps numpy.linalg.eig to sort eigenvalues, put eigenvectors in rows, and suppress complex.

    Parameters
    ----------
    mat : ndarray
        (n, n) Square matrix to diagonalize.

    Returns
    -------
    ndarray, ndarray
        (n, ), (n, n) sorted eigenvalues and normalized corresponding eigenvectors in rows.

    Raises
    ------
    OptError
        When eigenvalue computation does not converge.

    """
    try:
        evals, evects = np.linalg.eig(mat)
    except np.LinAlgError as e:
        raise OptError("asymm_mat_eig: could not compute eigenvectors") from e

    idx = np.argsort(evals)
    evals = evals[idx]
    evects = evects[:, idx]

    return evals.real, evects.real.T


def symm_mat_inv(A, redundant=False, redundant_eval_tol=1.0e-10):
    """
    Return the inverse of a real, symmetric matrix.

    Parameters
    ----------
    A : np.ndarray
    redundant : bool
        allow generalized inverse
    redundant_eval_tol : float

    Returns
    -------
    np.ndarray

    """

    dim = A.shape[0]
    if dim == 0:
        return np.zeros((0, 0))
    det = 1.0

    try:
        evals, evects = symm_mat_eig(A)
    except LinAlgError:
        raise OptError("symmMatrixInv: could not compute eigenvectors")
        # could be LinAlgError?

    logger.debug("Eigenvalues for symm matrix inversion")
    logger.debug("\n" + print_array_string(evals, form=":12.3e"))

    for i in range(dim):
        det *= evals[i]

    if not redundant and fabs(det) < 1e-10:
        raise OptError("symmMatrixInv: non-generalized inverse failed; very small determinant")
        # could be LinAlgError?

    diagInv = np.zeros((dim, dim))

    if redundant:
        for i in range(dim):
            if fabs(evals[i]) > redundant_eval_tol:
                diagInv[i, i] = 1.0 / evals[i]
    else:
        for i in range(dim):
            diagInv[i, i] = 1.0 / evals[i]

    # A^-1 = P^t d^-1 P
    tmpMat = np.dot(diagInv, evects)
    AInv = np.dot(evects.T, tmpMat)
    return AInv


def symm_mat_root(A, Inverse=None):
    """
    Compute A^(1/2) for a positive-definite matrix

    Parameters
    ----------
    A : np.ndarray
    Inverse : bool
        calculate A^(-1/2)

    Returns
    -------
    np.ndarray

    """
    try:
        evals, evects = np.linalg.eigh(A)
        # Eigenvectors of A are in columns of evects
        # Evals in ascending order
    except LinAlgError:
        raise OptError("symm_mat_root: could not compute eigenvectors")

    evals[np.abs(evals) < 10 * np.finfo(np.float).resolution] = 0.0
    evects[np.abs(evects) < 10 * np.finfo(np.float).resolution] = 0.0

    rootMatrix = np.zeros((len(evals), len(evals)))
    if Inverse:
        for i in range(0, len(evals)):
            evals[i] = 1 / evals[i]

    for i in range(0, len(evals)):
        rootMatrix[i][i] = sqrt(evals[i])

    A = np.dot(evects, np.dot(rootMatrix, evects.T))

    return A
