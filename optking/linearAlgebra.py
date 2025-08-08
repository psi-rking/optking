import logging
from typing import Tuple

import numpy as np
from numpy.linalg import LinAlgError

from .exceptions import OptError
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
    try:
        return np.sqrt(np.mean(V**2))
    except FloatingPointError as error:
        print(V)
        raise error


def sign_of_double(d):
    if d > 0:
        return 1
    elif d < 0:
        return -1
    else:
        return 0


# Returns eigenvectors as rows?
def symm_mat_eig(mat) -> Tuple[np.ndarray, np.ndarray]:
    try:
        evals, evects = np.linalg.eigh(mat)
        if abs(min(evects[:, 0])) > abs(max(evects[:, 0])):
            evects[:, 0] *= -1.0
    except np.linalg.LinAlgError:
        raise OptError("symm_mat_eig: could not compute eigenvectors")
        # could be ALG_FAIL ?
    evects = evects.T
    return evals, evects


def lowest_eigenvector_symm_mat(mat) -> np.ndarray:
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
    except np.linalg.LinAlgError:
        raise OptError("symm_mat_eig: could not compute eigenvectors")
    return evects[:, 0]


def asymm_mat_eig(mat) -> Tuple[np.ndarray, np.ndarray]:
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
    except np.linalg.LinAlgError as e:
        raise OptError("asymm_mat_eig: could not compute eigenvectors") from e

    idx = np.argsort(evals)
    evals = evals[idx]
    evects = evects[:, idx]

    return evals.real, evects.real.T


def symm_mat_inv(A, redundant=False, threshold=1.0e-8) -> np.ndarray:
    """
    Return the inverse of a real, symmetric matrix.

    Parameters
    ----------
    A : np.ndarray
    redundant : bool
        allow generalized inverse
    threshold : float
        specifies how small of singular values to invert

    Returns
    -------
    np.ndarray

    """

    dim = A.shape[0]
    if dim == 0:
        return np.zeros((0, 0))

    try:
        if redundant:
            try:
                evals, evects = np.linalg.eigh(A)
                logger.debug("Eigenvalues for matrix to invert\n%s", evals)
            except LinAlgError:
                raise OptError("symm_mat_inv: could not compute eigenvectors")

            absEvals = np.abs(evals)
            # numpy uses a relative size comparison anything less than rcond * largest val
            # is zeroed out. Therefore larger values of rcond tighten the critera.
            # We want a `threshold` (params.linear_algebra_tol) to tighten the criteria
            # compute rcond such that any eigenvalues smaller than threshold are zeroed
            rcond = threshold / np.max(absEvals)

            # logger.debug("Singular | values | > %8.3e will be inverted." % threshold)
            val = np.min(absEvals[absEvals > threshold])
            if val < 1e-6:
                logger.warning("Inverting a small eigenvalue. System may include redundancies")
                logger.warning("Smallest inverted value is %8.3e." % val)

            return np.linalg.pinv(A, rcond)

        else:
            return np.linalg.inv(A)

    except LinAlgError:
        raise OptError("symmMatrixInv: could not compute eigenvectors")
        # could be LinAlgError?


def symm_mat_root(A, inverse=None, threshold=1e-10) -> np.ndarray:
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

    evals[np.abs(evals) < 10 * threshold] = 0.0
    evects[np.abs(evects) < 10 * threshold] = 0.0

    if inverse:
        evals = 1 / evals

    root_matrix = np.diagflat(np.sqrt(evals))
    return evects @ root_matrix @ evects.T
