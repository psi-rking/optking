import logging

import numpy as np
import qcelemental as qcel

from .printTools import printMatString
# from bend import *


def show(H, intcos):
    """ Print the Hessian in common spectroscopic units of [aJ/Ang^2], [aJ/deg^2] or [aJ/(Ang deg)]
    """
    logger = logging.getLogger(__name__)
    factors = np.asarray([intco.qShowFactor for intco in intcos])
    factors_inv = np.divide(1.0, factors)
    scaled_H = np.einsum('i,ij,j->ij', factors_inv, H, factors_inv)
    scaled_H *= qcel.constants.hartree2aJ
    logger.info("Hessian in [aJ/Ang^2], etc.\n" + printMatString(scaled_H))


def guess(intcos, geom, Z, connectivity=False, guessType="SIMPLE"):
    """ Generates diagonal empirical Hessian in a.u.

    Parameters
    ----------
    intcos : list of Stre, Bend, Tors
    geom : ndarray
        cartesian geometry
    connectivity : ndarray, optional
        connectivity matrix
    guessType: str, optional
        the default is SIMPLE. other options: FISCHER, LINDH_SIMPLE, SCHLEGEL

    Notes
    -----
    such as
      Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
      Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
    """

    diag_hess = np.asarray([intco.diagonalHessianGuess(geom, Z, connectivity, guessType)
                            for intco in intcos])
    H = np.diagflat(diag_hess)

    return H
