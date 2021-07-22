import logging
from math import sqrt

import numpy as np

from . import bend, oofp
from . import optparams as op
from . import tors
from .linearAlgebra import symm_mat_inv, symm_mat_root
from .printTools import print_array_string, print_mat_string

# Some of these functions act on an arbitrary list of simple internals,
# geometry etc. that may or may not be in a molecular system.

def q_values(intcos, geom):
    # available for simple intco lists
    vals = [intco.q(geom) for intco in intcos]
    return np.asarray(vals)


def Bmat(intcos, geom, masses=None):
    # Allocate memory for full system.
    # Returns mass-weighted Bmatrix if masses are supplied.
    # available for simple intco lists
    Nint = len(intcos)
    B = np.zeros((Nint, 3 * len(geom)))

    for i, intco in enumerate(intcos):
        intco.DqDx(geom, B[i])

    if type(masses) is np.ndarray:
        sqrtm = np.array([np.repeat(np.sqrt(masses), 3)] * Nint, float)
        B[:] = np.divide(B, sqrtm)

    return B


def tors_contains_bend(b, t):
    return b.atoms in [
        t.atoms[0:3],
        list(reversed(t.atoms[0:3])),
        t.atoms[1:4],
        list(reversed(t.atoms[1:4])),
    ]


def remove_old_now_linear_bend(atoms, intcos):
    """For given bend [A,B,C], remove any regular bends as well as any torsions
    which contain it
    """
    logger = logging.getLogger(__name__)
    b = bend.Bend(atoms[0], atoms[1], atoms[2])
    logger.info("Removing Old Linear Bend")
    logger.info(str(b) + "\n")
    intcos[:] = [coord for coord in intcos if coord != b]
    intcos[:] = [coord for coord in intcos if not (isinstance(coord, tors.Tors) and tors_contains_bend(b, coord))]
