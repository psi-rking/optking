import logging
from math import sqrt

import numpy as np

from . import bend
from . import tors
from . import log_name

# Some of these functions act on an arbitrary list of simple internals,
# geometry etc. that may or may not be in a molecular system.

logger = logging.getLogger(f"{log_name}{__name__}")


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
    tors_bends = [
        t.atoms[0:3],
        t.atoms[2::-1],
        t.atoms[1:4],
        t.atoms[4:0:-1],
    ]
    return b.atoms in tors_bends


def remove_old_now_linear_bend(atoms, intcos):
    """For given bend [A,B,C], remove any regular bends as well as any torsions
    which contain it
    """
    b = bend.Bend(atoms[0], atoms[1], atoms[2])
    logger.info("Removing Old Linear Bend")
    logger.info(str(b) + "\n")

    # linear bend check assumes that the bend we're trying to remove is the same definition
    # as the bend we're adding. This isn't the case when the bend goes through zero. In
    # this case we need to swap indices
    if b not in intcos:
        bend_alt = bend.Bend(atoms[0], atoms[2], atoms[1])
        if bend_alt in intcos:
            b = bend_alt

    intcos[:] = [coord for coord in intcos if coord != b]
    intcos[:] = [
        coord
        for coord in intcos
        if not (isinstance(coord, tors.Tors) and tors_contains_bend(b, coord))
    ]
