import logging
import json
import os
from pathlib import Path

import numpy as np
import qcelemental as qcel

from .bend import Bend
from .exceptions import OptError
from .printTools import print_mat_string
from .stre import Stre
from .tors import Tors
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")


def show(H: np.ndarray, oMolsys):
    """Print the Hessian in common spectroscopic units of [aJ/Ang^2], [aJ/deg^2] or [aJ/(Ang deg)]"""

    factors = np.zeros(oMolsys.num_intcos)
    cnt = -1
    for F in oMolsys._fragments:
        for I in F.intcos:
            cnt += 1
            factors[cnt] = I.q_show_factor
    for DI in oMolsys._dimer_intcos:
        for I in DI._pseudofrag._intcos:
            cnt += 1
            factors[cnt] = I.q_show_factor

    factors_inv = np.divide(1.0, factors)
    scaled_H = np.einsum("i,ij,j->ij", factors_inv, H, factors_inv)
    scaled_H *= qcel.constants.hartree2aJ
    logger.info("Hessian in [aJ/Ang^2], [aJ/deg^2], etc.\n" + print_mat_string(scaled_H))


# def guess(intcos, geom, Z, connectivity=None, guessType="SIMPLE"):
def guess(oMolsys, connectivity=None, guessType="SIMPLE"):
    """Generates diagonal empirical Hessian in a.u.

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

    diag = []
    for F in oMolsys._fragments:
        if F.num_intcos:
            geom = F.geom
            Z = F.Z
            connectivity = F.connectivity_from_distances()
            for intco in F._intcos:
                diag.append(intco.diagonal_hessian_guess(geom, Z, connectivity, guessType))

    # Since the reference points might not even be at atomic positions, let's not worry
    # about implementing various options for the diagonal Hessian guess.
    for DI in oMolsys._dimer_intcos:
        vals = DI.q()
        for i, intco in enumerate(DI._pseudo_frag._intcos):
            if isinstance(intco, Stre):
                h = 0.007
                if intco.inverse:
                    h *= pow(1.0 / vals[i], 4)
                    # i should be 0=stretch
            elif isinstance(intco, Bend):
                h = 0.003
            elif isinstance(intco, Tors):
                h = 0.001
            else:
                h = 0.111
            diag.append(h)

    H = np.diagflat(np.asarray(diag))
    return H


def from_file(filename: Path):
    """Read user provided hessian from disk"""

    if not filename.exists():
        raise OptError(f"The specified file {filename} in `hessian_file` does not exist")

    with filename.open() as f:
        if ".json" == filename.suffix:
            result = json.load(f)
            hess = result["return_result"]
            ncart = 3 * len(result["molecule"]["symbols"])
        else:
            # 2D split. Cast everything to floats and convert back to 2D list from map.
            lines = f.readlines()
            hess = [list(map(float, line.split())) for line in lines[1:]]
            ncart = int(lines[0].split()[1])  # assumes FCMFINAL / CFOUR format

    try:
        H = np.array(hess, dtype=float)
        H = H.reshape(ncart, ncart)
    except (IndexError, ValueError, TypeError) as error:
        logger.error("Hessian should be 3Nx3N cartesian force constant matrix or provided by MolSSI Schema")
        logger.error(error)
        raise ValueError("Could not load hessian from disk") from error
    else:
        return H
