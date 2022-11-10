"""
A simple set of functions to compute LJ energies and gradients.
"""

from itertools import combinations

import numpy as np


def calc_energy_and_gradient(positions, sigma, epsilon, do_gradient=True):
    r"""
    Computes the energy and gradient of a expression in the form
    V_{ij} = 4 \epsilon [ (sigma / r) ^ 12 - (sigma / r)^6]
    """

    natom = positions.shape[0]
    # Holds the energy and the energy gradient
    E = 0.0

    gradient = np.zeros((natom, 3))

    sigma6 = sigma**6
    sigma12 = sigma6**2

    # Double loop over all particles
    for i, j in combinations(range(natom), 2):
        v_ij = positions[j] - positions[i]
        r = np.linalg.norm(v_ij)
        v_ij[:] = v_ij / r
        r6 = np.power(r, 6)
        r12 = np.power(r6, 2)
        E += sigma12 / r12 - sigma6 / r6

        if do_gradient:
            dVdr = -12 * sigma12 / (r12 * r) + 6 * sigma6 / (r6 * r)
            for xyz in range(3):
                gradient[i, xyz] -= dVdr * v_ij[xyz]
                gradient[j, xyz] += dVdr * v_ij[xyz]

    E *= 4.0 * epsilon

    if do_gradient:
        gradient = 4.0 * epsilon * gradient.reshape(3 * natom)
        return E, gradient
    else:
        return E
