"""
A simple set of functions to compute LJ energies and gradients.
"""

import numpy as np
from itertools import combinations


def calc_energy_and_gradient(positions, sigma, epsilon, do_gradient=True):
    r"""
    Computes the energy and gradient of a expression in the form
    V_{ij} = 4 \epsilon [ (sigma / r) ^ 12 - (sigma / r)^6]
    """

    Natom = positions.shape[0]
    # Holds the energy and the energy gradient
    E = 0.0
    if do_gradient:
        gradient = np.zeros((Natom, 3))

    sigma6 = sigma**6
    sigma12 = sigma6**2

    # Double loop over all particles
    for I,J in combinations(range(Natom), 2):
        vIJ = positions[J] - positions[I]
        r = np.linalg.norm(vIJ)
        vIJ[:] = vIJ / r
        r6 = np.power(r, 6)
        r12 = np.power(r6, 2)
        E += sigma12/r12 - sigma6/r6

        if do_gradient:
            dVdr = - 12*sigma12/(r12*r) + 6*sigma6/(r6*r)
            for xyz in range(3):
                gradient[I,xyz] -=  dVdr * vIJ[xyz]
                gradient[J,xyz] +=  dVdr * vIJ[xyz]

    E *= 4.0 * epsilon

    if do_gradient:
        gradient = 4.0 * epsilon * gradient.reshape(3*Natom)
        return E, gradient
    else:
        return E
