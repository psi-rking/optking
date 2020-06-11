"""
A simple set of functions to compute LJ energies and gradients.
"""

import numpy as np


def calc_energy_and_gradient(positions, sigma, epsilon, do_gradient=True):
    r"""
    Computes the energy and gradient of a expression in the form
    V_{ij} = 4 \epsilon [ (sigma / r) ^ 12 - (sigma / r)^6]
    """

    # Holds the energy and the energy gradient
    E = 0.0
    if do_gradient:
        gradient = np.zeros((positions.shape[0], 3))

    sigma6 = sigma**6
    sigma12 = sigma6**2

    # Double loop over all particles
    for i in range(0, positions.shape[0] - 1):
        jvals = positions[(i + 1):]

        dr = jvals - positions[i]

        r = np.einsum("ij,ij->i", dr, dr)
        r6 = np.power(r, 6)
        r12 = np.power(r6, 2)

        E += np.sum((sigma12 / r12) - (sigma6 / r6))
        if do_gradient:
            g = 12 * r12 - 6 * r6
            g *= dr
            gradient[i] += -np.sum(g)
            gradient[i:] += g

    E *= 4.0 * epsilon

    if do_gradient:
        gradient *= 4.0 * epsilon
        return E, gradient
    else:
        return E
