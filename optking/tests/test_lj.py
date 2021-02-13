"""
Tests the LJ functions
"""
import optking
import pytest
import numpy as np

# Before 9-4-2020, these were in error because function expected
# sigma first, then epsilon.
@pytest.mark.parametrize(
    "R,ref",
    [
        (2.0, 1893.6914062),
        (3.0, 0.0000000),  # V = 0 at r = sigma
        (3.0 * np.power(2, 1 / 6), -4.0),  # Vmin=-epsilon at r = 2^(1/6)*sigma
        (4.0, -2.3408346),
        (6.0, -0.2460937500),
    ],
)
def test_lj_energy(R, ref):
    positions = np.array([[0, 0, -R / 2.0], [0, 0, R / 2.0]])
    energy = optking.lj_functions.calc_energy_and_gradient(positions, 3.0, 4.0, do_gradient=False)

    if not pytest.approx(ref) == energy:
        raise ValueError(
            "test_lj_energy for R=%.2f did not match reference (comp = %12.10f, ref = %12.10f)." % (R, energy, ref)
        )
