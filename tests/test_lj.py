"""
Tests the LJ functions
"""
import optking
import pytest
import numpy as np

@pytest.mark.parametrize("R,ref",
    [(1.5, 415.21017761),
     (2.0, -2.34083462),
     (2.5, -0.19329605),
     (3.0, -0.02191777),
     (5.0, -0.0000477756)]
)
def test_lj_energy(R, ref):
    positions = np.array([[0, 0, -R/2.0], [0, 0, R/2.0]])
    energy = optking.lj_functions.calc_energy_and_gradient(positions, 3.0, 4.0, do_gradient=False)

    if not pytest.approx(ref) == energy:
        raise ValueError("test_lj_energy for R=%.2f did not match reference (comp = %12.10f, ref = %12.10f)." % (R, energy, ref))
