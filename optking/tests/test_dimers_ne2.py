import psi4
import optking
import numpy as np
import pytest
from .utils import utils

# Optimize neon dimer
MP2minEnergy = -257.4109749


@pytest.mark.dimers
def test_dimers_ne2_long(check_iter):
    # Test from long distance start.
    ne2 = psi4.geometry(
        """
      0 1
      Ne  0.0  0.0  0.0
      --
      0 1
      Ne  4.0  0.0  0.0
      nocom
    """
    )

    psi4_options = {
        "basis": "aug-cc-pvdz",
        "geom_maxiter": 30,
        "frag_mode": "MULTI",
        "frag_ref_atoms": [[[1]], [[1]]],  # atoms for reference point on frag1  # atoms for reference point on frag2
        "g_convergence": "gau_verytight",
    }
    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("mp2")
    E = json_output["energies"][-1]
    assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from afar")
    utils.compare_iterations(json_output, 16, check_iter)

@pytest.mark.dimers
def test_dimers_ne2_short(check_iter):
    # Test from short distance start.
    ne2 = psi4.geometry(
        """
      0 1
      Ne  0.0  0.0  0.0
      --
      0 1
      Ne  2.5  0.0  0.0
      nocom
    """
    )

    psi4_options = {
        "basis": "aug-cc-pvdz",
        "geom_maxiter": 30,
        "frag_mode": "MULTI",
        "frag_ref_atoms": [[[1]], [[1]]],  # atoms for reference point on frag1  # atoms for reference point on frag2
        "g_convergence": "gau_verytight",
    }
    psi4.core.clean_options()
    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("mp2")
    E = json_output["energies"][-1]
    assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from close")
    utils.compare_iterations(json_output, 17, check_iter)

@pytest.mark.dimers
def test_dimers_ne2_auto(check_iter):  # auto reference pt. creation
    ne2 = psi4.geometry(
        """
      0 1
      Ne  0.0  0.0  0.0
      --
      0 1
      Ne  4.0  0.0  0.0
      nocom
    """
    )

    psi4.core.clean_options()
    psi4_options = {"basis": "aug-cc-pvdz", "geom_maxiter": 30, "frag_mode": "MULTI", "g_convergence": "gau_verytight"}
    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("mp2")
    E = json_output["energies"][-1]
    assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from afar, auto")
    utils.compare_iterations(json_output, 16, check_iter)
