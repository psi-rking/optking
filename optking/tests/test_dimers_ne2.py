import psi4
import optking
import numpy as np
import pytest
from .utils import utils

# Optimized neon dimer
MP2minEnergy = -257.4109749


#! (Ne)_2 with interfrag coordinates, specifying ref atoms, from long-range
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
        "frag_ref_atoms": [[[1]], [[2]]],  # reference point atoms, numbering now total
        "g_convergence": "gau_verytight",
    }
    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("mp2")
    E = json_output["energies"][-1]
    assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from afar")
    utils.compare_iterations(json_output, 14, check_iter)


#! (Ne)_2 with interfrag coordinates, specifying ref atoms, from short-range
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
        "frag_ref_atoms": [[[1]], [[2]]],  # reference point atoms, numbering now total
        "g_convergence": "gau_verytight",
    }
    psi4.core.clean_options()
    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("mp2")
    E = json_output["energies"][-1]
    assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from close")
    utils.compare_iterations(json_output, 15, check_iter)


#! (Ne)_2 with interfrag coordinates, auto-generated ref atoms
@pytest.mark.dimers
def test_dimers_ne2_auto(check_iter):  # auto reference pt. creation
    ne2 = psi4.geometry(
        """
      0 1
      Ne  0.0  0.0  0.0
      --
      0 1
      Ne  3.0  0.0  0.0
      nocom
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "aug-cc-pvdz",
        "geom_maxiter": 30,
        "frag_mode": "MULTI",
        "g_convergence": "gau_verytight",
    }
    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("mp2")
    E = json_output["energies"][-1]
    assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from afar, auto")
    utils.compare_iterations(json_output, 13, check_iter)


#! (Ne)_2 with interfrag coordinates, using 1/R distance
@pytest.mark.dimers
def test_dimers_ne2_inverseR(check_iter):
    ne2 = psi4.geometry(
        """
      0 1
      Ne  0.0  0.0  0.0
      --
      0 1
      Ne  3.0  0.0  0.0
      nocom
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "aug-cc-pvdz",
        "geom_maxiter": 20,
        "frag_mode": "MULTI",
        "g_convergence": "gau_verytight",
        "frag_ref_atoms": [[[1]], [[2]]],  # atoms for reference points
        "interfrag_dist_inv": True,
        "test_B": True,
    }
    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("mp2")
    E = json_output["energies"][-1]
    assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from afar, auto")
    utils.compare_iterations(json_output, 10, check_iter)


#! (Ne)_2 with interfrag coordinates, using full user dict input
@pytest.mark.dimers
def test_dimers_ne2_dict():
    ne2 = psi4.geometry(
        """
      0 1
      Ne  0.0  0.0  0.0
      Ne  4.0  0.0  0.0
      nocom
    """
    )
    psi4.core.clean_options()
    psi4_options = {
        "basis": "aug-cc-pvdz",
        "geom_maxiter": 30,
        "frag_mode": "MULTI",
        "g_convergence": "gau_verytight",
    }

    dimer = {
        "Natoms per frag": [1, 1],
        "A Frag": 1,
        "A Ref Atoms": [[1]],
        "A Label": "Ne atom 1",
        "B Frag": 2,
        "B Ref Atoms": [[2]],
        "B Label": "Ne atom 2",
    }

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("mp2", interfrag_coords=str(dimer))
    E = json_output["energies"][-1]
    assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from afar, auto")
