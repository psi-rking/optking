import psi4
import optking
import pytest

from .utils import utils

#! Optimization to 180 degree torsion from 120
@pytest.mark.parametrize("option, expected_steps", [("P_RFO", 14), ("RS_I_RFO", 13)])
def test_hooh_TS(check_iter, option, expected_steps):

    hooh = psi4.geometry(
        """
     0 1
     H
     O 1 0.95
     O 2 1.40 1 105.0
     H 3 0.95 2 105.0 1 120.0
    """
    )

    psi4.core.clean_options()
    psi4options = {
        "basis": "cc-pvdz",
        "geom_maxiter": 30,
        "opt_type": "TS",
        "scf_type": "pk",
        "docc": [5, 4],
        "intrafrag_step_limit": 0.1,
        "max_energy_g_convergence": 7,
    }
    psi4.set_options(psi4options)

    json_output = optking.optimize_psi4("hf", **{"step_type": option})

    E = json_output["energies"][-1]  # TEST
    # print( '{:15.10f}'.format(E) )
    C2H_TS_ENERGY = -150.7854114803  # TEST
    assert psi4.compare_values(C2H_TS_ENERGY, E, 6, "RHF Energy after optimization to C2H TS")  # TEST
    utils.compare_iterations(json_output, expected_steps, check_iter)


#! Optimization to 0 degree torsion from 100
@pytest.mark.parametrize("option, expected_steps", [("P_RFO", 21), ("RS_I_RFO", 11)])
def test_hooh_TS_zero(check_iter, option, expected_steps):

    hooh = psi4.geometry(
        """
     0 1
     H
     O 1 0.95
     O 2 1.40 1 105.0
     H 3 0.95 2 105.0 1 100.0
    """
    )

    psi4.core.clean_options()
    psi4options = {
        "basis": "cc-pvdz",
        "geom_maxiter": 40,
        "opt_type": "TS",
        "scf_type": "pk",
        "docc": [5, 4],
        "intrafrag_step_limit": 0.1,
        "max_energy_g_convergence": 7,
    }
    psi4.set_options(psi4options)

    json_output = optking.optimize_psi4("hf", **{"step_type": option})

    E = json_output["energies"][-1]  # TEST
    C2V_TS_ENERGY = -150.774009217562  # TEST
    assert psi4.compare_values(C2V_TS_ENERGY, E, 6, "RHF Energy after optimization to C2H TS")  # TEST
    utils.compare_iterations(json_output, expected_steps, check_iter)


def test_hooh_min(check_iter):
    hooh = psi4.geometry(
        """
     H
     O 1 0.95
     O 2 1.40 1 105.0
     H 3 0.95 2 105.0 1 100.0
    """
    )

    psi4.core.clean_options()
    psi4options = {
        "basis": "cc-pvdz",
        "geom_maxiter": 20,
        "opt_type": "min",
        "scf_type": "pk",
        "docc": [5, 4],
        "max_energy_g_convergence": 7,
    }
    psi4.set_options(psi4options)

    json_output = optking.optimize_psi4("hf")

    E = json_output["energies"][-1]  # TEST
    # print( '{:15.10f}'.format(E) )
    MIN_ENERGY = -150.7867668  # TEST
    assert psi4.compare_values(MIN_ENERGY, E, 6, "RHF Energy")  # TEST
    utils.compare_iterations(json_output, 7, check_iter)
