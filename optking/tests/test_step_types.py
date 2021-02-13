import psi4
import optking
import pytest

finalEnergy = -76.05776970  # TEST

#! SCF CC-PVTZ geometry optimzation, with Z-matrix input
@pytest.mark.parametrize("option, expected", [("RFO", finalEnergy), ("NR", finalEnergy), ("SD", finalEnergy)])
def test_h2o_rfo(option, expected):

    h2o = psi4.geometry(
        """
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "cc-pvtz",
        "e_convergence": "10",
        "d_convergence": "10",
        "scf_type": "pk",
        "max_energy_g_convergence": 7,
        "step_type": option,
    }
    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4("hf")

    E = json_output["energies"][-1]  # TEST
    assert psi4.compare_values(finalEnergy, E, 6, f"{option} Step Final Energy")  # TEST
