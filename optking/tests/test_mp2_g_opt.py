import psi4
import optking
from .utils import utils


def test_mp2_h2o(check_iter):
    h2o = psi4.geometry(
        """
        O
        H 1 1.0
        H 1 1.0 2 106.0
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "6-31G**",
        "reference": "rhf",
        "d_convergence": 9,
        "e_convergence": 9,
        "mp2_type": "conv",
        "max_energy_g_convergence": 7,
    }
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("mp2")

    this_nucenergy = result["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]  # TEST
    this_mp2 = result["energies"][-1]  # TEST
    REF_nucenergy = 9.1622581908184  # TEST
    REF_mp2 = -76.2224486598878  # TEST
    assert psi4.compare_values(REF_nucenergy, this_nucenergy, 3, "Nuclear repulsion energy")  # TEST
    assert psi4.compare_values(REF_mp2, this_mp2, 6, "CONV MP2 energy")  # TEST
    utils.compare_iterations(result, 5, check_iter)
