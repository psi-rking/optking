#! Linesearch tests
# memory 8gb
import optking
import psi4
import pytest
from .utils import utils

_schver = 2 if utils.psi4_runs_v2_qcschema(psi4.__version__) else 1

refnucenergy = 41.670589  # Eh
refenergy = -1053.880393  # Eh

@pytest.mark.long
def test_linesearch(check_iter):
    Ar2 = psi4.geometry(
        """
      Ar
      Ar 1 5.0
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "cc-pvdz",
        "d_convergence": 10,
        "geom_maxiter": 60,
        "g_convergence": "gau_tight",
        "step_type": "SD",
    }

    psi4.set_options(psi4_options)

    # "linesearch" is not currrently recognized by psi4 read_options.
    json_output = optking.optimize_psi4("mp2", **{"linesearch": True})
    print(json_output)
    if _schver == 1:
        E = json_output["energies"][-1]
        nucenergy = json_output["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]
    elif _schver == 2:
        E = json_output["trajectory_properties"][-1]["return_energy"]
        nucenergy = json_output["trajectory_results"][-1]["properties"]["nuclear_repulsion_energy"]
    assert psi4.compare_values(nucenergy, nucenergy, 3, "Nuclear repulsion energy")  # TEST
    assert psi4.compare_values(refenergy, E, 1, "Reference energy")  # TEST
    # utils.compare_iterations(json_output, 25, check_iter)
