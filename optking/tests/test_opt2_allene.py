#! SCF DZ allene geometry optimization, with Cartesian input, first in c2v symmetry,
#! then in Cs symmetry from a starting point with a non-linear central bond angle.

import psi4
import optking
from .utils import utils

# import importlib


def test_opt2_allene(check_iter):
    refnucenergy = 59.2532646680161  # TEST
    refenergy = -115.8302823663  # TEST

    # starting point is D2d/c2v
    allene = psi4.geometry(
        """
     H  0.0  -0.92   -1.8
     H  0.0   0.92   -1.8
     C  0.0   0.00   -1.3
     C  0.0   0.00    0.0
     C  0.0   0.00    1.3
     H  0.92  0.00    1.8
     H -0.92  0.00    1.8
    """
    )

    psi4.core.clean_options()

    psi4_options = {
        "basis": "DZ",
        "e_convergence": 10,
        "d_convergence": 10,
        "scf_type": "pk",
    }
    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4("hf")
    E = json_output["energies"][-1]
    nucenergy = json_output["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]
    assert psi4.compare_values(refnucenergy, nucenergy, 2, "Nuclear repulsion energy")  # TEST
    assert psi4.compare_values(refenergy, E, 6, "Reference energy")  # TEST

    # central C-C-C bond angle starts around 170 degrees to test the dynamic addition
    # of new linear bending coordinates, and the redefinition of dihedrals.
    allene = psi4.geometry(
        """
     H  0.0  -0.92   -1.8
     H  0.0   0.92   -1.8
     C  0.0   0.00   -1.3
     C  0.0   0.10    0.0
     C  0.0   0.00    1.3
     H  0.92  0.00    1.8
     H -0.92  0.00    1.8
    """
    )

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("hf")
    E = json_output["energies"][-1]
    nucenergy = json_output["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]
    assert psi4.compare_values(refnucenergy, nucenergy, 2, "Nuclear repulsion energy")  # TEST
    assert psi4.compare_values(refenergy, E, 6, "Reference energy")  # TEST
    utils.compare_iterations(json_output, 7, check_iter)
