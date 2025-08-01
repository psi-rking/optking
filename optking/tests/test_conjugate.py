#! Test conjugate gradient algorithms by optimizing propylamine
#! For comparison steepest descent converges in 16 iterations.
import pytest

import psi4
import optking
from .utils import utils

check_iter = True

REF_PROPYLAMINE = -173.3000252
cg_step_types = [
    ("FLETCHER", -150.786741206765),
    ("DESCENT", -150.786741176406),
    ("POLAK", -150.786740344611)
]

def test_conjugate_gradient_default(check_iter):
    _ = psi4.geometry(
     """
     N    1.8767   -0.1522   -0.0054 
     C   -0.5459   -0.5165    0.0053 
     C    0.5800    0.5145    0.0053 
     C   -1.9108    0.1542   -0.0052 
     H  -0.4629   -1.1669   -0.8740 
     H  -0.4700   -1.1586    0.8912 
     H   0.5127    1.1539    0.8920 
     H   0.5047    1.1629   -0.8741 
     H  -2.0434    0.7874    0.8779 
     H  -2.0351    0.7769   -0.8969 
     H  -2.7039   -0.6001   -0.0045 
     H   1.9441   -0.7570   -0.8233 
     H   1.9519   -0.7676    0.8039 
     """
    )

    psi4.core.clean_options()
    psi4_options = {"basis": "cc-pvdz", "d_convergence": 10, "geom_maxiter": 70}
    psi4.set_options(psi4_options)

    optking_options = {"step_type": "CONJUGATE"}

    json_output = optking.optimize_psi4("hf", **optking_options)
    thisenergy = json_output["energies"][-1]

    assert psi4.compare_values(REF_PROPYLAMINE, thisenergy, 5)
    utils.compare_iterations(json_output, 14, check_iter)

@pytest.mark.parametrize("option, energy", cg_step_types, ids=["FLETCHER", "DESCENT", "POLAK"])
def test_conjugate_gradient_type(option, energy, check_iter):

    _ = psi4.geometry(
        """
        H
        O 1 0.95
        O 2 1.39 1 102.0
        H 3 0.95 2 102.0 1 120.0
        """
    )

    psi4_options = {
        "basis": "cc-pvdz",
        "scf_type": "pk",
        "step_type": "CONJUGATE",
        "conjugate_gradient_type": option,
        "intrafrag_step_limit": 0.3,
        "intrafrag_step_limit_max": 0.4,
        "geom_maxiter": 5
    }

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("hf")  # Uses default program (psi4)
    thisenergy = json_output["energies"][-1]

    assert psi4.compare_values(energy, thisenergy, 5, "Final energy, every step Hessian")  # TEST
    assert len(json_output["energies"]) == 5

