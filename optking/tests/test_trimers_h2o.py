import psi4
import optking
import numpy as np
import pytest
from .utils import utils

MP2minEnergy = -228.669584375489

#! H2O trimer with auto-generated interfragment coordinates
@pytest.mark.multimers
def test_trimers_h2o_auto(check_iter):
    h2oT = psi4.geometry(
    """
      0 1
        H  -0.5296845029  -0.7745043757   1.6304058824
        O  -0.3979784755   0.1843932304   1.6549319122
        H  -0.9141487216   0.5272898351   2.4164161104
        H  -0.6187471122   3.1743020821  -0.1297043945
        O  -0.9756913099   2.7052566550   0.6370367142
        H  -0.6262324307   1.7890763023   0.6206096878
        H  -1.6570476579   2.6158975123   2.4264146351
        O  -1.8495640984   1.9929282874   3.1573108075
        H  -2.1329151957   2.5121708299   3.9231206251
    """
    )

    psi4_options = {
        "basis": "6-31+G*",
        "geom_maxiter": 30,
        "frag_mode": "MULTI",
        "intrafrag_step_limit": 0.2,
        "test_B": True
    }
    psi4.set_options(psi4_options)

    #json_output = optking.optimize_psi4("b3lyp-d3mbj")
    json_output = optking.optimize_psi4("mp2", **{"interfrag_trust": 0.2})

    E = json_output["energies"][-1]
    assert psi4.compare_values(MP2minEnergy, E, 6, "(H2O)_3 MP2 Energy opt, auto")
    utils.compare_iterations(json_output, 24, check_iter)

