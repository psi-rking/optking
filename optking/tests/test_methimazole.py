import psi4
import optking
import pytest
from .utils import utils

REF_E = -662.48908779


@pytest.mark.long
def test_methimazole(check_iter):
    psi4.geometry(
        """
      0 1
      S           -1.887836596846    -0.729301509543     0.000151677997 
      N            0.379763403154     0.904898490457    -0.000348322003 
      N            0.782863403154    -1.302501509543    -0.000048322003 
      C           -0.271536596846    -0.357101509543    -0.000448322003 
      C            1.743863403154     0.684098490457     0.000051677997 
      C           -0.201136596846     2.231498490457     0.000151677997 
      C            1.979363403154    -0.633601509543     0.000151677997 
      H            2.450363403154     1.498298490457     0.000251677997 
      H            0.132963403154     2.764798490457     0.894951677997 
      H           -1.292036596846     2.183998490457    -0.001348322003 
      H            0.135363403154     2.766598490457    -0.892748322003 
      H            0.687263403154    -2.310401509543     0.000151677997 
      H            2.917763403154    -1.162001509543     0.000551677997 
    """
    )
    psi4_options = {"basis": "pcseg-0"}
    psi4.set_options(psi4_options)
    result = optking.optimize_psi4("wb97x-d")
    E = result["energies"][-1]
    assert psi4.compare_values(E, REF_E, 5, "WB97X-D Min Energy")
    utils.compare_iterations(result, 12, check_iter)
