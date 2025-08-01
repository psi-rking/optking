import pytest
import psi4
import optking
from .utils import utils

#! B3LYP cc-pVDZ geometry optimzation of phenylacetylene, starting from
#! not quite linear structure
@pytest.mark.long
def test_b3lyp_phenylacetylene(check_iter):

    _ = psi4.geometry(
        """
      0 1
      C          0.50424        2.62143       -1.86897
      C         -0.79405        2.10443       -1.80601
      C         -1.78491        2.59819       -2.66154
      C         -1.47738        3.60745       -3.57782
      C         -0.17996        4.12418       -3.64086
      C          0.81143        3.63137       -2.78697
      H          1.27012        2.23852       -1.20693
      H         -2.24344        3.98980       -4.23978
      H          0.05756        4.90505       -4.35040
      H         -1.03189        1.32372       -1.09717
      H         -2.78881        2.19838       -2.61341
      C          2.14399        4.16411       -2.85667
      C          3.26501        4.60083       -2.90366
      H          4.24594        4.99166       -2.95361
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "guess": "sad",
        "basis": "cc-pVDZ",
    }
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("B3LYP")

    REF_b3lyp_E = -308.413691796  # TEST
    E = result["energies"][-1]  # TEST
    assert psi4.compare_values(REF_b3lyp_E, E, 5, "B3LYP energy")  # TEST
    utils.compare_iterations(result, 5, check_iter)
