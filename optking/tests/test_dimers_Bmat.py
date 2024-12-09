import psi4
import optking
import numpy as np
import pytest

# Test interfragment coordinate B matrix numerically.
@pytest.mark.dimers
def test_dimers_bmat():
    h2oA = psi4.geometry(
        """
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """
    )
    Axyz = h2oA.geometry().np
    h2oB = psi4.geometry(
        """
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """
    )
    Bxyz = Axyz.copy() + 5.0  # Move B not right on top of A.

    dimer = {
        "Natoms per frag": [3, 3],
        "A Frag": 1,
        "A Ref Atoms": [[1], [2, 3], [3]],
        "B Frag": 2,
        "B Ref Atoms": [[4], [5, 6], [6]],
    }

    # validate and standardize
    dimer_model = optking.optparams.InterfragCoords(**dimer)
    Itest = optking.dimerfrag.DimerFrag.from_user_dict(dimer_model.model_dump(by_alias=True))

    # Here is lower level method
    # Aref = [[0],[1,2],[2]]
    # Bref = [[0],[1,2],[2]]
    # Itest = optking.dimerfrag.DimerFrag(0,Aref,1,Bref)

    Itest.update_reference_geometry(Axyz, Bxyz)
    print(Itest)

    max_error = Itest.test_B(Axyz, Bxyz)

    print("Max. difference between analytic and numerical B matrix: {:8.3e}".format(max_error))
    assert max_error < 1.0e-9
