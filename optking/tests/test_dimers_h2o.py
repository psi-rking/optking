import psi4
import optking
import numpy as np
import pytest
from .utils import utils

# Demonstrate and test positioning two water molecules by specifying
# their interfragment reference points and coordinates.
@pytest.mark.dimers
def test_dimerfrag_orient_h2o_dimers():
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

    # Choose some reference atoms for each fragment.
    # These would be # the O atom, the point between the two H atoms,
    # and one of the hydrogen atoms for each fragment.

    dimer = {
        "NATOMS PER FRAG": [3, 3],
        "A FRAG": 1,
        "A REF ATOMS": [[1], [2, 3], [3]],
        "A LABEL": "Water-A",  # optional
        "B FRAG": 2,
        "B REF ATOMS": [[4], [5, 6], [6]],
        "B LABEL": "Water-B",  # optional
    }

    # The default weights are equal between involved atoms but
    # may be specified.  Fragment labels are optional.
    Itest = optking.dimerfrag.DimerFrag.from_user_dict(dimer)
    Itest.update_reference_geometry(Axyz, Bxyz)
    # print(Itest)

    # Create arbitrary target for displacement with illustrative names.
    R_A1B1 = 3.4
    theta_A2A1B1 = 2.5
    theta_A1B1B2 = 2.7
    tau_A2A1B1B2 = -1.5
    phi_A3A2A1B1 = 0.3
    phi_A1B1B2B3 = 0.6

    q_target = np.array([R_A1B1, theta_A2A1B1, theta_A1B1B2, tau_A2A1B1B2, phi_A3A2A1B1, phi_A1B1B2B3])
    Bxyz_new = Itest.orient_fragment(Axyz, Bxyz, q_target)

    # Test how we did
    Itest.update_reference_geometry(Axyz, Bxyz_new)
    rms_error = np.sqrt(np.mean((q_target - Itest.q_array()) ** 2))
    print("RMS deviation from target interfragment coordinates: {:8.3e}".format(rms_error))
    assert rms_error < 1.0e-10


MP2minEnergy = -152.5352095


@pytest.mark.dimers
@pytest.mark.parametrize("option, iter", [("gau_tight", 13), ("interfrag_tight", 11)])
def test_dimers_h2o_auto(check_iter, option, iter):  # auto reference pt. creation
    h2oD = psi4.geometry(
        """
      0 1
      H   0.280638    -1.591779    -0.021801
      O   0.351675    -1.701049     0.952490
      H  -0.464013    -1.272980     1.251761
      --
      0 1
      H  -0.397819    -1.918411    -2.373012
      O  -0.105182    -1.256691    -1.722965
      H   0.334700    -0.589454    -2.277374
      nocom
      noreorient
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "aug-cc-pvdz",
        "geom_maxiter": 40,
        "frag_mode": "MULTI",
        "g_convergence": f"{option}",
    }
    psi4.set_options(psi4_options)

    newOptParams = {"interfrag_collinear_tol": 0.2}  # increase to prevent too colinear reference points
    json_output = optking.optimize_psi4("mp2", **newOptParams)

    E = json_output["energies"][-1]
    assert psi4.compare_values(MP2minEnergy, E, 6, "MP2 Energy opt from afar, auto")

    utils.compare_iterations(json_output, iter, check_iter)
