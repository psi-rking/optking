#! Various constrained energy minimizations of HOOH with cc-pvdz RHF
#! Internal-coordinate constraints in internal-coordinate optimizations.
import pytest

import psi4
import optking

OH_frozen_stre_rhf = -150.781130356  # TEST
OOH_frozen_bend_rhf = -150.786372411  # TEST
HOOH_frozen_dihedral_rhf = -150.786766848  # TEST

f1 = {"frozen_distance": "1 2 3 4"}
f2 = {"frozen_bend": "1 2 3 2 3 4"}
f3 = {"frozen_dihedral": "1 2 3 4"}

frozen_coords = [(f1, OH_frozen_stre_rhf), (f2, OOH_frozen_bend_rhf), (f3, HOOH_frozen_dihedral_rhf)]


@pytest.mark.parametrize("option, expected", frozen_coords, ids=["frozen stretch", "frozen bend", "frozen dihedral"])
def test_frozen_coords(option, expected):
    # Constrained minimization with frozen bond, bend, and torsion
    hooh = psi4.geometry(
        """
      H
      O 1 0.90
      O 2 1.40 1 100.0 
      H 3 0.90 2 100.0 1 115.0
    """
    )

    psi4.core.clean_options()

    psi4_options = {"diis": "false", "basis": "cc-PVDZ", "scf_type": "pk", "print": 4, "g_convergence": "gau_tight"}
    psi4.set_options(psi4_options)
    psi4.set_module_options("OPTKING", option)

    json_output = optking.optimize_psi4("hf")
    thisenergy = json_output["energies"][-1]

    assert psi4.compare_values(expected, thisenergy, 6)  # TEST
