#! Various constrained energy minimizations of HOOH with cc-pvdz RHF
#! Internal-coordinate constraints in internal-coordinate optimizations.
import pytest

import psi4
import optking
from .utils import utils


OH_frozen_stre_rhf = -150.781130356  # TEST
OOH_frozen_bend_rhf = -150.786372411  # TEST
HOOH_frozen_dihedral_rhf = -150.786766848  # TEST

f1 = {"frozen_distance": "1 2 3 4"}
f2 = {"frozen_bend": "1 2 3 2 3 4"}
f3 = {"frozen_dihedral": "1 2 3 4"}

optking__frozen_coords = [
    (f1, OH_frozen_stre_rhf, 9),
    (f2, OOH_frozen_bend_rhf, 6),
    (f3, HOOH_frozen_dihedral_rhf, 6)
]

@pytest.mark.parametrize(
    "option, expected, num_steps",
    optking__frozen_coords,
    ids=["frozen stretch", "frozen bend", "frozen dihedral"]
)
def test_frozen_coords(option, expected, num_steps, check_iter):
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

    psi4_options = {
        "diis": "false",
        "basis": "cc-PVDZ",
        "scf_type": "pk",
        "print": 4,
        "g_convergence": "gau_tight"
    }
    psi4.set_options(psi4_options)
    psi4.set_options(option)

    json_output = optking.optimize_psi4("hf")
    thisenergy = json_output["energies"][-1]

    assert psi4.compare_values(expected, thisenergy, 6)  # TEST
    utils.compare_iterations(json_output, num_steps, check_iter)


def test_butane_frozen(check_iter):
    _ = psi4.geometry("pubchem:butane")

    psi4.core.clean_options()
    psi4_options = {
        "basis": "6-31G",
        "g_convergence": "gau_tight",
    }
    psi4.set_options(psi4_options)

    tmp = {"freeze_all_dihedrals": True,}
    result = optking.optimize_psi4("scf", **tmp)
    E1 = result["energies"][-1]  # TEST

    psi4.core.clean_options()
    psi4_options = {
        "basis": "6-31G",
        "g_convergence": "gau_tight",
        "frozen_dihedral": """
            1 2 4 12
            1 2 4 13
            1 2 4 14
            2 1 3 9
            2 1 3 10
            2 1 3 11
            3 1 2 4
            3 1 2 7
            3 1 2 8
            4 2 1 5
            4 2 1 6
            5 1 2 7
            5 1 2 8
            5 1 3 9
            5 1 3 10
            5 1 3 11
            6 1 2 7
            6 1 2 8
            6 1 3 9
            6 1 3 10
            6 1 3 11
            7 2 4 12
            7 2 4 13
            7 2 4 14
            8 2 4 12
            8 2 4 13
            8 2 4 14
        """
    }
    psi4.set_options(psi4_options)
    result = optking.optimize_psi4("scf")
    E2 = result["energies"][-1]

    assert psi4.compare_values(E1, E2, 8, "RHF energy")  # TEST
    utils.compare_iterations(result, 5, check_iter)

def test_butane_skip_frozen(check_iter):
    _ = psi4.geometry("pubchem:butane")

    psi4.core.clean_options()
    psi4_options = {
        "basis": "6-31G",
        "g_convergence": "gau_tight",
    }
    tmp = {
        "freeze_all_dihedrals": True,
        "unfreeze_dihedrals": """
            8 2 4 12
            8 2 4 13
            8 2 4 14
            3 1 2 8
            5 1 2 8
            6 1 2 8"""}

    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("scf", **tmp)
    E1 = result["energies"][-1]  # TEST

    psi4.core.clean_options()
    psi4_options = {
        "basis": "6-31G",
        "g_convergence": "gau_tight",
        "frozen_dihedral": """
            1 2 4 12
            1 2 4 13
            1 2 4 14
            2 1 3 9
            2 1 3 10
            2 1 3 11
            3 1 2 4
            3 1 2 7
            4 2 1 5
            4 2 1 6
            5 1 2 7
            5 1 3 9
            5 1 3 10
            5 1 3 11
            6 1 2 7
            6 1 3 9
            6 1 3 10
            6 1 3 11
            7 2 4 12
            7 2 4 13
            7 2 4 14
        """
    }
    psi4.set_options(psi4_options)
    result = optking.optimize_psi4("scf")
    E2 = result["energies"][-1]

    assert psi4.compare_values(E1, E2, 8, "RHF energy")  # TEST
    utils.compare_iterations(result, 5, check_iter)
