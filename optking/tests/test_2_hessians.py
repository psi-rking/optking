import logging
import pytest
import psi4
import optking
from .utils import utils

final_energy = -150.786766850
hess_every = [(-1, final_energy, 10), (0, final_energy, 10), (1, final_energy, 4), (3, final_energy, 5)]
hess_guess = [
    ("fischer", final_energy, 9),
    ("lindH_simple", final_energy, 11),
    ("simple", final_energy, 12),
    ("lindh", final_energy, 16),
]
hess_update = [("MS", final_energy, 10), ("powell", final_energy, 11), ("bofill", final_energy, 10)]

logger = optking.logger


@pytest.mark.parametrize("every, expected, num_steps", hess_every, ids=["None", "First Step", "Every", "Every 3"])
def test_hess_every(check_iter, every, expected, num_steps):

    hooh = psi4.geometry(
        """
        H
        O 1 0.95
        O 2 1.39 1 102.0
        H 3 0.95 2 102.0 1 130.0
        """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "cc-pvdz",
        "scf_type": "pk",
        "g_convergence": "gau_verytight",
        "full_hess_every": every,
        "print": 4,
    }

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("hf")  # Uses default program (psi4)
    E = json_output["energies"][-1]

    assert psi4.compare_values(expected, E, 8, "Final energy, every step Hessian")  # TEST

    utils.compare_iterations(json_output, num_steps, check_iter)


@pytest.mark.parametrize("guess, expected, num_steps", hess_guess)
def test_hess_guess(check_iter, guess, expected, num_steps):

    hooh = psi4.geometry(
        """
        H
        O 1 0.95
        O 2 1.39 1 102.0
        H 3 0.95 2 102.0 1 130.0
        """
    )

    psi4.core.clean_options()
    psi4_options = {"basis": "cc-pvdz", "scf_type": "pk", "g_convergence": "gau_verytight", "intrafrag_hess": guess}

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("hf")  # Uses default program (psi4)
    E = json_output["energies"][-1]
    print(f"Number of steps taken {len(json_output['trajectory'])}")
    assert psi4.compare_values(expected, E, 8, "Final energy, every step Hessian")  # TEST

    utils.compare_iterations(json_output, num_steps, check_iter)


@pytest.mark.parametrize("update, expected, num_steps", hess_update)
def test_hess_update(check_iter, update, expected, num_steps):

    hooh = psi4.geometry(
        """
        H
        O 1 0.95
        O 2 1.39 1 102.0
        H 3 0.95 2 102.0 1 130.0
        """
    )

    psi4.core.clean_options()
    psi4_options = {"basis": "cc-pvdz", "scf_type": "pk", "g_convergence": "gau_verytight", "hess_update": update}

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("hf")  # Uses default program (psi4)
    E = json_output["energies"][-1]

    assert psi4.compare_values(expected, E, 8, "Final energy, every step Hessian")  # TEST

    utils.compare_iterations(json_output, num_steps, check_iter)

def test_hess_read(check_iter):
    import os
    hooh = psi4.geometry(
        """
        H
        O 1 0.95
        O 2 1.39 1 102.0
        H 3 0.95 2 102.0 1 130.0
        """
    )

    psi4_options = {
        "basis": "cc-pvdz",
        "scf_type": "pk",
        "g_convergence": "gau_verytight",
        "hessian_write": True,
    }

    psi4.set_options(psi4_options)
    json_output = psi4.frequencies("hf")  # Uses default program (psi4)

    psi4.core.clean_options()
    psi4_options = {
        "basis": "cc-pvdz",
        "scf_type": "pk",
        "g_convergence": "gau_verytight",
        "cart_hess_read": True,
    }

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("hf", **{"cart_hess_read": True, "hessian_file": f"stdout.default.{os.getpid()}.hess"})
    E = json_output["energies"][-1]
    assert psi4.compare_values(final_energy, E, 8, "Final energy, every step Hessian")  # TEST
    utils.compare_iterations(json_output, 10, check_iter)

