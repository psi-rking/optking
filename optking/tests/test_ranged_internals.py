#! Test of 'ranged' coordinates.  Intcos that cannot move
#  out of a prescribed range.
import pytest
import psi4
import optking
from .utils import utils

conv_RHF_OO_at_135 = -150.7853238  # minimum is 1.39
init_OO_distance = [("1.28", 9), ("1.2999", 8), ("1.325", 8), ("1.35", 9), ("1.38", 8)]


@pytest.mark.parametrize("option, num_steps", init_OO_distance)
def test_ranged_stretch(option, num_steps, check_iter):
    geom_input_string = (
        """
      H
      O 1 0.90
      O 2 """
        + option
        + """ 1 100.0
      H 3 0.90 2 100.0 1 115.0 """
    )
    hooh = psi4.geometry(geom_input_string)

    psi4.core.clean_options()
    psi4options = {"basis": "cc-PVDZ", "print": 4, "g_convergence": "gau_tight", "geom_maxiter": 20}
    psi4.set_options(psi4options)

    xtra = {"ranged_distance": "2 3 1.30 1.35"}
    json_output = optking.optimize_psi4("hf", **xtra)

    thisenergy = json_output["energies"][-1]
    assert psi4.compare_values(conv_RHF_OO_at_135, thisenergy, 6)
    utils.compare_iterations(json_output, num_steps, check_iter)


conv_RHF_HOO_at_105 = -150.7861769  # minimum is 102 degrees
init_HOO_bend = [("100", 8), ("105", 6), ("108", 7), ("110", 8), ("115", 9)]


@pytest.mark.parametrize("option, num_steps", init_HOO_bend)
def test_ranged_bend(option, num_steps, check_iter):
    geom_input_string = (
        """
      H
      O 1 0.90
      O 2 1.35 1 """
        + option
        + """
      H 3 0.90 2 """
        + option
        + """ 1 115.0 """
    )
    hooh = psi4.geometry(geom_input_string)

    psi4.core.clean_options()
    psi4options = {"basis": "cc-PVDZ", "g_convergence": "gau_tight", "geom_maxiter": 20}
    psi4.set_options(psi4options)

    xtra = {"ranged_bend": "(1 2 3 105.0 110.0) (2 3 4 105.0 110.0)"}
    json_output = optking.optimize_psi4("hf", **xtra)

    thisenergy = json_output["energies"][-1]
    assert psi4.compare_values(conv_RHF_HOO_at_105, thisenergy, 6)
    utils.compare_iterations(json_output, num_steps, check_iter)


conv_RHF_HOOH_at_110 = -150.7866419  # minimum is 115 degrees
init_HOOH_tors = ["95", "100", "105", "110", "115"]


@pytest.mark.parametrize("option", init_HOOH_tors)
def test_ranged_tors(option):
    geom_input_string = (
        """
      H
      O 1 0.90
      O 2 1.35 1 100.0
      H 3 0.90 2 100.0 1 """
        + option
    )
    hooh = psi4.geometry(geom_input_string)

    psi4.core.clean_options()
    psi4options = {"basis": "cc-PVDZ", "g_convergence": "gau_tight", "geom_maxiter": 20}
    psi4.set_options(psi4options)

    xtra = {"ranged_dihedral": "(1 2 3 4 100.0 110.0)"}
    json_output = optking.optimize_psi4("hf", **xtra)

    thisenergy = json_output["energies"][-1]
    assert psi4.compare_values(conv_RHF_HOOH_at_110, thisenergy, 6)


conv_RHF_HOOH = -150.7866742  # this is the global minimum (GM)
cart_limits = [
    "1 x 0.77 0.80",  # Make H x be within 0.77 - 0.80; converge to GM
    "2 y 0.50 0.60",  # Make O y be within 0.50 - 0.60; converge to GM
    "(1 x 0.77 0.80) (2 y 0.50 0.60)",
]  # parentheses optional


@pytest.mark.parametrize("option", cart_limits)
def test_ranged_cart(option):
    hooh = psi4.geometry(
        """
        H   0.7551824472   0.7401035426   0.5633005896
        O   0.0870189589   0.6693673665  -0.0354930582
        O  -0.0870189589  -0.6693673665  -0.0354930582
        H  -0.7551824472  -0.7401035426   0.5633005896"""
    )

    psi4.core.clean_options()
    psi4options = {"basis": "cc-PVDZ", "g_convergence": "gau_tight", "geom_maxiter": 20}
    psi4.set_options(psi4options)

    xtra = {"ranged_cartesian": option}
    json_output = optking.optimize_psi4("hf", **xtra)

    thisenergy = json_output["energies"][-1]
    assert psi4.compare_values(conv_RHF_HOOH, thisenergy, 6)
