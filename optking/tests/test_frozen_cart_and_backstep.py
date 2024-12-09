import psi4
import optking
import pytest
from .utils import utils
import numpy as np
from qcelemental import constants
bohr2angstroms = constants.bohr2angstroms

#! Various constrained energy minimizations of HOOH with cc-pvdz RHF.
#! Cartesian-coordinate constrained optimizations of HOOH in Cartesians.
#! 1. Cartesian optimization.
#! 2. Cartesian optimization with frozen H's.
#! 3. Cartesian optimization with frozen O's.
HOOH_E = -150.7866742  # TEST
HOOH_E_frozen_H_xyz = -150.7866491  # TEST
HOOH_E_frozen_O_xyz = -150.7866390  # TEST

f0 = """"""
# Freeze H xyz in HOOH.
f1 = """ 1 Xyz 4 xYz """
# Freeze O xyz in HOOH.
f2 = """ 2 xyz 3 xyz """
# Freeze H xyz with individual input entries in HOOH.
f3 = """
     1 x
     1 y
     1 Z
     4 x
     4 Y
     4 z """

opt0 = {"frozen_cartesian": f0}
opt1 = {"frozen_cartesian": f1}
opt2 = {"frozen_cartesian": f2}
opt3 = {"frozen_cartesian": f3}
opt4 = {"frozen_cartesian": f1, "opt_coordinates": "redundant"}

optking__freeze_params = [
    (opt0, HOOH_E, 15),
    (opt1, HOOH_E_frozen_H_xyz, 13),
    (opt2, HOOH_E_frozen_O_xyz, 13),
    (opt3, HOOH_E_frozen_H_xyz, 13),
    (opt4, HOOH_E_frozen_H_xyz, 13),
]


@pytest.mark.parametrize(
    "options, expected, num_steps",
    # freeze_params,
    optking__freeze_params,
    ids=["Only backstep", "freeze H", "freeze O", "freeze individual x,y,z", "freeze then change coord"],
)
def test_hooh_freeze_xyz_Hs(check_iter, options, expected, num_steps):

    hooh = psi4.geometry(
        """
      H  0.90  0.80  0.5
      O  0.00  0.70  0.0
      O  0.00 -0.70  0.0
      H -0.90 -0.80  0.5
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "cc-pvdz",
        "opt_coordinates": "cartesian",
        "g_convergence": "gau_tight",
        "geom_maxiter": 20,
        "consecutive_backsteps": 1,
    }
    psi4.set_options(psi4_options)
    psi4.set_options(options)

    json_output = optking.optimize_psi4("hf")

    thisenergy = json_output["energies"][-1]  # TEST
    assert psi4.compare_values(expected, thisenergy, 6)  # TEST

    utils.compare_iterations(json_output, num_steps, check_iter)


#! test if we can keep oxygen atom from moving off of the point (1,1,1)
def test_frozen_cart_h2o(check_iter):

    h2o = psi4.geometry(
        """
        O   1.000000   1.000000   1.000000
        H   2.000000   1.000000   1.000000
        H   1.000000   2.000000   1.000000
        units angstrom
        no_com
        no_reorient
    """
    )

    psi4.core.clean_options()
    psi4_options = {"basis": "cc-pvdz", "reference": "rhf", "scf_type": "df", "max_energy_g_convergence": 7}
    psi4.set_options(psi4_options)
    psi4.set_options({"frozen_cartesian": """1 xyz"""})

    json_output = optking.optimize_psi4("hf")

    optGeom = bohr2angstroms*np.asarray(json_output['final_molecule']['geometry']).reshape(-1,3)

    thisenergy = json_output["energies"][-1]
    assert psi4.compare_values(-76.0270327834836, thisenergy, 6, "RHF Energy")
    assert psi4.compare_values(optGeom[0,0], 1.0, 6, "X Frozen coordinate")
    assert psi4.compare_values(optGeom[0,1], 1.0, 6, "Y Frozen coordinate")
    assert psi4.compare_values(optGeom[0,2], 1.0, 6, "Z Frozen coordinate")

    utils.compare_iterations(json_output, 6, check_iter)


#! test h2o dimer with frozen oxygen atoms
def test_frozen_cart_h2o_dimer(check_iter):
    h2oDimer = psi4.geometry(
        """
        O   -0.3289725   -1.4662712    0.0000000
        H   -1.3007725   -1.5158712    0.0000000
        H    0.0634274    0.4333287    0.0000000
        O    0.3010274    1.3644287    0.0000000
        H    0.8404274    1.3494287    0.7893000
        H    0.8404274    1.3494287   -0.7893000
        units = angstrom
        no_com
        no_reorient
        """
    )
    inputGeom= np.asarray(h2oDimer.geometry())

    psi4.core.clean_options()
    psi4_options = {
        "basis": "cc-pVDZ",
        "g_convergence": "gau_tight",
        "frozen_cartesian": """ 1 xyz 4 xyz """,
        "geom_maxiter": 40,
    }
    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4("mp2")

    optGeom = np.asarray(json_output['final_molecule']['geometry']).reshape(-1,3)

    thisenergy = json_output["energies"][-1]  # TEST
    assert psi4.compare_values(-152.47381494, thisenergy, 8, "MP2 Energy")

    assert psi4.compare_values(inputGeom[0,0], optGeom[0,0], 6, "O1 X Frozen coordinate")
    assert psi4.compare_values(inputGeom[0,1], optGeom[0,1], 6, "O1 Y Frozen coordinate")
    assert psi4.compare_values(inputGeom[0,2], optGeom[0,2], 6, "O1 Z Frozen coordinate")
    assert psi4.compare_values(inputGeom[3,0], optGeom[3,0], 6, "O2 X Frozen coordinate")
    assert psi4.compare_values(inputGeom[3,1], optGeom[3,1], 6, "O2 Y Frozen coordinate")
    assert psi4.compare_values(inputGeom[3,2], optGeom[3,2], 6, "O2 Z Frozen coordinate")
    assert np.abs(inputGeom[1,0] - optGeom[1,0]) > 0.01 # Check a not-frozen coordinate.

    utils.compare_iterations(json_output, 25, True)

