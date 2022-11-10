#! Test out-of-plane angles.  Added when only one central atom.
# Usually not needed, but problem in a situation like this, very nearly
# planar formaldehyde. Three bends become redundant as one approaches planarity.
# So eigenvalues of (B*B^t) contain a very small eval.  If you include this in
# your matrix inversion it blows up, and if you don't, then you can't tightly
# converge.  Adding out-of-planes solves problem.

b3lyp_opt_energy = -114.41550257
# Range limit both H-C-O(-H) out-of-plane angles to >30 degrees
b3lyp_ranged_oop_30_energy = -114.4032481
# Add harmonic force to push out-of-plane angles to +30 and -30 degrees,
# because extra force is additional, final values are not exactly 30/-30
# and final is a bit lower.
b3lyp_ext_force_oop_30_energy = -114.4032891

import psi4
import optking
from .utils import utils


def test_oofp_formaldehyde(check_iter):
    form = psi4.geometry(
        """
       O      0.6   -0.00007   0.0
       C     -0.6   -0.00007   0.0
       H     -1.2    0.24    -0.9
       H     -1.2   -0.24     0.9
       symmetry c1
    """
    )
    psi4.core.clean_options()
    psi4_options = {
        "basis": "def2-SVP",
        "g_convergence": "gau_tight",
        "test_B": True,
    }
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("b3lyp")
    E = result["energies"][-1]  # TEST

    assert psi4.compare_values(b3lyp_opt_energy, E, 8, "B3LYP energy")
    utils.compare_iterations(result, 8, check_iter)


def test_ranged_oofp(check_iter):
    form = psi4.geometry(
        """
      O   0.08  0.60  -0.0
      C  -0.04 -0.60  -0.0
      H  -0.4  -1.14  -0.92
      H  -0.4  -1.14   0.92
    """
    )
    psi4.core.clean_options()
    psi4_options = {"basis": "def2-SVP", "ensure_bt_convergence": True}
    psi4.set_options(psi4_options)

    xtra = {"ranged_oofp": "(3 2 1 4 -40.0 -30.0) (4 2 1 3 30.0 40.0)"}
    result = optking.optimize_psi4("b3lyp", **xtra)
    E = result["energies"][-1]

    assert psi4.compare_values(b3lyp_ranged_oop_30_energy, E, 5, "B3LYP energy")
    utils.compare_iterations(result, 15, check_iter)


def test_ext_force_oofp(check_iter):
    form = psi4.geometry(
        """
      O   0.08  0.60  -0.0
      C  -0.04 -0.60  -0.0
      H  -0.4  -1.14  -0.92
      H  -0.4  -1.14   0.92
    """
    )
    psi4.core.clean_options()
    psi4_options = {"basis": "def2-SVP", "ensure_bt_convergence": True}
    psi4.set_options(psi4_options)

    xtra = {"ext_force_oofp": "3 2 1 4 '-0.5*(x+30)' 4 2 1 3 '-0.5*(x-30.0)'"}
    result = optking.optimize_psi4("b3lyp", **xtra)
    E = result["energies"][-1]

    assert psi4.compare_values(b3lyp_ext_force_oop_30_energy, E, 5, "B3LYP energy")
    utils.compare_iterations(result, 12, check_iter)
