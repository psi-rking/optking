# SF4 problem case
# Has linear bend, breaks symmetry
# Converges fine if Cartesian
# needs debugging

import psi4
import optking
import pytest

from .utils import utils


def test_sf4_quasilinear_cart(check_iter):
    sf4 = psi4.geometry(
        """
      S  0.00000000  -0.00000000  -0.30618267
      F -1.50688420  -0.00000000   0.56381732
      F  0.00000000  -1.74000000  -0.30618267
      F -0.00000000   1.74000000  -0.30618267
      F  1.50688420   0.00000000   0.56381732
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "6-31G(d)",
        "scf_type": "pk",
        "opt_coordinates": "cartesian",
    }
    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4("hf")

    E = json_output["energies"][-1]
    REF_energy = -795.1433965
    assert psi4.compare_values(REF_energy, E, 6, "Reference energy")

    utils.compare_iterations(json_output, 9, check_iter)


# This needs debugged. RAK (Jan. 2020) thinks that when/if we get
# full symmetrization of our step (C2v) point group here, then the
# problem that appears in step 7-8 will go away.
# A secondary problem is that it is possible that our definition of
# linear bends forces symmetry breaking, but for now this has been
# tweaked for at least this case by choosing different bend axes
# inside optking. (see the arbitrary ref. axes in Helgaker/Bakken).
@pytest.mark.skip
def test_sf4_quasilinear():
    sf4 = psi4.geometry(
        """
      S  0.00000000  -0.00000000  -0.30618267
      F -1.50688420  -0.00000000   0.56381732
      F  0.00000000  -1.74000000  -0.30618267
      F -0.00000000   1.74000000  -0.30618267
      F  1.50688420   0.00000000   0.56381732
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "6-31G(d)",
        "scf_type": "pk",
        "g_convergence": "gau_tight",
    }
    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4("hf")

    E = json_output["energies"][-1]
    REF_energy = -795.1433965
    assert psi4.compare_values(REF_energy, E, 6, "Reference energy")
