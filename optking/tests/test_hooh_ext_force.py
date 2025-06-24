#! Various constrained energy minimizations of HOOH with cc-pvdz RHF.
#! For "fixed" coordinates, the final value is provided by the user.
import pytest
import psi4
import optking
from .utils import utils

# Minimized energy with OH bonds pushed toward 0.950 Angstroms.
OH_950_stre = -150.786669
# Minimized energy with OOH angles pushed toward 105.0 degrees.
OOH_105_bend = -150.786177
# Minimized energy with HOOH torsion pushed toward 120.0 degrees.
HOOH_120_dihedral = -150.786647
# Minimize energy with the x and y coordinates of atom 1 pushed
# to 1.0 and 1.0.  Just for fun.
HOOH_minimum = -150.7866742

f1 = ({"ext_force_distance": "1 2 '-8.0*(x-0.950)' 3 4 '-8.0*(x-0.950)'"}, OH_950_stre, 9)
f2 = ({"ext_force_bend": "1 2 3 '-8.0*(x-105.0)' 2 3 4 '-8.0*(x-105.0)'"}, OOH_105_bend, 17)
f3 = ({"ext_force_dihedral": "1 2 3 4 '-8.0*(x-120.0)'"}, HOOH_120_dihedral, 15)
f4 = ({"ext_force_cartesian": "1 x '-2.0*(x-1.0)' 1 y '-2.0*(x-1.0)'"}, HOOH_minimum, 10)
# Same as f1, but 'soften'/dampen force at long range.
f5 = (
    {
        "ext_force_distance": "1 2 '-8.0 * (x-0.950) * exp(-20*abs(x-0.950) )' 3 4 '-8.0*(x-0.950) * exp(-20*abs(x-0.950))'"
    },
    OH_950_stre,
    13
)


@pytest.mark.parametrize("option, expected, num_steps", [f1, f2, f3, f4, f5])
def test_hooh_fixed_OH_stre(option, expected, num_steps, check_iter):
    hooh = psi4.geometry(
        """
      H
      O 1 0.90
      O 2 1.40 1 100.0
      H 3 0.90 2 100.0 1 115.0
    """
    )

    psi4.core.clean_options()
    psi4_options = {"diis": "false", "basis": "cc-pvdz", "g_convergence": "gau_tight"}
    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4("hf", **option)

    E = json_output["energies"][-1]
    assert psi4.compare_values(expected, E, 6, list(option.keys())[0])
    # utils.compare_iterations(json_output, num_steps, check_iter)
