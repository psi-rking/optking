import psi4
import os

def test_irc():
    psi4.geometry("""
	0 2
	C           -1.189911690573     0.201247528125     0.167673149223
	O           -0.091451900076    -0.485249388360    -0.243225315637
	H           -1.021145249899     1.181999865559     0.590345104870
	H           -2.058253276210    -0.023216256085    -0.441170124838
	H            1.572684844453    -0.372798307107     0.586272547903
	O            1.079106065627     0.284741534678     0.071090591276
	""")

    os.system(f"cp CH3O2_irc.hess stdout.default.{os.getpid()}.hess")

    psi4_options = {
        "basis": "6-31G(D)",
        "reference": "uhf",
        "maxiter": 200,
        "dft_radial_points": 99,
        "dft_spherical_points": 590,
        "dft_pruning_scheme": "robust",
        "opt_type": "IRC",
        "irc_direction": "backward",
        "irc_step_size": 0.1,
        "cart_hess_read": True,
        "irc_points": 60,
        "geom_maxiter": 300,
    }

    psi4.set_options(psi4_options)
    psi4.set_module_options(
        "Optking", {"ensure_bt_convergence": "True", "g_convergence": "gau_tight"}
    )
    # IRC
    E, history = psi4.optimize("MPW1PW", return_history=True)
    os.system(f"rm stdout.default.{os.getpid()}.hess")

    psi4.core.clean_options()
    psi4_options = {
        "basis": "6-31G(D)",
        "reference": "uhf",
        "maxiter": 200,
        "dft_radial_points": 99,
        "dft_spherical_points": 590,
        "dft_pruning_scheme": "robust",
        "geom_maxiter": 5,
    }
    psi4.set_options(psi4_options)
    psi4.optimize("MPW1PW")
