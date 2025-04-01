import psi4
import os
import pytest

@pytest.mark.parametrize(
    "direction, points, energies",
    (
        [
            "forward",
            [1, 5, 10, -1] ,
            [
                -189.211724350623,
                -189.212236241247,
                -189.213440976888,
                -189.215036034028,
            ]
        ],
        [
            "backward",
            [1, 5, 10, -1],
            [
                -189.211724350623,
                -189.212236241247,
                -189.213440976888,
                -189.215036034028,

            ]
        ]
    )
)
def test_irc(direction, points, energies):
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
        "opt_type": "IRC",
        "irc_direction": direction,
        "cart_hess_read": True,
        "irc_points": 30,
        "geom_maxiter": 200,
        "write_opt_history": True,
    }

    psi4.set_options(psi4_options)

    # IRC
    E, history = psi4.optimize("HF", return_history=True)
    os.system(f"rm stdout.default.{os.getpid()}.hess")
    assert False

# Reference values for test are from optking itself to check that IRC remains consistent
# Forward
# |  IRC   | New optking | Old optking | GAMESS | ORCA |
# | -------|-------------|-------------|--------|------|
# | Pt. 1. |  |
# | Pt. 5. |  |
# | Pt. 10.|  |
# | Pt. -1.|  |
# Backward
# |  IRC   | New optking | Old optking | GAMESS | ORCA |
# | -------|-------------|-------------|--------|------|
# | Pt. 1. |  |
# | Pt. 5. |  |
# | Pt. 10.|  |
# | Pt. -1.|  |

@pytest.mark.parametrize(
    "direction, point, energy",
    (
        [
            "forward",
            [1, 5, 10, -1],
            [
                -91.565770840626,
                -91.583446688556,
                -91.609217364630,
                -91.643652784351
            ],
        ],
        [
            "backward",
            [1, 5, 10, -1],
            [
                -91.565801178404
                -91.591173512151
                -91.640628526650
                -91.674194750859
            ],
        ]
    )
)
def test_irc_HCN(direction, point, energy):
    psi4.geometry("""
        N  -0.0428368408   0.5748930708   0.0000000000
        C  -0.0428349579  -0.6464622551   0.0000000000
        H   1.1052104074  -0.2904356612   0.0000000000
    """)

    # find Hessian
    psi4.set_options({
        "basis": "STO-3G",
        "maxiter": 200,
        "opt_type": "IRC",
        "irc_direction": direction,
        "cart_hess_read": True,
        "irc_points": 40,
        "geom_maxiter": 300,
    })

    E, history = psi4.optimize("HF", return_history=True)
    # Check final. Should procure a reference history at some point to check against
    assert False

@pytest.mark.parametrize(
    "direction, point, energy",
    (
        [
            "forward",
            [1, 10, 20, -1],
            [
                -40.160833152926
                -40.192017969747
                -40.193389299887
                -40.193425771411
            ],
        ],
        [
            "backward",
            [1, 10, 20, -1],
            [
                -40.159536620299,
                -40.191073983759,
                -40.194493549512,
                -40.194556669918,
            ]
        ]
    )
)
def test_irc_CH5(direction, point, energy):
    psi4.geometry("""
        C   0.1513220558  -0.0642324356  -0.1019101693
        H   0.8996390163  -0.3080157036   0.6421558506
        H   0.4938705392   0.6945107521  -0.7947392057
        H  -0.2237254315  -0.9441710656  -0.6099357324
        H  -0.8573989572   0.4500919469   0.5417903431
        H  -1.5661738247   0.8114829756   0.9940823997

    """)

    # find Hessian
    psi4.set_options({
        "basis": "STO-3G",
        "reference": "uhf",
        "maxiter": 200,
        "opt_type": "IRC",
        "irc_direction": direction,
        "cart_hess_read": True,
        "irc_points": 40,
        "geom_maxiter": 300,
    })

    E, history = psi4.optimize("HF", return_history=True)
    # Check final. Should procure a reference history at some point to check against
    assert False
