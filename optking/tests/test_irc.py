import psi4
import os
import pytest
import pathlib
import json
import numpy as np
import qcelemental as qcel
import optking

# Absolute path to tests
test_dir = pathlib.Path(__file__).parent

energies = [
    -189.211724350623,
    -189.212236241247,
    -189.213440976888,
    -189.215036034028,
]
points = [1, 5, 10, -1]


@pytest.mark.parametrize(
    "direction, points, energies",
    (
        ["forward", points, energies],
        ["backward", points, energies],
    ),
)
def test_irc_CH3O2(direction, points, energies):

    # Random IRC calculation that was ending before reaching a minimum

    psi4.geometry("""
        0 2
        C           -1.189911690573     0.201247528125     0.167673149223
        O           -0.091451900076    -0.485249388360    -0.243225315637
        H           -1.021145249899     1.181999865559     0.590345104870
        H           -2.058253276210    -0.023216256085    -0.441170124838
        H            1.572684844453    -0.372798307107     0.586272547903
        O            1.079106065627     0.284741534678     0.071090591276
	""")

    psi4_options = {
        "basis": "6-31G(D)",
        "reference": "uhf",
        "maxiter": 200,
        "opt_type": "IRC",
        "irc_direction": direction,
        "cart_hess_read": True,
        "irc_step_size": 0.2,
        "irc_points": 30,
        "geom_maxiter": 200,
        "hessian_file": f"{test_dir}/test_data/CH3O2_irc.hess"
    }

    psi4.set_options(psi4_options)

    # IRC
    optking.optimize_psi4("HF")


@pytest.mark.parametrize(
    "direction, point, energy",
    (
        [
            "forward",
            [
                1,
                5,
                10,
                -1,
            ],
            [
                -91.565770840626,
                -91.583446688556,
                -91.609217364630,
                -91.643652784351,
            ],
        ],
        [
            "backward",
            [
                1,
                5,
                10,
                -1,
            ],
            [
                -91.565801178404,
                -91.591173512151,
                -91.640628526650,
                -91.674194750859,
            ],
        ],
    ),
)
def test_irc_HCN(direction, point, energy):
    HCN = psi4.geometry("""
        N  -0.0428368408   0.5748930708   0.0000000000
        C  -0.0428349579  -0.6464622551   0.0000000000
        H   1.1052104074  -0.2904356612   0.0000000000
    """)

    # find Hessian
    psi4.set_options(
        {
            "basis": "STO-3G",
            "maxiter": 200,
            "opt_type": "IRC",
            "irc_direction": direction,
            "cart_hess_read": True,
            "irc_step_size": 0.2,
            "irc_points": 10,
            "geom_maxiter": 300,
            "hessian_file": f"{test_dir}/test_data/HCN_irc.hess",
        }
    )

    optking.optimize_psi4("HF")
    # Check final. Should procure a reference history at some point to check against
    # The forward direction visually matches SCHELGEL but due to how we add a linear bend to the
    # system vs their bend we do not match the backward direction. Their bend goes to zero ours
    # to 180.

@pytest.mark.parametrize(
    "direction, point, energy",
    (
        [
            "forward",
            [1, 5, 10, -1],
            [-40.160833152926, -40.192017969747, -40.193389299887, -40.193425771411],
        ],
        [
            "backward",
            [1, 5, 10, -1],
            [
                -40.159536620299,
                -40.191073983759,
                -40.194493549512,
                -40.194556669918,
            ],
        ],
    ),
)
def test_irc_CH5(direction, point, energy):
    ch5 = psi4.geometry("""
        C   0.1513220558  -0.0642324356  -0.1019101693
        H   0.8996390163  -0.3080157036   0.6421558506
        H   0.4938705392   0.6945107521  -0.7947392057
        H  -0.2237254315  -0.9441710656  -0.6099357324
        H  -0.8573989572   0.4500919469   0.5417903431
        H  -1.5661738247   0.8114829756   0.9940823997
    """)

    # TEST - leave in to check that OPTKING successfully utilizes Psi's standard hess file
    os.system(f"cp {test_dir}/test_data/CH5_irc.hess {psi4.core.get_writer_file_prefix(ch5.name())}.hess")

    # find Hessian
    psi4.core.clean_options()
    psi4.set_options(
        {
            "basis": "STO-3G",
            "reference": "uhf",
            "maxiter": 200,
            "opt_type": "IRC",
            "irc_direction": direction,
            "cart_hess_read": True,
            "irc_step_size": 0.2,
            "irc_points": 11,
            "write_opt_history": True,
        }
    )

    E, history = psi4.optimize("HF", return_history=True)
    # Check final. Should procure a reference history at some point to check against

    filename = pathlib.Path(f"{psi4.core.get_writer_file_prefix(ch5.name())}.opt.json")
    assert filename.exists()  # TEST make sure that write_opt_history works

    with filename.open() as f:
        opt_result = json.load(f)
        rxn_path = opt_result["extras"]["irc_rxn_path"]

    qs = np.asarray([[step_dict["q"][val] for step_dict in rxn_path] for val in [3, 4, 12]])

    with open(f"{test_dir}/CH5_irc_coords.json") as f:
        ch5_irc = json.load(f)

    # Check that point 11 on IRC matches closely
    label = 'f' if direction == "forward" else "b"
    assert np.isclose(qs[0][10], ch5_irc[f"coord_3_{label}"][10], atol=1e-5)
    assert np.isclose(qs[1][10], ch5_irc[f"coord_4_{label}"][10], atol=1e-5)
    assert np.isclose(qs[2][10], ch5_irc[f"coord_12_{label}"][10], atol=1e-5)

    # If IRC issues are encountered check plot
    # plot_irc(ch5_irc, qs)

    os.system(f"rm stdout.default.{os.getpid()}.hess")
    os.system(f"rm {filename}")

def plot_ch5_irc(irc, new_qs):
    import matplotlib.pyplot as plt

    b2ang = qcel.constants.bohr2angstroms
    coord_4 = np.asarray(irc["coord_4_b"][::-1] + irc["coord_4_f"][1:]) * b2ang
    coord_3 = np.asarray(irc["coord_3_b"][::-1] + irc["coord_3_f"][1:]) * b2ang
    coord_12 = np.asarray(irc["coord_12_b"][::-1] + irc["coord_12_f"][1:]) * 180 / np.pi
    coord_3_full = np.asarray(irc["coord_3_bf"][::-1] + irc["coord_3_ff"][1:]) * b2ang
    coord_12_full = np.asarray(irc["coord_12_bf"][::-1] + irc["coord_12_ff"][1:]) * 180 / np.pi

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(coord_3, coord_12, marker='X', s=80)
    ax[0].scatter(coord_3_full, coord_12_full, color='red', marker='+', s=80)
    ax[0].scatter(new_qs[0] * b2ang, new_qs[2] * 180 / np.pi, color='y', marker='D', s=80)
    ax[0].set(xlim=(1, 3), ylim=(95, 110), xticks=np.linspace(1, 3, 5), yticks=np.linspace(95, 110, 7))
    
    ax[1].scatter(coord_3, coord_4, marker='X', s=80)
    ax[1].scatter(new_qs[0] * b2ang, new_qs[1] * b2ang, color='y', marker='D', s=80)
    ax[1].set(xlim=(1, 3), ylim=(0, 3), xticks=np.linspace(1, 3, 5), yticks=np.linspace(0, 3, 7))
    plt.show()

