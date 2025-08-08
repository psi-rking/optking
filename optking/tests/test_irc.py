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

@pytest.mark.long
@pytest.mark.parametrize(
    "direction, NUM_POINTS, REF_ENERGY",
    (
        ["forward", 11, -190.133172125480],  # 11 because we've computed 10 points (after TS)
        ["backward", 9, -190.129671982538],  # Very short IRC in forward direction
    ),
)
def test_irc_CH3O2(direction, NUM_POINTS, REF_ENERGY):

    # Random IRC calculation that was ending before reaching a minimum

    psi4.geometry("""
        0 2
        C           -1.189911690573     0.201247528125     0.167673149223
        O           -0.091451900076    -0.485249388360    -0.243225315637
        H           -1.021145249899     1.181999865559     0.590345104870
        H           -2.058253276210    -0.023216256085    -0.441170124838
        H            1.572684844453    -0.372798307107     0.586272547903
        O            1.079106065627     0.284741534678     0.071090591276
        nocom
        noreorient
	""")

    psi4_options = {
        "basis": "6-31G(D)",
        "reference": "uhf",
        "maxiter": 200,
        "opt_type": "IRC",
        "full_hess_every": 0,
        "g_convergence": "gau_tight",
        "irc_direction": direction,
        "irc_step_size": 0.2,
        "irc_points": 10,
        "geom_maxiter": 200,
    }

    psi4.set_options(psi4_options)
    optking_options = {"hessian_file": f"{test_dir}/test_data/CH3O2_irc.hess"}

    # IRC
    json_output = optking.optimize_psi4("MPW1PW", **optking_options)
    rxn_path = json_output['extras']['irc_rxn_path']
    energy = rxn_path[-1]["energy"]
    assert np.isclose(len(rxn_path), NUM_POINTS)
    assert np.isclose(REF_ENERGY, energy, atol=1e-4)


@pytest.mark.parametrize(
    "direction, NUM_POINTS, REF_ENERGY",
    (
        ["forward", 22, -91.644039001250],
        ["backward", 20, -91.674187446520]  # ref values are from new optking. Disagreement with old
        # optking because PYOPTKING adds linear coordinates. Matches up to 1 or 2 points before
        # terminating. linear bends only get added at end.
    ),
)
def test_irc_HCN(direction, NUM_POINTS, REF_ENERGY):
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
            "irc_points": 23,  # just enough to make sure we complete
            "geom_maxiter": 300,
        }
    )

    optking_options = {"hessian_file": f"{test_dir}/test_data/HCN_irc.hess"}
    json_output = optking.optimize_psi4("HF", **optking_options)
    rxn_path = json_output["extras"]["irc_rxn_path"]
    assert np.isclose(REF_ENERGY, rxn_path[-1]["energy"], atol=1e-5)
    assert len(rxn_path) == NUM_POINTS

with open(f'{test_dir}/test_data/CH5_irc_ref.json', 'r+') as f:
    ch5_ref = json.load(f)

@pytest.mark.parametrize(
    "direction, coords, energies",
    (
        ["backward", ch5_ref["CH5_BACKWARD_REF_COORDS"], ch5_ref["CH5_BACKWARD_REF_ENS"]],
        ["forward", ch5_ref["CH5_FORWARD_REF_COORDS"], ch5_ref["CH5_FORWARD_REF_ENS"]]
    ),
)
def test_irc_CH5(direction, coords, energies):
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

    # Collect values for coords, and enegies. Reference data may / will be longer than ours
    q3 = [step_dict["q"][3] for step_dict in rxn_path]
    q4 = [step_dict["q"][4] for step_dict in rxn_path]
    q12 = [step_dict["q"][12] for step_dict in rxn_path]
    irc_energies = [step_dict["energy"] for step_dict in rxn_path]

    with open(f"{test_dir}/CH5_irc_coords.json") as f:
        ch5_irc = json.load(f)

    # Check that point 11 on IRC matches closely
    # label = 'f' if direction == "forward" else "b"

    for e_ref, e in zip(energies, irc_energies):
        assert np.isclose(e_ref, e, atol=2e-5)

    for q, q3_ref in zip(q3, coords["q3"]):
        assert np.isclose(q3_ref, q, atol=2e-4)

    for q, q4_ref in zip(q4, coords["q4"]):
        assert np.isclose(q4_ref, q, atol=2e-4)

    # If IRC issues are encountered check plot
    # qs = np.asarray([q3, q4, q12])
    # plot_ch5_irc(ch5_irc, qs)

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

