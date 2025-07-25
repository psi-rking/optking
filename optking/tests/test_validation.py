import optking
import pytest

""" Check that various keyword combinations are being processed correctly.
This list of tests is of course not exhaustive. New tests should be added for any keywords that
depend upon one another. """


@pytest.mark.parametrize(
    "options, expected_vals",
    [
        ({"OPT_TYPE": "IRC", "IRC_POINTS": 5}, 5 * 15),
        ({"GEOM_MAXITER": 200}, 200),
        ({"OPT_TYPE": "IRC", "GEOM_MAXITER": 200, "IRC_POINTS": 10}, 200),
        ({"OPT_TYPE": "IRC", "GEOM_MAXITER": 200}, 200),
        ({"IRC_POINTS": 5, "GEOM_MAXITER": 40}, 40),
        ({"OPT_TYPE": "IRC", "IRC_POINTS": 5, "GEOM_MAXITER": 40}, 40),
        ({"opt_type": "irc", "irc_points": 5}, 5 * 15),
        ({"geom_maxiter": 200}, 200),
        ({"opt_type": "irc", "geom_maxiter": 200, "irc_points": 10}, 200),
        ({"opt_type": "irc", "geom_maxiter": 200}, 200),
        ({"irc_points": 5, "geom_maxiter": 40}, 40),
        ({"opt_type": "irc", "irc_points": 5, "geom_maxiter": 40}, 40),

    ],
)
def test_iter(options, expected_vals):
    params = optking.op.OptParams(**options)
    assert params.geom_maxiter == expected_vals


@pytest.mark.parametrize(
    "options, expected_vals",
    [
        (
            {"OPT_TYPE": "IRC", "HESSIAN_FILE": "./irc_hooh.hess"},
            {"CART_HESS_READ": True, "HESS_UPDATE": "BOFILL"},
        ),
        (
            {"OPT_TYPE": "IRC"},
            {"HESS_UPDATE": "BOFILL", "FULL_HESS_EVERY": 0, "INTRAFRAG_STEP_LIMIT": 0.2},
        ),
        (
            {"opt_type": "irc", "hessian_file": "./irc_hooh.hess"},
            {"cart_hess_read": True, "hess_update": "bofill"},
        ),
        (
            {"opt_type": "irc"},
            {"hess_update": "bofill", "full_hess_every": 0, "intrafrag_step_limit": 0.2},
        ),
    ],
)
def test_irc_defaults(options, expected_vals):
    params = optking.op.OptParams(**options)
    for key, val in expected_vals.items():
        if isinstance(val, str) and key.upper() != "HESSIAN_FILE":
            compare = val.upper()
        else:
            compare = val
        assert params.to_dict(by_alias=True).get(key.upper()) == compare


options = [
    ({"STEP_TYPE": "RS_I_RFO"}, {"INTRAFRAG_STEP_LIMIT": 0.2, "OPT_TYPE": "TS"}),
    ({"OPT_TYPE": "TS"}, {"STEP_TYPE": "RS_I_RFO", "INTRAFRAG_STEP_LIMIT": 0.2}),
    (
        {"OPT_TYPE": "TS", "INTRAFRAG_STEP_LIMIT": 0.1},
        {"STEP_TYPE": "RS_I_RFO", "INTRAFRAG_STEP_LIMIT": 0.1},
    ),
    ({"step_type": "rs_i_rfo"}, {"intrafrag_step_limit": 0.2, "opt_type": "ts"}),
    ({"opt_type": "ts"}, {"step_type": "rs_i_rfo", "intrafrag_step_limit": 0.2}),
    (
        {"opt_type": "ts", "intrafrag_step_limit": 0.1},
        {"step_type": "rs_i_rfo", "intrafrag_step_limit": 0.1},
    ),
]

@pytest.mark.parametrize("options, expected_vals", options)
def test_ts_defaults(options, expected_vals):
    params = optking.op.OptParams(**options)
    for key, val in expected_vals.items():
        # Whatever we get from the params dict will be capitalized except for HESSIAN_FILE
        # which is case sensitive. Just checking that users can use lower case
        if isinstance(val, str):
            compare = val.upper()
        else:
            compare = val
        assert params.to_dict(by_alias=True).get(key.upper()) == compare


def test_convergence():
    """Check that active convergence criteria can be chosen correctly and that a preset can be chosen."""

    # Confirm g_convergence behavior before checking edge cases with exchaustive code
    params = optking.op.OptParams(**{"G_CONVERGENCE": "GAU_TIGHT"})

    assert params.g_convergence == "GAU_TIGHT"
    assert params.conv_max_force == 1.5e-5
    assert params.conv_rms_force == 1.0e-5
    assert params.conv_max_disp == 6.0e-5
    assert params.conv_rms_disp == 4.0e-5

    assert params._i_max_force
    assert params._i_rms_force
    assert params._i_max_disp
    assert params._i_max_force
    assert not params._i_max_DE

    # Discards G_CONVERGENCE preset
    params = optking.op.OptParams(
        **{"G_CONVERGENCE": "GAU_TIGHT", "RMS_FORCE_G_CONVERGENCE": 1.0e-6}
    )
    assert params.g_convergence == "GAU_TIGHT"
    assert params.conv_max_force == 1.5e-5
    assert params.conv_rms_force == 1.0e-6
    assert params.conv_max_disp == 6.0e-5
    assert params.conv_rms_disp == 4.0e-5

    assert not params._i_max_force
    assert params._i_rms_force
    assert not params._i_max_disp
    assert not params._i_max_force
    assert not params._i_max_DE

    # Discards G_CONVERGENCE preset
    params = optking.op.OptParams(
        **{"g_convergence": "gau_tight", "rms_force_g_convergence": 1.0E-6}
    )
    assert params.g_convergence == "GAU_TIGHT"
    assert params.conv_max_force == 1.5e-5
    assert params.conv_rms_force == 1.0e-6
    assert params.conv_max_disp == 6.0e-5
    assert params.conv_rms_disp == 4.0e-5

    assert not params._i_max_force
    assert params._i_rms_force
    assert not params._i_max_disp
    assert not params._i_max_force
    assert not params._i_max_DE

    # Keep G_CONVERGENCE preset and update value for RMS_FORCE
    params = optking.op.OptParams(
        **{
            "G_CONVERGENCE": "GAU_TIGHT",
            "RMS_FORCE_G_CONVERGENCE": 1.0e-6,
            "flexible_g_convergence": True,
        }
    )
    assert params.g_convergence == "GAU_TIGHT"
    assert params.flexible_g_convergence
    assert params.conv_max_force == 1.5e-5
    assert params.conv_rms_force == 1.0e-6
    assert params.conv_max_disp == 6.0e-5
    assert params.conv_rms_disp == 4.0e-5

    assert params._i_max_force
    assert params._i_rms_force
    assert params._i_max_disp
    assert params._i_max_force
    assert not params._i_max_DE
