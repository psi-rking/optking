"""Test combinations of met criteria
tests vals * (max_vals / 100)  so that our random tests will approach the actual value as vals approaches 99.
Optking does not test for equality. Convergence is strictly less than so testing edge values
(i.e. whether a criteria is right at the limit is not necessary)
"""

import itertools

import pytest
import numpy as np

import optking

# Generate values over 2 orders of magnitude which approach the convergence threshold
rng = np.random.default_rng()
rng_vals = rng.integers(2, 98, size=10)
scalings = np.zeros(12)
scalings[0] = 1
scalings[-1] = 99
scalings[1:-1] = rng_vals

# Values are copied from optkings documentation to compare against optking internals

presets = {
    "NWCHEM_LOOSE": [None, 4.5e-3, 3.0e-3, 5.4e-3, 3.6e-3],
    "GAU_LOOSE": [None, 2.5e-3, 1.7e-3, 1.0e-2, 6.7e-3],
    # "GAU": [None, 4.5e-4, 3.0e-4, 1.8e-3, 1.2e-3],
    "CFOUR": [None, None, 1e-4, None, None],
    "QCHEM": [1.0e-6, 3.0e-4, None, 1.2e-3, None],
    # "MOLPRO": [1.0e-6, 3.0e-4, None, 3.0e-4, None],
    # "GAU_TIGHT": [None, 1.5e-5, 1.0e-5, 6.0e-5, 4.0e-5],
    "GAU_VERYTIGHT": [None, 2.0e-6, 1.0e-6, 6.0e-6, 4.0e-6],
    "TURBOMOLE": [1.0e-6, 1.0e-3, 5.0e-4, 1.0e-3, 5.0e-4],
}

program_mappings = {
    "GAU_LOOSE": "GAU",
    "GAU": "GAU",
    "GAU_TIGHT": "GAU",
    "GAU_VERYTIGHT": "GAU",
    "CFOUR": "CFOUR",
    "MOLPRO": "Q-M",
    "QCHEM": "Q-M",
    "TURBOMOLE": "TURBO",
    "NWCHEM_LOOSE": "NWCHEM",
}

tests = {
    1: {
        "increment": [0, 0, 0, 0, 0],  # All met
        "GAU": True,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": True,
        "NWCHEM": True,
    },
    2: {
        "increment": [2, 0, 0, 0, 0],  # energy not met
        "GAU": True,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": True,
    },
    3: {
        "increment": [0, 2, 0, 0, 0],  # max_force not met
        "GAU": False,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    4: {
        "increment": [0, 0, 2, 0, 0],  # rms_force not met
        "GAU": False,
        "Q-M": True,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    5: {
        "increment": [0, 0, 0, 2, 0],  # max_disp not met
        "GAU": False,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    6: {
        "increment": [0, 0, 0, 0, 2],  # rms_disp not met
        "GAU": False,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    7: {
        "increment": [2, 2, 0, 0, 0],  # rms_force and * disp not met
        "GAU": False,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    8: {
        "increment": [2, 0, 2, 0, 0],  # max_force and *disp not met
        "GAU": False,
        "Q-M": True,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    9: {
        "increment": [2, 0, 0, 2, 0],  # forces and rms_disp met
        "GAU": False,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    10: {
        "increment": [2, 0, 0, 0, 2],  # forces and max_disp met
        "GAU": False,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    11: {
        "increment": [0, 2, 2, 0, 0],  # energy and *disp met
        "GAU": False,
        "Q-M": False,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    12: {
        "increment": [0, 2, 0, 2, 0],  # energy and rms* met
        "GAU": False,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    13: {
        "increment": [0, 2, 0, 0, 2],  # energy rms_force max_disp met
        "GAU": False,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    14: {
        "increment": [0, 0, 2, 2, 0],  # energy max_force rms_disp met
        "GAU": False,
        "Q-M": True,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    15: {
        "increment": [0, 0, 2, 0, 2],  # energy max* met
        "GAU": False,
        "Q-M": True,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    16: {
        "increment": [0, 0, 0, 2, 2],  # energy * force met
        "GAU": False,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    17: {
        "increment": [2, 2, 2, 0, 0],  # *disp met
        "GAU": False,
        "Q-M": False,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    18: {
        "increment": [2, 2, 0, 2, 0],  # rms* met
        "GAU": False,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    19: {
        "increment": [2, 2, 0, 0, 2],  # rms_force max_disp met
        "GAU": False,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    20: {
        "increment": [2, 0, 2, 2, 0],  # max_force rms_disp met
        "GAU": False,
        "Q-M": False,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    21: {
        "increment": [2, 0, 2, 0, 2],  # max*
        "GAU": False,
        "Q-M": True,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    22: {
        "increment": [2, 0, 0, 2, 2],  # *forces met
        "GAU": False,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    23: {
        "increment": [0, 2, 2, 2, 0],  # enegy and rms_disp met
        "GAU": False,
        "Q-M": False,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    24: {
        "increment": [0, 2, 2, 0, 2],  # energy and max_disp met
        "GAU": False,
        "Q-M": False,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    25: {
        "increment": [0, 2, 0, 2, 2],  # energy and rms_force met
        "GAU": False,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    26: {
        "increment": [0, 0, 2, 2, 2],  # energy and max_force met
        "GAU": False,
        "Q-M": True,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    27: {
        "increment": [2, 2, 2, 2, 0],  # rms_disp met
        "GAU": False,
        "Q-M": False,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    28: {
        "increment": [2, 2, 2, 0, 2],  # max_disp met
        "GAU": False,
        "Q-M": False,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    29: {
        "increment": [2, 2, 0, 2, 2],  # rms_force met
        "GAU": False,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    30: {
        "increment": [2, 0, 2, 2, 2],  # max_force met
        "GAU": False,
        "Q-M": False,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    31: {
        "increment": [0, 2, 2, 2, 2],  # energy met
        "GAU": False,
        "Q-M": False,
        "CFOUR": False,
        "TURBO": False,
        "NWCHEM": False,
    },
    32: {
        "increment": [0, 0, -2, 0, 0],  # flat_potential all met
        "GAU": True,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": True,
        "NWCHEM": True,
    },
    33: {
        "increment": [0, 2, -2, 2, 2],  # flat_potential energy met
        "GAU": True,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    34: {
        "increment": [2, 0, -2, 2, 2],  # flat_potential max_force met
        "GAU": True,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    35: {
        "increment": [2, 2, -2, 0, 2],  # flat_potential max_disp met
        "GAU": True,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    36: {
        "increment": [2, 2, -2, 2, 0],  # flat_potential rms_disp
        "GAU": True,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    37: {
        "increment": [0, 2, -2, 2, 0],  # flat_potential energy rms_disp met
        "GAU": True,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    38: {
        "increment": [0, 2, -2, 0, 2],  # flat_potential energy max_disp met
        "GAU": True,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    39: {
        "increment": [0, 0, -2, 2, 2],  # flat_potential energy max_force met
        "GAU": True,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    40: {
        "increment": [2, 0, -2, 0, 2],  # flat_potential max* met
        "GAU": True,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    41: {
        "increment": [2, 0, -2, 2, 0],  # flat_potential max_force rms_disp met
        "GAU": True,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    42: {
        "increment": [2, 2, -2, 0, 0],  # flat_potential disps met
        "GAU": True,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    43: {
        "increment": [2, 0, -2, 0, 0],  # flat_potential energy not met
        "GAU": True,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": True,
    },
    44: {
        "increment": [0, 2, -2, 0, 0],  # flat_potential max_force not met
        "GAU": True,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    45: {
        "increment": [0, 0, -2, 2, 0],  # flat_potential max_disp not met
        "GAU": True,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    46: {
        "increment": [0, 0, -2, 0, 2],  # flat_potential rms_disp not met
        "GAU": True,
        "Q-M": True,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
    47: {
        "increment": [2, 2, -2, 2, 2],  # flat_potential none met
        "GAU": True,
        "Q-M": False,
        "CFOUR": True,
        "TURBO": False,
        "NWCHEM": False,
    },
}

@pytest.mark.parametrize("conv_test", tests)
@pytest.mark.parametrize("conv_preset", presets)
def test_convergence_presets(conv_test, conv_preset):
    """test gaussian for flat_potential and all criteria met, flat potential and only rms_force,
    all criteria and not flat_potential

    Notes
    -----
    Increments is an instruction set to create a series of numbers up to ~ 100 times greater or down to ~100 times smaller
    than the default convergence threshold

    """

    changes = tests.get(conv_test).get("increment")
    defaults = presets.get(conv_preset)

    for index, value in enumerate(defaults):
        if value is None:
            defaults[index] = rng.random()

    for random_scale in rng_vals:
        criteria = _create_variations(defaults, changes, random_scale)

        params = optking.optwrapper.initialize_options({"g_convergence": conv_preset})
        params_dict = params.conv_criteria()

        # Actual conv_check test.
        conv_met, conv_active = optking.convcheck._transform_criteria(criteria, params_dict)
        conv_met.update(
            {"flat_potential": 100 * criteria.get("rms_force") < params_dict.get("conv_rms_force")}
        )
        state = optking.convcheck._test_for_convergence(conv_met, conv_active, params=params)
        expected = tests.get(conv_test).get(program_mappings.get(conv_preset))

        assert state == expected


options = [
    "max_energy_g_convergence",
    "max_force_g_convergence",
    "rms_force_g_convergence",
    "max_disp_g_convergence",
    "rms_disp_g_convergence",
]

conv_mapping = {
    "max_energy_g_convergence": "max_DE",
    "max_force_g_convergence": "max_force",
    "rms_force_g_convergence": "rms_force",
    "max_disp_g_convergence": "max_disp",
    "rms_disp_g_convergence": "rms_disp",
}

preset_subset = ["QCHEM", "GAU_TIGHT", "NWCHEM_LOOSE"]

# only testing up combinations of user settings up to 2
combos = [
    val
    for i in range(1, len(options))
    for index, val in enumerate(itertools.combinations(options, i))
    if index < 1
]
# remove even tests for brevity

for i in range(1, len(tests), 2):
    tests.pop(i)

@pytest.mark.parametrize("conv_options", combos)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("preset", preset_subset)
@pytest.mark.parametrize("flexible_on", (True, False))
def test_user_tampering(conv_options, test_name, preset, flexible_on):
    test = tests.get(test_name)
    changes = test.get("increment")

    print(conv_options)
    options_dict = {key: 1e-5 for key in conv_options}
    options_dict.update({"flexible_g_convergence": flexible_on, "g_convergence": preset})
    params = optking.optwrapper.initialize_options(options_dict)

    keys = ["max_DE", "max_force", "rms_force", "max_disp", "rms_disp"]

    opt_params = params.conv_criteria()
    thresh1 = [opt_params.get(f"conv_{key}") for key in keys]
    thresholds = [val if val > 0 else 0.1 for val in thresh1]

    for random_scale in rng_vals:
        criteria = _create_variations(thresholds, changes, random_scale)

        conv_met, conv_active = optking.convcheck._transform_criteria(criteria, opt_params)
        conv_met.update(
            {"flat_potential": 100 * criteria.get("rms_force") < opt_params.get("conv_rms_force")}
        )
        state = optking.convcheck._test_for_convergence(conv_met, conv_active, params=params)

        extra_requirements = [conv_mapping.get(val) for val in conv_options]
        # all keys that should be met based on changes
        chosen = [keys[index] for index, val in enumerate(changes) if val <= 0]

        extra_flag = True
        for key in extra_requirements:
            if key in chosen:
                if not conv_met.get(key):
                    # optking doesn't think criteria is met but should be
                    extra_flag = False
            else:
                if conv_met.get(key):
                    # optking thinks criteria is met but shouldn't be
                    extra_flag = False

        if flexible_on:
            default_satisfied = test.get(program_mappings.get(preset))
            assert state == (default_satisfied and extra_flag)
        else:
            assert extra_flag


def _create_variations(thresholds, changes, coefficient):
    criteria = {"max_DE": 1, "max_force": 1, "rms_force": 1, "max_disp": 1, "rms_disp": 1}
    conv_criteria_names = list(criteria.keys())
    magnitudes = np.floor(np.log10(thresholds))

    for index, val in enumerate(changes):
        if val > 0:
            # increase by two orders of magnitude then bring back down (near the threshold)
            temp = thresholds[index] * 100 + rng.random() * (10 ** (magnitudes[index] + 2))
            criteria[conv_criteria_names[index]] = temp / coefficient
            assert temp / coefficient > thresholds[index]
        elif val < 0:
            # decrease by two orders then decrease further!
            temp = thresholds[index] / 100 - rng.random() * (10 ** (magnitudes[index] - 2))
            criteria[conv_criteria_names[index]] = temp / coefficient
            assert temp / coefficient < thresholds[index]
        else:
            # make sure criteria ends up below the threshold but above flat_potential threshold
            # for random_scale = 1 and below original for random_scale = 99
            temp = thresholds[index] / 100 + rng.random() * (10 ** (magnitudes[index] - 5))
            criteria[conv_criteria_names[index]] = temp * coefficient
            assert thresholds[index] / 100 < temp * coefficient < thresholds[index]

    return criteria
