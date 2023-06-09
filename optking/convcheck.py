""" Functions for checking the convergence of the optimization procedure.

Methods
-------
conv_check: Primary wrapper. Take information (as dictionary) from previous step(s). print summary or return summary string

"""


import logging
from math import fabs

import numpy as np

from . import optparams as op
from .linearAlgebra import abs_max, rms
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")

CONVERGENCE_PRESETS = {
    "QCHEM_MOLPRO": {"required": ["max_force"], "one of": ["max_DE", "max_disp"], "alternate": [None]},
    "GAUSSIAN": {
        "required": ["max_force", "rms_force", "max_disp", "rms_disp"],
        "one of": [None],
        "alternate": ["flat_potential"],
    },
    "IRC": {"required": ["max_DE, max_disp, rms_disp"], "one of": [None], "alternate": [None]},
    "TURBOMOLE": {
        "required": ["max_force", "rms_force", "max_DE", "max_disp", "rms_disp"],
        "one of": [None],
        "alternate": [None],
    },
    "CFOUR": {"required": ["rms_force"], "one of": [None], "alternate": [None]},
    "NWCHEM_LOOSE": {
        "required": ["max_force", "rms_force", "max_disp", "rms_disp"],
        "one of": [None],
        "alternate": [None],
    },
}


def conv_check(conv_info: dict, params: dict, required=None, str_mode=""):
    """
    Parameters
    ----------
    conv_info: dict
        A dictionary providing the following keys
        step_type: "IRC", "STANDARD" (DEFAULT) or "LINESEARCH (experimental)
        dq: np.ndarray
        fq: np.ndarray
        energies: tuple
        iternum: int
        sub_step_number: int
            DEFAULT None
    required: str (optional)
        one of "ENERGY", "GRADIENT", "HESSIAN" what calculations does the optimization require. Cannot
        check convergence if only ENERGY
    str_mode: str (optional)
        `table` just the string form is desired
        `both` if the table is desired in string form and printing is desired
        '' if no string is desired

    Returns
    -------
    return_str: if requested
    """

    return_str = True if str_mode in ["both", "table"] else False
    if not return_str:
        logger.info("Performing convergence check.")

    criteria = _get_conv_criteria(conv_info.get("dq"), conv_info.get("fq"), conv_info.get("energies"), required)
    conv_met, conv_active = _transform_criteria(criteria, params)
    conv_str = _print_convergence_table(conv_info, criteria, conv_met, conv_active, params, return_str)

    # flat potential cannot be activated by the user - purely an internal tool for gau_type convergence
    conv_met.update({"flat_potential": 100 * criteria.get("rms_force") < op.Params.conv_rms_force})

    if str_mode == "table":
        return conv_str
    elif str_mode == "both":
        return conv_str, _test_for_convergence(conv_met, conv_active, return_str)
    else:
        return _test_for_convergence(conv_met, conv_active)


def _get_conv_criteria(dq, f_vec, energies, required=None):
    """Creates a dictionary of step information for convergence test"""

    energy = energies[-1]
    last_energy = energies[-2] if len(energies) > 1 else 0.0

    criteria = {
        "max_DE": energy - last_energy,
        "max_force": abs_max(f_vec),
        "rms_force": rms(f_vec),
        "max_disp": abs_max(dq),
        "rms_disp": rms(dq),
    }

    if required:
        if "gradient" not in required:
            criteria.pop("rms_force")
            criteria.pop("max_force")

    return criteria


def _transform_criteria(criteria, params_dict):
    """create dictionaries with boolean values indicating whether a criteria is active and met

    Parameters
    ----------
    criteria : dict
        contains keys for each convergence criteria and the current value for this step

    Returns
    -------
    (dict, dict)
        Two dictionaries with same keys as input containing bools for whether the criteria is met and
        whether the criteria is active (True) or not

    """

    conv_met = {key: fabs(val) < params_dict.get(f"conv_{key}") for key, val in criteria.items()}
    conv_active = {key: params_dict.get(f"i_{key}") for key in criteria}

    return conv_met, conv_active


def _get_criteria_symbol(criteria_met, criteria_active):
    """Return symbol for inactive: "o" met: "*" or unmet: " " """

    symbol = "*" if criteria_met else ""

    if not criteria_active:
        symbol = "o"

    return symbol


def _test_for_convergence(conv_met, conv_active, return_str=False):
    """Test whether the current point is sufficiently converged. Have all needed criteria been met.

    Parameters
    ----------
    conv_met : dict
        values are True if criteria is met False otherwise
    conv_active : dict
        values are True if criteria is active False otherwise

    Notes
    -----

    If one or more convergence criteria are specifically set by the user, no other criteria are
    used to check for convergence.

    """

    if op.Params.i_untampered:
        # flexible_criteria forces this route, but with an adjusted value for an individual criteria
        if "GAU" in op.Params.g_convergence or op.Params.g_convergence == "INTERFRAG_TIGHT":
            conv_requirements = CONVERGENCE_PRESETS.get("GAUSSIAN")
        elif op.Params.g_convergence in ["QCHEM", "MOLPRO"]:
            conv_requirements = CONVERGENCE_PRESETS.get("QCHEM_MOLPRO")
        else:
            conv_requirements = CONVERGENCE_PRESETS.get(op.Params.g_convergence)

    else:
        conv_requirements = {
            "required": [key for key in conv_active if conv_active.get(key)],
            "one of": [None],
            "alternate": [None],
        }

    # mirrors the requirements but with booleans indicating whether each condition is met
    conv_status = {
        key: [conv_met.get(item, True) if key == "one of" else conv_met.get(item, False) for item in val_list]
        for key, val_list in conv_requirements.items()
    }

    converged = False
    if all(conv_status.get("required")) and any(conv_status.get("one of")):
        converged = True

    if all(conv_status.get("alternate")):
        converged = True

    if return_str:
        return _print_active_criteria(conv_status, conv_requirements)
    if converged and op.Params.opt_type != "IRC":
        logger.info("%s", _print_active_criteria(conv_status, conv_requirements))

    return converged


def _print_convergence_table(conv_info, criteria, conv_met, conv_active, params_dict, return_str=False):
    """Print a nice looking table for the current step"""

    # define all the variable strings that can get used below
    std_header = (
        f"\t {'Step': ^8}{'Total Energy': ^17}{'Delta E': ^12}{'Max Force': ^14}{'RMS Force': ^14}"
        + f"{'Max Disp': ^14}{'RMS Disp': ^14}"
    )
    std_vals = (
        "\t  {0:4d} {1:16.8f} {2:11.2e} {7:1s} {3:11.2e} {8:1s} {4:11.2e} {9:1s} {5:11.2e} {10:1s}"
        + " {6:11.2e} {11:1s}  ~\n"
    )

    irc_header = (
        f"\t {'Step': ^8}{'Sphere Step': ^16}{'Total Energy': ^17}{'Delta E': ^12}{'Max Force': ^14}"
        + f"{'RMS Force': ^14}{'Max Disp': ^14}{'RMS Disp': ^14}"
    )
    irc_vals = (
        "\t  {0:4d}     {1:^11d} {2:16.8f} {3:11.2e} {8:1s} {4:11.2e} {9:1s} {5:11.2e} {10:1s} {6:11.2e}"
        + " {11:1s} {7:11.2e} {12:1s}  ~\n"
    )

    # Get all the values for convergence critera and active criteria markers
    conv_symbols = {key: _get_criteria_symbol(conv_met.get(key), conv_active.get(key)) for key in conv_met}
    print_vals = (
        [conv_info.get("iternum"), conv_info.get("energies")[-1]]
        + list(criteria.values())
        + list(conv_symbols.values())
    )  # All columns

    # For each active criteria print the target value and the met/inactive/unmet symbol
    # easier to just redetermine instead of adapt conv_symbols above
    active = lambda x: f"{params_dict.get('conv_' + x) :11.2e} {'*'}"
    conv_active_str = [active(key) if conv_active.get(key) else f"{'o': >13}" for key in conv_active]

    suffix = "~\n" if conv_info.get("iternum") == 1 else "\n"

    # adjust printing spacing for irc
    if conv_info.get("step_type", "standard") == "standard":
        dash_length, extra_irc_space, header = 94, "", std_header
    else:
        dash_length, extra_irc_space, header = 114, " " * 16, irc_header

    conv_str = f"""\n\t{'==> Convergence Check <==': ^92}
    \n\tMeasures of convergence in internal coordinates in au.
    \n\tCriteria marked as inactive (o), active & met (*), and active & unmet ( ).\n\n"""
    conv_str += "\t" + "-" * dash_length + suffix
    conv_str += header
    conv_str += suffix
    conv_str += "\t" + "-" * dash_length + suffix
    conv_str += "\t  Convergence Criteria  " + extra_irc_space
    conv_str += " ".join(conv_active_str)
    conv_str += suffix
    conv_str += "\t" + "-" * dash_length + suffix

    # unpack all information for the step: energy, force max / rms etc into strings
    if conv_info.get("step_type") == "standard":
        conv_str += std_vals.format(*print_vals)
    else:
        print_vals = print_vals[:1] + [conv_info["sub_step_num"]] + print_vals[1:]
        conv_str += irc_vals.format(*print_vals)

    conv_str += "\t" + "-" * dash_length + "\n\n"
    if return_str:
        return conv_str
    logger.info(conv_str)


def _print_active_criteria(conv_status, conv_requirements):
    """Get string describing all convergence criteria being considered for the optimization"""
    conv_str = f"\n\t {'===> Final Convergence Report <===': ^76}\n"
    conv_str += "\n\t" + "-" * 76
    conv_str += f"\n\t|{'Required Criteria': ^24}|{'One or More Criteria': ^24}|{'Alternate Criteria': ^24}|"
    conv_str += "\n\t" + "-" * 76

    print_len = max(
        len(conv_status.get("required")), max(len(conv_status.get("one of")), len(conv_status.get("alternate")))
    )

    for i in range(print_len):

        conv_str += "\n\t|"

        for key in conv_status:

            # conv_requirments[key] is an empty list if no criteria match (not guaranteed for conv_status)
            if conv_requirements.get(key)[0] is None or i >= len(conv_status.get(key)):
                conv_str += f"{'': ^24}|"
            else:
                conv_str += f"{' [x]' if conv_status.get(key)[i] else ' [ ]'}{conv_requirements.get(key)[i]: ^20}|"

    conv_str += "\n\t" + "-" * 76 + "\n\n"
    return conv_str


#
# We MIGHT need to remove coordinates for analysis of convergence
# check, but this is not clear.
# 1. if frozen, fq was set to 0, and then displace tried its best
# to make sure dq was also zero..
# 2. if additional ext_force is applied, we presume total force
# should still -> 0.
# 3. If ranged and 'at wall', fq already set to 0.
#
# For now, saving here the old code to remove the deprecated 'fixed'
# has_fixed = False
# for F in o_molsys._fragments:
#    if any([ints.fixed_eq_val for ints in F.intcos]):
#        has_fixed = True
# for DI in o_molsys._dimer_intcos:
#    if any([ints.fixed_eq_val for ints in DI._pseudo_frag._intcos]):
#        has_fixed = True
# Remove arbitrary forces for user-specified equilibrium values.
# if has_fixed:
#    cnt = -1
#    for F in o_molsys.fragments:
#        for I in F.intcos:
#            cnt += 1
#            if I.fixed_eq_val:
#                f[cnt] = 0
#    for DI in o_molsys.dimer_intcos:
#        for I in DI.pseudo_frag.intcos:
#            cnt += 1
#            if I.fixed_eq_val:
#                f[cnt] = 0
