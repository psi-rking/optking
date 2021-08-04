import logging
from math import fabs

import numpy as np

from . import optparams as op
from .linearAlgebra import abs_max, rms
from .printTools import print_array_string, print_mat_string

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


def conv_check(iternum, o_molsys, dq, f, energies, irc_data=None):
    """Wrapper method to test for stationary point convergence

    Computes energy, force, and displacement changes for current step. Prints main convergence report
    (printed every step).


    Returns
    -------
    True if geometry is optimized.

    Notes
    -----
    Default convergence checks maximum force component and (Delta(E) or maximum displacement)

    Convergence for each IRC step uses uses forces perpendicular to the second halfstep and tangential to the
    hypersphere on which the constrained optimization is performed.

    """

    logger = logging.getLogger(__name__)
    logger.info("Performing convergence check.")

    params_dict = op.Params.__dict__
    criteria = _get_conv_criteria(o_molsys, dq, f, energies, irc_data)
    conv_met, conv_active = _transform_criteria(criteria, params_dict)
    _print_convergence_table(iternum, energies[-1], criteria, conv_met, conv_active, params_dict)

    # flat potential cannot be activated by the user - purely an internal tool for gau_type convergence
    conv_met.update({"flat_potential": 100 * criteria.get("rms_force") < op.Params.conv_rms_force})
    return _test_for_convergence(conv_met, conv_active)


def _get_conv_criteria(o_molsys, dq, f, energies, irc_data=None):
    """ Creates a dictionary of step information for convergence test """

    energy = energies[-1]
    last_energy = energies[-2] if len(energies) > 1 else 0.0

    if op.Params.opt_type == "IRC":
        f_vec = irc_data._project_forces(f, o_molsys)
    else:
        f_vec = f

    criteria = {
        "max_DE": energy - last_energy,
        "max_force": abs_max(f_vec),
        "rms_force": rms(f_vec),
        "max_disp": abs_max(dq),
        "rms_disp": rms(dq),
    }

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
    """ Return symbol for inactive: "o" met: "*" or unmet: " " """

    symbol = "*" if criteria_met else ""

    if not criteria_active:
        symbol = "o"

    return symbol


def _test_for_convergence(conv_met, conv_active):
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

    logger = logging.getLogger(__name__)

    if op.Params.i_untampered:
        # flexible_criteria forces this route, but with an adjusted value for an individual criteria
        if "GAU" in op.Params.g_convergence:
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

    if converged and op.Params.opt_type != "IRC":
        logger.info("%s", _print_active_criteria(conv_status, conv_requirements))

    return converged


def _print_convergence_table(iternum, energy, criteria, conv_met, conv_active, params_dict):
    """Print a nice looking table for the current step """

    logger = logging.getLogger(__name__)

    conv_symbols = {key: _get_criteria_symbol(conv_met.get(key), conv_active.get(key)) for key in conv_met}
    print_vals = [iternum + 1, energy] + list(criteria.values()) + list(conv_symbols.values())  # All columns

    suffix = "~\n" if iternum == 0 else "\n"

    conv_str = f"""\n\t{'==> Convergence Check <==': ^92}
    \n\tMeasures of convergence in internal coordinates in au.
    \n\tCriteria marked as inactive (o), active & met (*), and active & unmet ( ).\n\n"""
    conv_str += "\t" + "-" * 94 + suffix

    conv_str += "\t  Step     Total Energy     Delta E     MAX Force     RMS Force      MAX Disp      RMS Disp   "
    conv_str += suffix
    conv_str += "\t" + "-" * 94 + suffix

    # For each active criteria print the target value and the met/inactive/unmet symbol
    # easier to just redetermine instead of adapt conv_symbols above
    active = lambda x: f"{params_dict.get('conv_' + x) :11.2e} {'*'}"
    conv_active_str = [active(key) if conv_active.get(key) else f"{'o': >13}" for key in conv_active]

    conv_str += "\t  Convergence Criteria  "
    conv_str += " ".join(conv_active_str)
    conv_str += suffix
    conv_str += "\t" + "-" * 94 + suffix
    conv_str += (
        "\t  {0:4d} {1:16.8f} {2:11.2e} {7:1s} {3:11.2e} {8:1s} {4:11.2e} {9:1s} {5:11.2e} {10:1s} {6:11.2e} {11:1s}"
        "  ~\n"
    ).format(*print_vals)

    conv_str += "\t" + "-" * 94 + "\n\n"
    logger.info(conv_str)


def _print_active_criteria(conv_status, conv_requirements):
    """Get string describing all convergence criteria being considered for the optimization """
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
