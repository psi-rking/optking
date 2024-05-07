# wrapper for optking's optimize function for input by psi4API
# creates a moleuclar system from psi4s and generates optkings options from psi4's lsit of options
import copy
import json
import logging

from qcelemental.models import OptimizationInput, OptimizationResult
from qcelemental.util.serialization import json_dumps

import optking

from . import caseInsensitiveDict, molsys
from . import optparams as op
from .compute_wrappers import ComputeWrapper, Psi4Computer, QCEngineComputer, UserComputer
from .exceptions import OptError
from .optimize import optimize
from .printTools import welcome
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")


def optimize_psi4(calc_name, program="psi4", dertype=None, **xtra_opt_params):
    """
    Wrapper for optimize.optimize() Looks for an active psi4 molecule and optimizes.
    This is the written warning that Optking will try to use psi4 if no program is provided
    Parameters
    ----------
    calc_name: str
        level of theory for optimization. eg MP2
    program: str
        program used for gradients, hessians...
    dertype: ?
        hack to try to get finite differences working in psi4
    xtra_opt_params: dictionary
        extra keywords currently forbidden by psi4's read_options, but supported by optking

    Returns
    -------
    opt_output: dict
        dictionary serialized MOLSSI OptimizationResult.
        see https://github.com/MolSSI/QCElemental/blob/master/qcelemental/models/procedures.py
    """

    opt_input = {}
    opt_output = {}

    try:
        op.Params, oMolsys, computer, opt_input = initialize_from_psi4(
            calc_name, program, computer_type="psi4", dertype=dertype, **xtra_opt_params
        )
        opt_output = copy.deepcopy(opt_input)
        opt_output = optimize(oMolsys, computer)
    except (OptError, KeyError, ValueError, AttributeError) as error:
        opt_output = {
            "success": False,
            "error": {"error_type": error.err_type, "error_message": error.mesg},
        }
        logger.critical(f"Error placed in qcschema: {opt_output}")
        logger.debug(str(opt_output))
    except Exception as error:
        logger.critical("An unknown error has occured and evaded error checking")
        opt_output = {
            "success": False,
            "error": {"error_type": error, "error_message": str(error)},
        }
        logger.critical(f"Error placed in qcschema: {opt_output}")
    finally:
        opt_input.update({"provenance": optking._optking_provenance_stamp})
        opt_input["provenance"]["routine"] = "optimize_psi4"
        opt_input.update(opt_output)
        return opt_input


def initialize_from_psi4(calc_name, program, computer_type, dertype=None, **xtra_opt_params):
    """Gathers information from an active psi4 instance. to cleanly run optking from a
    psithon or psi4api input file

    Parameters
    ----------
    calc_name: str
    computer_type: str
    dertype: Union[int, None]
    program: str
    **xtra_opt_params
        extra keywords which are not recognized by psi4

    Returns
    -------
    params: op.OptParams
    o_molsys: molsys.Molsys
    computer: ComputeWrapper
    opt_input: qcelemental.models.OptimizationInput

    """
    import psi4

    mol = psi4.core.get_active_molecule()
    o_molsys, qc_mol = molsys.Molsys.from_psi4(mol)

    # Get optking options and globals from psi4
    # Look through optking module specific options first. If a global has already appeared
    # in optking's options, don't include as a qc package option
    logger.debug("Getting module and psi4 options for qcschema construction")
    module_options = psi4.driver.p4util.prepare_options_for_modules()
    all_options = psi4.core.get_global_option_list()
    opt_keys = {"program": program}

    qc_keys = {}
    if dertype is not None:
        qc_keys["dertype"] = 0

    optking_options = module_options["OPTKING"]
    for opt, optval in optking_options.items():
        if optval["has_changed"]:
            opt_keys[opt.lower()] = optval["value"]

    if xtra_opt_params:
        for xtra_key, xtra_value in xtra_opt_params.items():
            opt_keys[xtra_key.lower()] = xtra_value

    for option in all_options:
        if psi4.core.has_global_option_changed(option):
            if option not in opt_keys:
                qc_keys[option.lower()] = psi4.core.get_global_option(option)

    # Make a qcSchema OptimizationInput
    opt_input = {
        "keywords": opt_keys,
        "initial_molecule": qc_mol,
        "input_specification": {
            "model": {"basis": qc_keys.pop("basis"), "method": calc_name},
            "driver": "gradient",
            "keywords": qc_keys,
        },
    }
    logger.debug("Creating OptimizationInput")

    opt_input = OptimizationInput(**opt_input)
    # Remove numpy elements to allow at will json serialization
    opt_input = json.loads(json_dumps(opt_input))

    initialize_options(opt_keys)
    params = op.Params
    computer = make_computer(opt_input, computer_type)
    return params, o_molsys, computer, opt_input


def optimize_qcengine(opt_input, computer_type="qc"):
    """Try to optimize, find TS, or find IRC of the system as specifed by a QCSchema
    OptimizationInput.

        Parameters
        ----------
        opt_input: Union[OptimizationInput, dict]
            Pydantic Schema of the OptimizationInput model.
            see https://github.com/MolSSI/QCElemental/blob/master/qcelemental/models/procedures.py
        computer_type: str

        Returns
        -------
        dict
    """

    if isinstance(opt_input, OptimizationInput):
        opt_input = json.loads(json_dumps(opt_input))  # Remove numpy elements turn into dictionary
    opt_output = copy.deepcopy(opt_input)

    # Make basic optking molecular system
    oMolsys = molsys.Molsys.from_schema(opt_input["initial_molecule"])
    try:
        initialize_options(opt_input["keywords"])
        computer = make_computer(opt_input, computer_type)
        opt_output = optimize(oMolsys, computer)
    except (OptError, KeyError, ValueError, AttributeError) as error:
        opt_output = {
            "success": False,
            "error": {"error_type": error.err_type, "error_message": error.mesg},
        }
        logger.critical(f"Error placed in qcschema: {opt_output}")
    except Exception as error:
        logger.critical("An unknown error has occured and evaded all error checking")
        opt_output = {
            "success": False,
            "error": {"error_type": error, "error_message": str(error)},
        }
        logger.critical(f"Error placed in qcschema: {opt_output}")
    finally:
        opt_input.update(opt_output)
        opt_input.update({"provenance": optking._optking_provenance_stamp})
        opt_input["provenance"]["routine"] = "optimize_qcengine"
        return opt_input

    # QCEngine.procedures.optking.py takes 'output_data', unpacks and creates Optimization Schema
    # from qcel.models.procedures.py


def make_computer(opt_input: dict, computer_type):
    logger.debug("Creating a Compute Wrapper")
    program = op.Params.program

    # This gets updated so it shouldn't be a reference
    molecule = copy.deepcopy(opt_input["initial_molecule"])

    # Sorting by spec_schema_name isn't foolproof b/c opt_input might not be a
    #   constructed model at this point if it's not arriving through QCEngine.
    spec_schema_name = opt_input["input_specification"].get("schema_name", "qcschema_input")
    if spec_schema_name == "qcschema_manybodyspecification":
        model = "(proc_spec_in_options)"
        options = opt_input["input_specification"]
    else:
        qc_input = opt_input["input_specification"]
        options = qc_input["keywords"]
        model = qc_input["model"]

    if computer_type == "psi4":
        # Please note that program is not actually used here
        return Psi4Computer(molecule, model, options, program)
    elif computer_type == "qc":
        return QCEngineComputer(molecule, model, options, program)
    elif computer_type == "user":
        logger.info("Creating a UserComputer")
        return UserComputer(molecule, model, options, program)
    else:
        raise OptError("computer_type is unknown")


def initialize_options(opt_keys, silent=False):
    if not silent:
        logger.info(welcome())

    userOptions = caseInsensitiveDict.CaseInsensitiveDict(opt_keys)
    # Save copy of original user options. Commented out until it is used
    # origOptions = copy.deepcopy(userOptions)

    # Create full list of parameters from user options plus defaults.
    try:
        op.Params = op.OptParams(userOptions)
    except (KeyError, ValueError, AttributeError) as e:
        logger.error(str(e))
        raise OptError("unable to parse params from userOptions")

    # TODO we should make this just be a normal object
    #  we should return it to the optimize method
    if not silent:
        logger.info(str(op.Params))
