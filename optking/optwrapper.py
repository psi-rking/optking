# wrapper for optking's optimize function for input by psi4API
# creates a moleuclar system from psi4s and generates optkings options from psi4's lsit of options
import copy
import json
import logging
import pprint

from qcelemental.models import OptimizationInput, OptimizationResult
from qcelemental.util.serialization import json_dumps
from pydantic import ValidationError
from pydantic.v1.error_wrappers import ValidationError as v1ValidationError

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
        opt_output = optimize(oMolsys, computer)
    except (ValidationError, v1ValidationError) as error:
        logger.critical("A ValidationError has occured: %s", error, exc_info=True)
        opt_output = {
            "success": False,
            "error": {"error_type": "ValidationError", "error_message": str(error)}
        }
    except TypeError as error:
        logger.critical("A TypeError has occured: %s", error, exc_info=True)
        logger.critical("This TypeError is likely related to option validation.")
        opt_output = {
            "success": False,
            "error": {"error_type": "ValidationError", "error_message": str(error)}
        }
    except Exception as error:
        logger.critical(
            "A critical exception has occured:\n%s - %s",
            type(error),
            str(error),
            exc_info=True
        )
        opt_output = {
            "success": False,
            "error": {"error_type": type(error), "error_message": str(error)},
        }
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

    # Get Molecule and Options from Psi4 as QCSchema
    logger.debug("Setting up optimization from Psi4's current state: p4util.state_to_atomic_input")
    atomic_input = psi4.driver.p4util.state_to_atomicinput(
        driver="gradient",
        method=calc_name,
    ).dict()

    # QCEngine doesn't expect information to already have been validated
    atomic_input.pop("id")
    atomic_input.pop("provenance")
    atomic_input.pop("protocols")
    o_molsys = molsys.Molsys.from_schema(atomic_input.get("molecule"))

    optking_canon = op.OptParams().model_dump(by_alias=True).keys()
    opt_keys = {"program": program}

    # Psi4 options can get mixed in with optking's options in prepare_options_for_module anyway.
    # Atomic_input will contain all set options so only accept options that are present in
    # options dict. Assume everything else is for Psi4.
    for opt, optval in atomic_input.get("keywords").items():
        if opt.upper() in optking_canon:
            opt_keys[opt] = optval

    if xtra_opt_params:
        for xtra_key, xtra_value in xtra_opt_params.items():
            opt_keys[xtra_key] = xtra_value

    # Make a qcSchema OptimizationInput
    opt_input = {
        "keywords": opt_keys,
        "initial_molecule": atomic_input.pop("molecule"),
        "input_specification": atomic_input,
    }

    # in case user has selected a specific dertype
    if dertype:
        opt_input.get("input_specification").get("keywords").update("dertype", dertype)

    try:
        logger.debug("Creating OptimizationInput")
        opt_input = OptimizationInput(**opt_input)
    except (ValidationError, v1ValidationError) as error:
        logger.critical("A Validation Error has occured while initializing optking: %s", error)
        logger.critical("Could not create an OptimizationInput")
        logger.critical(
            """Note: `optimize_psi4` is not recommended for users. Consider calling `psi4.optimize`
            or see `tests/test_opthelper` for an example of using the OptHelper interface."""
    )

    # Remove numpy elements that can appear from psi4 to allow at will json serialization
    # Json cannot handle numpy types only python builtins
    opt_input = json.loads(json_dumps(opt_input))
    params = initialize_options(opt_keys)
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
    except Exception as error:
        logger.critical("A critical error has occured: %s - %s", type(error), error, exc_info=True)
        opt_output = {
            "success": False,
            "error": {"error_type": type(error), "error_message": str(error)},
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

    # Turn keys but not vals to upper. Optparams will handle lowercase values
    opt_keys = {key.upper(): val for key, val in opt_keys.items()}

    # Create full list of parameters from user options plus defaults.
    try:
        params = op.OptParams(**opt_keys)
    except (KeyError, ValueError, AttributeError, ValidationError) as e:
        raise OptError("unable to parse params from userOptions") from e

    # TODO we should make this just be a normal object
    #  we should return it to the optimize method
    if not silent:
        logger.info(params)
    op.Params = params

    return params
