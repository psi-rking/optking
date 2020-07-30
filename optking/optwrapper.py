# wrapper for optking's optimize function for input by psi4API
# creates a moleuclar system from psi4s and generates optkings options from psi4's lsit of options
import logging
import json
import copy

from qcelemental.util.serialization import json_dumps
from qcelemental.models import OptimizationInput, OptimizationResult

import optking
from . import caseInsensitiveDict
from . import molsys
from . import optparams as op
from .optimize import optimize
from .compute_wrappers import QCEngineComputer, Psi4Computer
from .printTools import welcome
from .exceptions import OptError


def optimize_psi4(calc_name, program='psi4', dertype=None, XtraOptParams=None):
    """
    Wrapper for optimize.optimize() Looks for an active psi4 molecule and optimizes.
    This is the written warning that Optking will try to use psi4 if no program is provided
    Parameters
    ----------
    calc_name: str
        level of theory for optimization. eg MP2
    program: str
        program used for gradients, hessians...

    Returns
    -------
    opt_output: dict
        dictionary serialized MOLSSI OptimizationResult.
        see https://github.com/MolSSI/QCElemental/blob/master/qcelemental/models/procedures.py
    """

    import psi4

    logger = logging.getLogger(__name__)
    mol = psi4.core.get_active_molecule()
    oMolsys = molsys.Molsys.from_psi4_molecule(mol)

    # Get optking options and globals from psi4
    # Look through optking module specific options first. If a global has already appeared
    # in optking's options, don't include as a qc package option

    logger.debug("Getting module and psi4 options for qcschema construction")
    module_options = psi4.driver.p4util.prepare_options_for_modules()
    all_options = psi4.core.get_global_option_list()
    opt_keys = {'program': program}
    qc_keys = {}
    if dertype is not None:
        qc_keys['dertype'] = 0

    optking_options = module_options['OPTKING']
    for opt, optval in optking_options.items():
        if optval['has_changed']:
            opt_keys[opt.lower()] = optval['value']

    for option in all_options:
        if psi4.core.has_global_option_changed(option):
            if option in opt_keys:
                pass
            else:
                qc_keys[option.lower()] = psi4.core.get_global_option(option)

    if XtraOptParams is not None:
        for XtraKey, XtraValue in XtraOptParams.items():
            opt_keys[XtraKey.lower()] = XtraValue

    # Make a qcSchema OptimizationInput
    opt_input = {"keywords": opt_keys,
                 "initial_molecule": oMolsys.molsys_to_qc_molecule(),
                 "input_specification": {
                     "model": {
                         'basis': qc_keys.pop('basis'),
                         'method': calc_name},
                     "driver": "gradient",
                     "keywords": qc_keys}}

    logger.debug("Creating OptimizationInput")
    opt_input = OptimizationInput(**opt_input)

    # Remove numpy elements to allow at will json serialization
    opt_input = json.loads(json_dumps(opt_input))
    opt_output = copy.deepcopy(opt_input)

    try:
        initialize_options(opt_keys)
        computer = make_computer(opt_input)
        opt_output = optimize(oMolsys, computer)
    except (OptError, KeyError, ValueError, AttributeError) as error:
        opt_output = {"success": False, "error": {"error_type": error.err_type,
                                                  "error_message": error.mesg}}
        logger.critical(f"Error placed in qcschema: {opt_output}")
    except Exception as error:
        logger.critical("An unknown error has occured and evaded all error checking")

        opt_output = {"success": False, "error": {"error_type": error,
                                                  "error_message": str(error)}}
        logger.critical(f"Error placed in qcschema: {opt_output}")
    finally:
        opt_input.update({"provenance": optking._optking_provenance_stamp})
        opt_input["provenance"]["routine"] = "optimize_psi4"
        opt_input.update(opt_output)
        return opt_input


def optimize_qcengine(opt_input):
    """ Try to optimize, find TS, or find IRC of the system as specifed by a QCSchema
    OptimizationInput.

        Parameters
        ----------
        opt_input: Union[OptimizationInput, dict]
            Pydantic Schema of the OptimizationInput model.
            see https://github.com/MolSSI/QCElemental/blob/master/qcelemental/models/procedures.py
        Returns
        -------
        dict
    """
    logger = logging.getLogger(__name__)

    if isinstance(opt_input, OptimizationInput):
        opt_input = json.loads(json_dumps(opt_input))  # Remove numpy elements turn into dictionary
    opt_output = copy.deepcopy(opt_input)  # If we can't even make it into optimize

    # Make basic optking molecular system
    oMolsys = molsys.Molsys.from_json_molecule(opt_input['initial_molecule'])
    try:
        initialize_options(opt_input['keywords'])
        computer = make_computer(opt_input)
        opt_output = optimize(oMolsys, computer)
    except (OptError, KeyError, ValueError, AttributeError) as error:
        opt_output = {"success": False, "error": {"error_type": error.err_type,
                                                  "error_message": error.mesg}}
        logger.critical(f"Error placed in qcschema: {opt_output}")
    except Exception as error:
        logger.critical("An unknown error has occured and evaded all error checking")
        opt_output = {"success": False, "error": {"error_type": error,
                                                  "error_message": str(error)}}
        logger.critical(f"Error placed in qcschema: {opt_output}")
    finally:
        opt_input.update(opt_output)
        opt_input.update({'provenance': optking._optking_provenance_stamp})
        opt_input['provenance']['routine'] = "optimize_qcengine"
        return opt_input

    # QCEngine.procedures.optking.py takes 'output_data', unpacks and creates Optimization Schema
    # from qcel.models.procedures.py


def make_computer(opt_input: dict, computer_type='qc'):
    logger = logging.getLogger(__name__)
    logger.debug("Creating a Compute Wrapper")
    program = op.Params.program

    # This gets updated so it shouldn't be a reference
    molecule = copy.deepcopy(opt_input['initial_molecule'])
    qc_input = opt_input['input_specification']
    options = qc_input['keywords']
    model = qc_input['model']

    if computer_type == 'psi4':
        # Please note that program is not actually used here
        return Psi4Computer(molecule, model, options, program)
    else:
        return QCEngineComputer(molecule, model, options, program)


def initialize_options(opt_keys):
    logger = logging.getLogger(__name__)
    logger.info(welcome())
    userOptions = caseInsensitiveDict.CaseInsensitiveDict(opt_keys)
    # Save copy of original user options. Commented out until it is used
    # origOptions = copy.deepcopy(userOptions)

    # Create full list of parameters from user options plus defaults.
    try:
        op.Params = op.OptParams(userOptions)
    except (KeyError, ValueError, AttributeError) as e:
        logger.debug(str(e))
        raise

    # TODO we should make this just be a normal object
    #  we should return it to the optimize method
    #logger.debug(str(op.Params))
    logger.info(str(op.Params))

