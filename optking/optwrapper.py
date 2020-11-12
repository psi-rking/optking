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
from .compute_wrappers import ComputeWrapper, QCEngineComputer, Psi4Computer, UserComputer
from .printTools import welcome
from .exceptions import OptError


def optimize_psi4(calc_name, program='psi4', dertype=None, **xtra_opt_params):
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

    logger = logging.getLogger(__name__)

    opt_input = {}
    opt_output = {}

    try:
        op.Params, oMolsys, computer, opt_input = initialize_from_psi4(calc_name, program,
                                                                       computer_type='psi4',
                                                                       dertype=dertype,
                                                                       **xtra_opt_params)
        opt_output = copy.deepcopy(opt_input)
        logger.info("psi4 has been initialized")
        opt_output = optimize(oMolsys, computer)
        print("optimize has finished")
    except (OptError, KeyError, ValueError, AttributeError) as error:
        opt_output = {"success": False, "error": {"error_type": error.err_type,
                                                  "error_message": error.mesg}}
        logger.critical(f"Error placed in qcschema: {opt_output}")
        logger.debug(str(opt_output))
    except Exception as error:
        logger.critical("An unknown error has occured and evaded error checking")
        opt_output = {"success": False, "error": {"error_type": error,
                                                  "error_message": str(error)}}
        logger.critical(f"Error placed in qcschema: {opt_output}")
    finally:
        opt_input.update({"provenance": optking._optking_provenance_stamp})
        opt_input["provenance"]["routine"] = "optimize_psi4"
        opt_input.update(opt_output)
        return opt_input


def initialize_from_psi4(calc_name, program, computer_type, dertype=None, **xtra_opt_params):
    """ Gathers information from an active psi4 instance. to cleanly run optking from a
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
    oMolsys: molsys.Molsys
    computer: ComputeWrapper
    opt_input: qcelemental.models.OptimizationInput

    """
    import psi4
    logger = logging.getLogger(__name__)
    mol = psi4.core.get_active_molecule()
    oMolsys = molsys.Molsys.from_psi4_molecule(mol)
    logger.debug('Optking molsys from psi4.core.get_active_molecule():')
    logger.debug( oMolsys )

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

    if xtra_opt_params:
        for xtra_key, xtra_value in xtra_opt_params.items():
            opt_keys[xtra_key.lower()] = xtra_value

    for option in all_options:
        if psi4.core.has_global_option_changed(option):
            if option not in opt_keys:
                qc_keys[option.lower()] = psi4.core.get_global_option(option)

    # Make a qcSchema OptimizationInput
    opt_input = {"keywords": opt_keys,
                 "initial_molecule": oMolsys.molsys_to_qc_molecule(),
                 "input_specification": {
                     "model": {
                         'basis': qc_keys.pop('basis'),
                         'method': calc_name},
                     "driver": "gradient",
                     "keywords": qc_keys}}
    logger.debug("Creating OptimizationInput, Initial qc molecule from Molsys:")
    logger.debug(opt_input["initial_molecule"])

    opt_input = OptimizationInput(**opt_input)
    # Remove numpy elements to allow at will json serialization
    opt_input = json.loads(json_dumps(opt_input))

    initialize_options(opt_keys)
    params = op.Params
    computer = make_computer(opt_input, computer_type)
    return params, oMolsys, computer, opt_input


def optimize_qcengine(opt_input, computer_type='qc'):
    """ Try to optimize, find TS, or find IRC of the system as specifed by a QCSchema
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
    logger = logging.getLogger(__name__)

    if isinstance(opt_input, OptimizationInput):
        opt_input = json.loads(json_dumps(opt_input))  # Remove numpy elements turn into dictionary
    opt_output = copy.deepcopy(opt_input)

    # Make basic optking molecular system
    oMolsys = molsys.Molsys.from_json_molecule(opt_input['initial_molecule'])
    try:
        initialize_options(opt_input['keywords'])
        computer = make_computer(opt_input, computer_type)
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


def make_computer(opt_input: dict, computer_type):
    logger = logging.getLogger(__name__)
    logger.info("Creating a Compute Wrapper")
    program = op.Params.program

    # This gets updated so it shouldn't be a reference
    molecule = copy.deepcopy(opt_input['initial_molecule'])
    qc_input = opt_input['input_specification']
    options = qc_input['keywords']
    model = qc_input['model']

    if computer_type == 'psi4':
        # Please note that program is not actually used here
        return Psi4Computer(molecule, model, options, program)
    elif computer_type == 'qc':
        return QCEngineComputer(molecule, model, options, program)
    elif computer_type == 'user':
        logger.info("Creating a UserComputer")
        return UserComputer(molecule, model, options, program)
    else:
        raise OptError("computer_type is unknown")


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
        raise OptError("unable to parse params from userOptions")

    # TODO we should make this just be a normal object
    #  we should return it to the optimize method
    # logger.debug(str(op.Params))
    logger.info(str(op.Params))

