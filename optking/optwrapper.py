# wrapper for optking's optimize function for input by psi4API
# creates a moleuclar system from psi4s and generates optkings options from psi4's lsit of options
import os
import logging
import json

from qcelemental.util.serialization import json_dumps
from qcelemental.models import OptimizationInput, OptimizationResult

import optking

from . import molsys
from .optimize import optimize


def optimize_psi4(calc_name, psi4_options, program='psi4'):
    """ Wrapper for optimize.optimize() Looks for an active psi4 molecule and optimizes.
        Optkings options can be set normally according to psi4 input syntax. Psi4's options must
        be passed to optking as a dictionary see psiAPI for details.   

        Warning. Meant to be depracted. Users should just use psi4.optimize available.
        Optking will try to use Psi4 if no program is provided in options

        Parameters
        ----------
        calcName: str
            level of theory for optimization. eg MP2
        qm_options: dict
            all options for the desired quantum chemistry package.
        program: str
            program used for gradients, hessians...
            
        Returns
        -------
        opt_output: dict
            dictionary serialized MOLSSI OptimizationResult. 
            see https://github.com/MolSSI/QCElemental/blob/master/qcelemental/models/procedures.py

        Notes
        -----
        Must include basis in options dictionary. eg {'basis': 'sto-3g'} 

    """

    import psi4

    logger = logging.getLogger(__name__)
    mol = psi4.core.get_active_molecule()
    oMolsys = molsys.Molsys.from_psi4_molecule(mol)

    #Get optking options from psi4
    module_options = psi4.driver.p4util.prepare_options_for_modules()
    keywords = {'OPTKING': {}}
    optking_options = module_options['OPTKING']
    for opt, optval in optking_options.items():
        if optval['has_changed']:
            keywords['OPTKING'][opt] = optval['value']
    keywords['OPTKING'].update({'program': program})
    
    #Combine all options for optking and the QC package together
    psi4_options_lower = {k.lower(): v for k, v in psi4_options.items()}
    keywords.update({'QM': psi4_options_lower})
    keywords['QM'].update({'method': calc_name})
    logger.info(json.dumps(keywords, indent=2))
    opt_output = optimize(oMolsys, keywords)
    opt_input['provenance']['creator'] = 'optking'
    return opt_output

def optimize_qcengine(opt_input):
    """ Try to optimize, find TS, or find IRC of the system as specifed by a QCSchema OptimizationInput.
        
        Parameters
        ----------
        optimization_input: OptimizationInput, dict
            Pydantic Schema of the OptimizationInput model.
            see https://github.com/MolSSI/QCElemental/blob/master/qcelemental/models/procedures.py
    
        Returns
        -------
        dict
            
    """
    
    if isinstance(opt_input, OptimizationInput):
        opt_input = opt_input.dict()  # Could fail if numpy elements present. Shouldn't be. 
        # replace with qcelemental.util.serialization method json_dumps(). Then use json.loads() if fails
    
    # Make basic optking molecular system
    oMolsys = molsys.Molsys.from_JSON_molecule(opt_input['initial_molecule'])  
    keywords = {'OPTKING': opt_input['keywords'], 'QM': {}}  # store optking keywords leave QM blank
    qc_input = opt_input['input_specification']  # will be used to create a QCSchema AtomicInput

    opt_output = optimize(oMolsys, keywords, qc_input)
    opt_input.update(opt_output)
    opt_input['provenance']['creator'] = "optking"
    # QCEngine.procedures.optking.py takes 'output_data', unpacks and creates Optimization Schema
    # from qcel.models.procedures.py
    return opt_input

