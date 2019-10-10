# wrapper for optking's optimize function for input by psi4API
# creates a moleuclar system from psi4s and generates optkings options from psi4's lsit of options
import os

import optking

from . import molsys
from .optimize import optimize

def optimize_psi4(calcName, psi4_options):
    """Method call for optimizing a molecule. Gives pyOptking a molsys from
    psi4s molecule class, a set containing all optking keywords, a function to
    set the geometry in psi4, and functions to get the gradient, hessian, and
    energy from psi4. Returns energy or (energy, trajectory) if trajectory== True.
    """

    import psi4

    mol = psi4.core.get_active_molecule()
    oMolsys = molsys.Molsys.fromPsi4Molecule(mol)

    # def collect_psi4_options(options):
    """Is meant to look through the dictionary of psi4 options being passed in and
    pick out the basis set and QM method used (Calcname)
    which are appened to the list of psi4 options
    """
    keywords = {}
    for opt in options['PSI4']:
        keywords[opt] = options['PSI4'][opt]

    basis = keywords['BASIS']
    del keywords['BASIS']
    QM_method = keywords['CALCNAME']
    del keywords['CALCNAME']

    # return QM_method, basis, keywords


# def get_optking_options_psi4(all_options):
    optking_user_options = {}
    optking_options = all_options['OPTKING']
    optking_user_options['OPTKING'] = {}
    for opt, optval in optking_options.items():
        if optval['has_changed']:
            optking_user_options[opt] = optval['value']

    # return optking_user_options

    #old way currently used for testing suite. need to comment out when working
    #with psi4.optimize()
    all_options = psi4.driver.p4util.prepare_options_for_modules()
    optking_user_options = psi4methods.get_optking_options_psi4(all_options)
    psi4_options_upper = {k.upper(): v for k, v in psi4_options.items()}
    optking_user_options['PSI4'] = psi4_options_upper
    optking_user_options["PSI4"]['CALCNAME'] = calcName #change to psi4_options when using psi4.optimize()

    # all we should need to do for psi4.optimize
    #psi4_options["PSI4"]["CALCNAME"] = calcName
    #pass psi4_options directly optimize

    optking_json_dict = optimize(oMolsys, optking_user_options)

    return optking_json_dict

def optimize_qcengine(optimization_input):

    pass
