#wrapper for optking's optimize function for input by psi4API
#creates a moleuclar system from psi4s and generates optkings options from psi4's lsit of options

import psi4
import optking
import numpy as np 
import json

optking.printInit(psi4.core.print_out)
import optking.molsys
import optking.psi4methods
    
#Please note this is a hack. we do not want to read in all of psi4's options
#Right now I am reading in all global and optking options and appending the calcname to that
#dictionary just as a way to get the infromation into optking for generating json files
def Psi4Opt(calcName, psi4_options):
    """Method call for optimizing a molecule. Gives pyOptking a molsys from
    psi4s molecule class, a set containing all optking keywords, a function to
    set the geometry in psi4, and functions to get the gradient, hessian, and 
    energy from psi4. Returns energy or (energy, trajectory) if trajectory== True.
    """
    
    mol = psi4.core.get_active_molecule()
    oMolsys = optking.molsys.Molsys.fromPsi4Molecule(mol)

    all_options = psi4.driver.p4util.prepare_options_for_modules()
    optking_user_options = optking.psi4methods.get_optking_options_psi4(all_options) 
    
    optking_user_options['PSI4'] = psi4_options
    optking_user_options['PSI4']['calcName'] = calcName
    
    returnVal, nucenergy = optking.optimize(oMolsys, optking_user_options)
    return returnVal, nucenergy
