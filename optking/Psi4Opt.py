import psi4
import numpy as np
import optking


def setGeometry_func(newGeom):
    """Define a function for optking that takes a Cartesian, numpy geometry, and
    puts it in place for subsequent gradient computations.  This function
    may move COM or reorient; will return possible changed cartesian geometry.
    """

    mol = psi4.core.get_active_molecule()
    psi_geom =  psi4.core.Matrix.from_array( newGeom )
    mol.set_geometry( psi_geom )
    mol.update_geometry()
    return np.array( mol.geometry() )
#    newGeom[:] = np.array( mol.geometry() )

def gradient_func(xyz, printResults=True):
    """Define a function for optking that returns an energy and cartesian
    gradient in numpy array format."""    

    mol = psi4.core.get_active_molecule()
    xyz[:] = setGeometry_func(xyz)
    psi4gradientMatrix, wfn = psi4.driver.gradient(calcName, molecule=mol, return_wfn=True)
    gradientMatrix = np.array( psi4gradientMatrix )
    E = wfn.energy()
    if printResults:
       psi4.core.print_out( '\tEnergy: %15.10f\n' % E)
       psi4.core.print_out( '\tGradient\n')
       psi4.core.print_out( str(gradientMatrix) )
       psi4.core.print_out( "\n")
    return E, np.reshape(gradientMatrix, (gradientMatrix.size))

def hessian_func(xyz, printResults=False): 
    mol = psi4.core.get_active_molecule()
    xyz[:] = setGeometry_func(xyz)
    H = psi4.driver.hessian(calcName, molecule=mol)
    if printResults:
        psi4.core.print_out( 'Hessian\n')
        H.print_out()
        psi4.core.print_out( "\n")
    Hnp = np.array( H )
    return Hnp

#  This function only matters for special purposes like 1D line searching.
# The energy is usually obtained from the gradient function.
def energy_func(xyz, printResults=True):
    """This function only matters for special purposes like 1D line searching.
    The energy is usually obtained from the gradient function.
    """

    mol = psi4.core.get_active_molecule()
    xyz[:] = setGeometry_func(xyz)
    E, wfn = psi4.driver.energy(calcName, molecule=mol, return_wfn=True)
    if printResults:
        psi4.core.print_out('\tEnergy: %15.10f\n' % E)
    return E

def get_optking_keywords():
    """ searches psi4s options for any optking module options and returns optking
     keywords as a set
    """

    all_options = psi4.driver.p4util.prepare_options_for_modules()
    optking_options = all_options['OPTKING']
    optking_user_options = {}
    for opt,optval in optking_options.items():
        if optval['has_changed'] == True:
            optking_user_options[opt] = optval['value']

    return optking_user_options

def make_optking_molsys():
    """Create initial molecular structure object in optking (atomic numbers,
    cartesian coordinates, masses, ...) using class method for PSI4 molecule.
    """
    mol = psi4.core.get_active_molecule()
    import optking.molsys
    return optking.molsys.MOLSYS.fromPsi4Molecule(mol)

def Psi4Opt():
    """Method call for optimizing a molecule. Gives pyOptking a molsys from
    psi4s molecule class, a set containing all optking keywords, a function to
    set the geometry in psi4, and functions to get the gradient, hessian, and 
    energy from psi4. Returns energy or (energy, trajectory) if trajectory== True.
    """

    optking.printInit(psi4.core.print_out)
    
    returnVal = optking.optimize(
        make_optking_molsys(), get_optking_keywords(), setGeometry_func, gradient_func, \
        hessian_func, energy_func)
    return returnVal


