import psi4
from psi4 import *
from psi4.core import *
import numpy as np

# Collect the user-specified OPTKING keywords in a dict.
all_options = p4util.prepare_options_for_modules()
optking_options = all_options['OPTKING']
optking_user_options = {}
for opt,optval in optking_options.items():
    if optval['has_changed'] == True:
        optking_user_options[opt] = optval['value']
    
# Create initial molecular structure object in optking (atomic numbers,
# cartesian coordinates, masses, ...) using class method for PSI4 molecule.
# Initialize printing.
import optking
optking.printInit(print_out)
#optking.printInit()  # for default

mol = core.get_active_molecule()
import optking.molsys
Molsys = optking.molsys.MOLSYS.fromPsi4Molecule(mol)

# Define a function for optking that takes a Cartesian, numpy geometry, and
#  puts it in place for subsequent gradient computations.  This function
#  may move COM or reorient; changes argument to match such geometry.
def setGeometry_func( newGeom ):
    psi_geom = core.Matrix.from_array( newGeom )
    mol.set_geometry( psi_geom )
    mol.update_geometry()
    newGeom[:] = np.array( mol.geometry() )

calcName = 0
# Define a function for optking that returns an energy and cartesian
#  gradient in numpy array format.
def gradient_func(printResults=True):
    psi4gradientMatrix, wfn = driver.gradient(calcName, molecule=mol, return_wfn=True)
    gradientMatrix = np.array( psi4gradientMatrix )
    E = wfn.energy()
    if printResults:
        print_out( '\tEnergy: %15.10f\n' % E)
        print_out( '\tGradient\n')
        print_out( str(gradientMatrix) )
        print_out( "\n")
    return E, np.reshape(gradientMatrix, (gradientMatrix.size))

def hessian_func(printResults=True):
    H = driver.hessian(calcName, molecule=mol)
    if printResults:
        print_out( '\t Hessian\n')
        print_out( str(H) )
        print_out( "\n")
    return H

#  This function only matters for special purposes like 1D line searching.
# The energy is usually obtained from the gradient function.
def energy_func(printResults=True):
    E, wfn = driver.energy(calcName, molecule=mol, return_wfn=True)
    if printResults:
        print_out('\tEnergy: %15.10f\n' % E)
    return E

# Also send python printing function. Otherwise print to stdout will be done

def Psi4Opt():
    thisenergy = optking.optimize( Molsys, optking_user_options, setGeometry_func, gradient_func, \
                    hessian_func, energy_func)
    return thisenergy


