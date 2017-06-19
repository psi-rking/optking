import psi4
from psi4 import *
from psi4.core import *
import numpy as np

# Create initial molecular structure object in optking (atomic numbers,
# cartesian coordinates, masses, ...) using class method for PSI4 molecule.
mol = core.get_active_molecule()
calcName = 0
import optking
Molsys = optking.molsys.MOLSYS.fromPsi4Molecule(mol)

# Collect the user-specified OPTKING keywords in a dict.
all_options = p4util.prepare_options_for_modules()
optking_options = all_options['OPTKING']
optking_user_options = {}
for opt,optval in optking_options.items():
    if optval['has_changed'] == True:
        optking_user_options[opt] = optval['value']
    
# Define a function for optking that takes a Cartesian, numpy geometry, and
#  puts it in place for subsequent gradient computations.  This function
#  may move COM or reorient; changes argument to match such geometry.
def setGeometry_func( newGeom ):
    psi_geom = core.Matrix.from_array( newGeom )
    mol.set_geometry( psi_geom )
    mol.update_geometry()
    newGeom[:] = np.array( mol.geometry() )

# Define a function for optking that returns an energy and cartesian
#  gradient in numpy array format.
def gradient_func(printResults=True):
    psi4gradientMatrix, wfn = driver.gradient(calcName, molecule=mol, return_wfn=True)
    gradientMatrix = np.array( psi4gradientMatrix )
    E = wfn.energy()
    if printResults:
        print '\tEnergy: %15.10f' % E
        print '\tGradient'
        print gradientMatrix
    return E, np.reshape(gradientMatrix, (gradientMatrix.size))

def hessian_func(printResults=True):
    H = driver.hessian(calcName, molecule=mol)
    if printResults:
        print '\t Hessian'
        print H
    return H

#  This function only matters for special purposes like 1D line searching.
# The energy is usually obtained from the gradient function.
def energy_func(printResults=True):
    E, wfn = driver.energy(calcName, molecule=mol, return_wfn=True)
    if printResults:
        print '\tEnergy: %15.10f' % E
    return E

def Psi4Opt():
    thisenergy = optking.optimize( Molsys, optking_user_options, setGeometry_func, gradient_func, \
                    hessian_func, energy_func )
    return thisenergy


