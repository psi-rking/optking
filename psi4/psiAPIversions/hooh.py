import psi4
from psi4 import *
from psi4.core import *
from psi4.driver.diatomic import anharmonicity
from psi4.driver.gaussian_n import *
from psi4.driver.aliases import *
from psi4.driver.driver_cbs import *
from psi4.driver.wrapper_database import database, db, DB_RGT, DB_RXN
from psi4.driver.wrapper_autofrag import auto_fragments
from psi4.driver.constants.physconst import *
psi4_io = core.IOManager.shared_object()
psioh = psi4.core.IOManager.shared_object()

psioh.set_default_path('/tmp2')
psi4_io.set_default_path("/tmp2")

geometry("""
0 1
H
H 1 0.74
""","blank_molecule_psi4_yo")
import numpy as np
core.efp_init()

hooh = geometry("""
    H            1.699924772228     1.549001852664     0.729368159665
    O           -0.027495833355     1.120334367050     0.682522182417
    O           -0.047683750414    -1.071778830756    -0.755485307218
    H           -0.506770221333    -2.319613449532     0.428609578964
 unit au
""","hooh")

core.IO.set_default_namespace("hooh")
core.set_global_option("BASIS", "dz")
core.set_local_option("OPTKING", "GEOM_MAXITER", 1)
core.set_local_option("OPTKING", "STEP_TYPE", "rfo")
core.set_local_option("OPTKING", "HESS_UPDATE", "bfgs")
core.set_local_option("OPTKING", "INTRAFRAG_HESS", "SIMPLE")

mol = core.get_active_molecule()
import optking
Molsys = optking.molsys.MOLSYS.fromPsi4Molecule(mol)
all_options = p4util.prepare_options_for_modules()
optking_options = all_options['OPTKING']
optking_user_options = {}
for opt,optval in optking_options.items():
    if optval['has_changed'] == True:
        optking_user_options[opt] = optval['value']

def setGeometry_func( newGeom ):
    psi_geom = core.Matrix.from_array( newGeom )
    mol.set_geometry( psi_geom )
    mol.update_geometry()
    newGeom[:] = np.array( mol.geometry() )

def gradient_func(printResults=True):
    psi4gradientMatrix, wfn = driver.gradient('hf', molecule=mol, return_wfn=True)
    gradientMatrix = np.array( psi4gradientMatrix )
    E = wfn.energy()
    if printResults:
        print '\tEnergy: %15.10f' % E
        print '\tGradient'
        print gradientMatrix
    return E, np.reshape(gradientMatrix, (gradientMatrix.size))

def hessian_func(printResults=True):
    H = driver.hessian('hf', molecule=mol)
    if printResults:
        print '\t Hessian'
        print H
    return H

def energy_func(printResults=True):
    E, wfn = driver.energy('hf', molecule=mol, return_wfn=True)
    if printResults:
        print '\tEnergy: %15.10f' % E
    return E

optking.optimize( Molsys, optking_user_options, setGeometry_func, gradient_func, \
                hessian_func, energy_func )

