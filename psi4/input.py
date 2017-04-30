import numpy as np

molecule hooh {
  H
  O 1 0.90
  O 2 1.40 1 100.0
  H 3 0.90 2 100.0 1 115.0
--
  H 0.0 1.0  1.0
  O 0.0 1.0  0.0
  H 0.0 1.0 -1.0
}

set {
  basis dz
  puream false
  df_scf_guess false
  scf_type pk
  guess sad
  d_convergence 10
}

set optking {
  Geom_maxiter  1
  step_type     rfo
  hess_update   bfgs
  consecutive_backsteps 2
  intrafrag_hess SIMPLE
#  full_hess_every 1
}

mol = core.get_active_molecule()
mol.update_geometry()
mol.print_out()

print 'PSI4 molecule with %d fragments' % mol.nfragments()

# Create initial molecular structure object in optking
#  (atomic numbers, cartesians, masses, ...) using class method
#  in pyopt.frag.py for a PSI4 molecule.
import pyopt
pyopt_molsys = pyopt.molsys.MOLSYS.fromPsi4Molecule(mol)

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
    print H
    return H
#    molname = ref_wfn.molecule().name()
#    prefix = core.get_writer_file_prefix(molname)
#    with open(prefix+".hess", 'w') as fp:
#        fp.read("%5d%5d\n" % (natoms, junk))
#        Hin = np.array(natoms,3)
#        for row in fp:
#            fp.read("%20.10f%20.10f%20.10f\n", Hin[row,0], Hin[row,1], Hin[row,2])
#        np.reshape(Hin, (3*natoms,3*natoms))
#    print 'hessian'
#    print Hin

#  This function only matters for special purposes like 1D line searching.
# The energy is usually obtained from the gradient function.
def energy_func(printResults=True):
    E, wfn = driver.energy('hf', molecule=mol, return_wfn=True)
    if printResults:
        print '\tEnergy: %15.10f' % E
    return E

# Provide optimizer function the following arguments:
# 1. pyopt.molsys.MOLSYS system (defined by class method above.
# 2. a dictionary of optking options
# 3. a method which prepares a given Cartesian, numpy geometry
#     for subsequent gradient computation or analysis.
# 4. a method to compute the gradient, which returns the energy and gradient.
# 5. optional hessian function
# 6. optional energy function for line-searching algorithms

pyopt.optimize( pyopt_molsys, optking_user_options, setGeometry_func, gradient_func, \
                hessian_func, energy_func )

