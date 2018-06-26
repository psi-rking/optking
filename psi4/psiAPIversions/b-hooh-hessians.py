#! SCF CC-PVDZ geometry optimzation, with Z-matrix input

import psi4

finalEnergy = -150.786766850  #TEST

hooh = psi4.geometry("""
  H
  O 1 0.9
  O 2 1.4 1 100.0
  H 3 0.9 2 100.0 1 170.0
""")

psi4.set_options({
  'basis': 'cc-pvdz',
  'g_convergence': 'gau_verytight',
  'scf_type': 'pk',
})

#set optking print 3
psi4.set_module_options('Optking', {'print': 3, 'geom_maxiter': 100}) 

import Psi4Opt
Psi4Opt.calcName = 'hf'
E = Psi4Opt.Psi4Opt()
psi4.compare_values(finalEnergy, E, 8, "Final energy, empirical Hessian")                                #TEST

hooh = psi4.geometry("""
  H
  O 1 0.9
  O 2 1.4 1 100.0
  H 3 0.9 2 100.0 1 170.0
""")

psi4.set_options({
  'basis': 'cc-pvdz',
  'g_convergence': 'gau_verytight',
  'scf_type': 'pk',
})
#set optking full_hess_every 0
psi4.set_module_options('Optking', {'print': 5, 'full_hess_every': 0, 'geom_maxiter': 200})

#reload(Psi4Opt)
Psi4Opt.calcName = 'hf'
E = Psi4Opt.Psi4Opt()
psi4.compare_values(finalEnergy, E, 8, "Final energy, initial Hessian")                                #TEST

hooh = psi4.geometry("""
  H
  O 1 0.9
  O 2 1.4 1 100.0
  H 3 0.9 2 100.0 1 170.0
""")

#set optking full_hess_every 3

psi4.set_options({
  'basis': 'cc-pvdz',
  'g_convergence': 'gau_verytight',
  'scf_type': 'pk',
})
psi4.set_module_options('Optking', {'full_hess_every': 3, 'geom_maxiter': 200})

#reload(Psi4Opt)
Psi4Opt.calcName = 'hf'
E = Psi4Opt.Psi4Opt()
psi4.compare_values(finalEnergy, E, 8, "Final energy, every 3rd step Hessian")                                #TEST

hooh = psi4.geometry("""
  H
  O 1 0.9
  O 2 1.4 1 100.0
  H 3 0.9 2 100.0 1 170.0
""")

psi4.set_options({
  'basis': 'cc-pvdz',
  'g_convergence': 'gau_verytight',
  'scf_type': 'pk',
})

#set optking full_hess_every 1
psi4.set_module_options('Optking', {'full_hess_every': 1})

#reload(Psi4Opt)
Psi4Opt.calcName = 'hf'
E = Psi4Opt.Psi4Opt()
psi4.compare_values(finalEnergy, E, 8, "Final energy, every step Hessian")                                #TEST

