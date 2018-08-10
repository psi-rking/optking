#! Linesearch tests
#memory 8gb

nucenergy = 41.670589 #Eh
refenergy = -1053.880393 #Eh

import Psi4Opt
import psi4

#molecule methaneDimer {
#H       -0.000000     -0.400064     -0.773367
#C       -0.000000     -0.400064      0.322633
#H        1.033319     -0.400064      0.687966
#H       -0.516660     -1.294945      0.687966
#H       -0.516660      0.494816      0.687966
#H       -0.000000     -7.403491     -1.251485
#C       -0.000000     -7.403491     -0.155485
#H        1.033319     -7.403491      0.209849
#H       -0.516660     -8.298372      0.209848
#H       -0.516660     -6.508611      0.209848
#}

Ar2 = psi4.geometry("""
  Ar
  Ar 1 5.0
""")

psi4.set_options({
  'basis': 'cc-pvdz',
  'd_convergence': 10,
  'geom_maxiter': 20,
  'g_convergence': 'gau_tight'
})


psi4.set_module_options('OPTKING', {'step_type': 'linesearch'})

#Psi4Opt.calcName = 'b3lyp-d'
Psi4Opt.calcName = 'mp2'
thisenergy = Psi4Opt.Psi4Opt()


psi4.compare_values(nucenergy, Ar2.nuclear_repulsion_energy(), 3, "Nuclear repulsion energy")  #TEST
psi4.compare_values(refenergy, thisenergy, 1, "Reference energy")  #TEST

