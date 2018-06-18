#! Linesearch tests
#memory 8gb

nucenergy = 41.670589 #Eh
refenergy = -1053.880393 #Eh

import Psi4Opt
import psi4

def test_linesearch():
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
    
    assert psi4.compare_values(nucenergy, Ar2.nuclear_repulsion_energy(), 3, "Nuclear repulsion energy")  #TEST
    assert psi4.compare_values(refenergy, thisenergy, 1, "Reference energy")  #TEST
    
