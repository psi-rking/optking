#! Linesearch tests
#memory 8gb

refnucenergy = 41.670589 #Eh
refenergy = -1053.880393 #Eh

import psi4optwrapper
import psi4

def test_linesearch():
    Ar2 = psi4.geometry("""
      Ar
      Ar 1 5.0
    """)

    psi4options = {
      'basis': 'cc-pvdz',
      'd_convergence': 10,
      'geom_maxiter': 20,
      'g_convergence': 'gau_tight'
    }

    psi4.set_options(psi4options)
    
    psi4.set_module_options('OPTKING', {'step_type': 'linesearch'})
    thisenergy, nucenergy = psi4optwrapper.Psi4Opt('mp2', psi4options)
    
    assert psi4.compare_values(nucenergy, nucenergy, 3, "Nuclear repulsion energy")  #TEST
    assert psi4.compare_values(refenergy, thisenergy, 1, "Reference energy")  #TEST
    
