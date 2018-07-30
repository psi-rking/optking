#! SCF STO-3G geometry optimzation, with Z-matrix input
import psi4
import optking
# These values are from a tightly converged QChem run

def test_opt1_h2o():
    refnucenergy = 8.9064890670                                                                     #TEST
    refenergy = -74.965901192                                                                    #TEST
    
    h2o = psi4.geometry("""
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """)


    psi4options = {
      'diis': False,
      'basis': 'sto-3g',
      'e_convergence': 10,
      'd_convergence': 10,
      'scf_type': 'pk'
    }    

    psi4.set_options(psi4options)
     
    thisenergy, nucenergy = optking.Psi4Opt('hf', psi4options)
    
    assert psi4.compare_values(refnucenergy, nucenergy, 3, "Nuclear repulsion energy")    #TEST
    assert psi4.compare_values(refenergy, thisenergy, 6, "Reference energy")                                #TEST
