#! SCF CC-PVTZ geometry optimzation, with Z-matrix input

finalEnergy = -76.05776970 #TEST
import psi4
import optking

def test_h2o_rfo():
    h2o = psi4.geometry("""
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """)
    
    psi4options = {
      'basis': 'cc-pvtz',
      'e_convergence': '10',
      'd_convergence': '10',
      'scf_type': 'pk',  
    }

    psi4.set_options(psi4options)
    
    psi4.set_module_options('Optking', {'step_type': 'rfo'})
    E, nucenergy = optking.Psi4Opt('hf', psi4options)
    assert psi4.compare_values(finalEnergy, E, 6, "RFO Step Final Energy")                                #TEST

def test_h2o_nr():
    h2o = psi4.geometry("""
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """)
   
 
    psi4options = {
      'basis': 'cc-pvtz',
      'e_convergence': '10',
      'd_convergence': '10',
      'scf_type': 'pk',  
    }

    psi4.set_options(psi4options)
    
    psi4.set_module_options('Optking', {'step_type': 'nr'})
    
    E, nuc = optking.Psi4Opt('hf', psi4options)
    assert psi4.compare_values(finalEnergy, E, 6, "NR Step Final Energy")                                #TEST

def test_h2o_SD():    
    h2O = psi4.geometry("""
    O
    H 1 1.0
    H 1 1.0 2 104.5
    """)
    
    psi4options = { 
      'basis': 'cc-pvtz',
      'e_convergence': '10',
      'd_convergence': '10',
      'scf_type': 'pk',  
    }

    psi4.set_options(psi4options)
    
    psi4.set_module_options('Optking', {'step_type': 'SD'})
    
    E, nucenergy = optking.Psi4Opt('hf', psi4options)
    assert psi4.compare_values(finalEnergy, E, 6, "SD Step Final Energy")                                #TEST


