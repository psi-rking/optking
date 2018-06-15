#! SCF CC-PVTZ geometry optimzation, with Z-matrix input

finalEnergy = -76.05776970 #TEST
import psi4
import Psi4Opt

def test_h2o_rfo():
    h2o = psi4.geometry("""
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """)
    
    psi4.set_options({
      'basis': 'cc-pvtz',
      'e_convergence': '10',
      'd_convergence': '10',
      'scf_type': 'pk',  
    })
    
    psi4.set_module_options('Optking', {'step_type': 'rfo'})
    Psi4Opt.calcName = 'hf'
    E = Psi4Opt.Psi4Opt()
    assert psi4.compare_values(finalEnergy, E, 6, "RFO Step Final Energy")                                #TEST

def test_h2o_nr():
    h2o = psi4.geometry("""
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """)
    
    psi4.set_options({
      'basis': 'cc-pvtz',
      'e_convergence': '10',
      'd_convergence': '10',
      'scf_type': 'pk',  
    })
    
    psi4.set_module_options('Optking', {'step_type': 'nr'})
    
    Psi4Opt.calcName = 'hf'
    E = Psi4Opt.Psi4Opt()
    assert psi4.compare_values(finalEnergy, E, 6, "NR Step Final Energy")                                #TEST

def test_h2o_SD():    
    h2O = psi4.geometry("""
    O
    H 1 1.0
    H 1 1.0 2 104.5
    """)
    
    psi4.set_options({
      'basis': 'cc-pvtz',
      'e_convergence': '10',
      'd_convergence': '10',
      'scf_type': 'pk',
    })
    
    psi4.set_module_options('Optking', {'step_type': 'SD'})
    
    Psi4Opt.calcName = 'hf'
    E = Psi4Opt.Psi4Opt()
    assert psi4.compare_values(finalEnergy, E, 6, "SD Step Final Energy")                                #TEST


