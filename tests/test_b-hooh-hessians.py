#! SCF CC-PVDZ geometry optimzation, with Z-matrix input

import psi4
import runpsi4API

finalEnergy = -150.786766850  #TEST

def test_nokeywordhess_every():
    hooh = psi4.geometry("""
      H
      O 1 0.9
      O 2 1.4 1 100.0
      H 3 0.9 2 100.0 1 170.0
    """)

    psi4options = {
      'basis': 'cc-pvdz',
      'g_convergence': 'gau_verytight',
      'scf_type': 'pk',
    }

    psi4.set_options(psi4options)
    
    psi4.set_module_options('Optking', {'print': 3, 'geom_maxiter': 100}) 
    
    E, nucenergy = runpsi4API.Psi4Opt('hf', psi4options)
    assert psi4.compare_values(finalEnergy, E, 8, "Final energy, empirical Hessian")                                #TEST

def test_hess_every0():
    hooh = psi4.geometry("""
      H
      O 1 0.9
      O 2 1.4 1 100.0
      H 3 0.9 2 100.0 1 170.0
    """)
   
    psi4options = {
      'basis': 'cc-pvdz',
      'g_convergence': 'gau_verytight',
      'scf_type': 'pk',
    }

    psi4.set_options(psi4options)
 
    psi4.set_module_options('Optking', {'print': 5, 'full_hess_every': 0, 'geom_maxiter': 200})
    
    E, nucenergy = runpsi4API.Psi4Opt('hf', psi4options)
    assert psi4.compare_values(finalEnergy, E, 8, "Final energy, initial Hessian")                                #TEST

def test_hess_every3():    
    hooh = psi4.geometry("""
      H
      O 1 0.9
      O 2 1.4 1 100.0
      H 3 0.9 2 100.0 1 170.0
    """)

    psi4options = {
      'basis': 'cc-pvdz',
      'g_convergence': 'gau_verytight',
      'scf_type': 'pk',
    }

    psi4.set_options(psi4options)
    
    psi4.set_module_options('Optking', {'full_hess_every': 3, 'geom_maxiter': 200})
    
    E, nucenergy = runpsi4API.Psi4Opt('hf', psi4options)
    assert psi4.compare_values(finalEnergy, E, 8, "Final energy, every 3rd step Hessian")                                #TEST

def test_hess_every1():    
    hooh = psi4.geometry("""
      H
      O 1 0.9
      O 2 1.4 1 100.0
      H 3 0.9 2 100.0 1 170.0
    """)
   
    psi4options = {
      'basis': 'cc-pvdz',
      'g_convergence': 'gau_verytight',
      'scf_type': 'pk',
    }

    psi4.set_options(psi4options)

    psi4.set_module_options('Optking', {'full_hess_every': 1})
    
    E, nucenergy = runpsi4API.Psi4Opt('hf', psi4options)
    psi4.compare_values(finalEnergy, E, 8, "Final energy, every step Hessian")                                #TEST

def test_hess_fischer():

    hooh = psi4.geometry("""
      H
      O 1 0.9
      O 2 1.4 1 100.0
      H 3 0.9 2 100.0 1 170.0
    """)

    psi4options = {
      'basis': 'cc-pvdz',
      'g_convergence': 'gau_verytight',
      'scf_type': 'pk',
    }

    psi4.set_options(psi4options)
    
    psi4.set_module_options('Optking', {'intrafrag_hess': 'fischer'})
    
    E, nucenergy = runpsi4API.Psi4Opt('hf', psi4options)
    psi4.compare_values(finalEnergy, E, 8, "Final energy, every step Hessian")                                #TEST

def test_hess_lindh_simple():
   
    hooh = psi4.geometry("""
      H
      O 1 0.9
      O 2 1.4 1 100.0
      H 3 0.9 2 100.0 1 170.0
    """)
   
    psi4options = {
      'basis': 'cc-pvdz',
      'g_convergence': 'gau_verytight',
      'scf_type': 'pk',
    }

    psi4.set_options(psi4options)

    psi4.set_module_options('Optking', {'intrafrag_hess': 'LindH_simple'})
    
    E, nucenergy = runpsi4API.Psi4Opt('hf', psi4options)
    assert psi4.compare_values(finalEnergy, E, 8, "Final energy, every step Hessian")                                #TEST



