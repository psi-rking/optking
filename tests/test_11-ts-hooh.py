#Tests the PRFO code finding the 180 degree tors angle transition state for hooh

import psi4
import runpsi4API

TORS_ENERGY      = -150.7854114 #TEST
ZERO_TORS_ENERGY = -150.786766850 #TEST

def test_hooh_TS():
    # Optimization to 180 degree torsion from 120

    hooh = psi4.geometry("""
     0 1
     H
     O 1 0.95
     O 2 1.40 1 105.0
     H 3 0.95 2 105.0 1 120.0
    """)

    psi4options = {
      'basis': 'cc-pvdz',
      'geom_maxiter': 20,
      'opt_type': 'TS',
      'scf_type': 'pk',
      'docc': [ 5 , 4 ],
      'intrafrag_step_limit': 0.1
    }    

    psi4.set_options(psi4options) 
    
    thisenergy, nucenergy = runpsi4API.Psi4Opt('hf', psi4options)
    assert psi4.compare_values(TORS_ENERGY, thisenergy, 6, "cc-pVDZ RHF transition-state opt. of HOOH (dihedral=180), energy") #TEST

def test_hooh_min():
    # Optimization to 0 degree torsion from 100
    hooh = psi4.geometry("""
     H
     O 1 0.95
     O 2 1.40 1 105.0
     H 3 0.95 2 105.0 1 100.0
    """)
    
    psi4options = {
      'basis': 'cc-pvdz',
      'geom_maxiter': 20,
      'opt_type': 'min',  
      'scf_type': 'pk',
      'docc': [ 5 , 4 ],
    }

    psi4.set_options(psi4options)
    
    thisenergy, nucenergy = runpsi4API.Psi4Opt('hf', psi4options)
    assert psi4.compare_values(ZERO_TORS_ENERGY, thisenergy, 6, "cc-pVDZ RHF transition-state opt. of HOOH (dihedral=0), energy") #TEST

