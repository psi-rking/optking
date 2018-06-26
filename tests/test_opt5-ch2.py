#! 6-31G** UHF CH2 3B1 optimization.  Uses a Z-Matrix with dummy atoms, just for demo and testing purposes.

import psi4
import runpsi4API

nucrefenergy = 6.197322440574482     #TEST
refenergy = -38.925486977153      #TEST

def test_ch2_with_dummy_atoms():
    ch2 = psi4.geometry("""
    0 3
    c
    x 1 1.0
    h 1 b1 2 a1
    h 1 b1 2 a1 3 180.0
    
    b1 = 1.0
    a1 = 60.0
    """)
   

    psi4options = {
      'reference': 'uhf',
      'basis': '6-31G(d,p)',
      'docc': [2, 0, 0, 1],
      'socc': [1, 0, 1, 0],
      'scf_type': 'pk'
    }

 
    psi4.set_options(psi4options)
    
    thisenergy, nucenergy = runpsi4API.Psi4Opt('hf', psi4options)
    
    assert psi4.compare_values(nucrefenergy, nucenergy, 3, "Nuclear repulsion energy")  #TEST
    assert psi4.compare_values(refenergy, thisenergy, 6, "Reference energy")  #TEST

