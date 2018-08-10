#! SCF DZ allene geometry optimization, with Cartesian input, first in c2v symmetry,
#! then in Cs symmetry from a starting point with a non-linear central bond angle.

import psi4
import optking
#import importlib

def test_opt2_allene():
    refnucenergy = 59.2532646680161                                                                 #TEST
    refenergy = -115.8302823663                                                                    #TEST
    
    # starting point is D2d/c2v
    allene = psi4.geometry("""
     H  0.0  -0.92   -1.8
     H  0.0   0.92   -1.8
     C  0.0   0.00   -1.3
     C  0.0   0.00    0.0
     C  0.0   0.00    1.3
     H  0.92  0.00    1.8
     H -0.92  0.00    1.8
    """)
   
    psi4options = {
      'basis': 'DZ',
      'e_convergence': 10,
      'd_convergence': 10,
      'scf_type': 'pk',
    }

    psi4.set_options(psi4options)

    json_output = optking.Psi4Opt('hf', psi4options)
    thisenergy = json_output['properties']['return_energy']
    nucenergy = json_output['properties']['nuclear_repulsion_energy']
    assert psi4.compare_values(refnucenergy, nucenergy, 2, "Nuclear repulsion energy")    #TEST
    assert psi4.compare_values(refenergy, thisenergy, 6, "Reference energy")                                   #TEST
    
    # central C-C-C bond angle starts around 170 degrees to test the dynamic addition
    # of new linear bending coordinates, and the redefinition of dihedrals.
    allene = psi4.geometry("""
     H  0.0  -0.92   -1.8
     H  0.0   0.92   -1.8
     C  0.0   0.00   -1.3
     C  0.0   0.10    0.0
     C  0.0   0.00    1.3
     H  0.92  0.00    1.8
     H -0.92  0.00    1.8
    """)
    
    
    json_output = optking.Psi4Opt('hf', psi4options)
    thisenergy = json_output['properties']['return_energy']
    nucenergy = json_output['properties']['nuclear_repulsion_energy']
    assert psi4.compare_values(refnucenergy, nucenergy, 2, "Nuclear repulsion energy")    #TEST
    assert psi4.compare_values(refenergy, thisenergy, 6, "Reference energy")                                   #TEST
