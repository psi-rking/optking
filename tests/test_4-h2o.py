#! SCF CC-PVTZ geometry optimzation, with z-matrix input
import pytest
import psi4
import optking

finalEnergy = -76.05776970 #TEST

@pytest.mark.parametrize("option, expected", [('RFO', finalEnergy), ('NR', finalEnergy), ('SD', finalEnergy)])
def test_h2o_rfo(option, expected):
    h2o = psi4.geometry("""
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """)

    psi4.core.clean_options()    
    psi4_options = {
        'basis': 'cc-pvtz',
        'e_convergence': '10',
        'd_convergence': '10',
        'scf_type': 'pk',
        'step_type': option 
    }

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4('hf') # Uses default program (psi4)
    E = json_output['energies'][-1]
    
    assert psi4.compare_values(finalEnergy, E, 6, f"{option} Step Final Energy")                                #TEST

