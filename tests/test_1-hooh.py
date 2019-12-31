# HF SCF CC-PVDZ geometry optimization of HOOH with Z-matrix input
import psi4
import optking

def test_B_dB_matrices():
    refnucenergy = 38.06177     #TEST
    refenergy = -150.786766850  #TEST

    hooh = psi4.geometry("""
      H
      O 1 0.9
      O 2 1.4 1 100.0
      H 3 0.9 2 100.0 1 114.0
    """)

    psi4.core.clean_options()

    psi4_options = {
        'basis': 'cc-pvdz',
        'g_convergence': 'gau_tight',
        'scf_type': 'pk',
        'TEST_B': True,
        'TEST_DERIVATIVE_B': True,
        "G_CONVERGENCE": "gau_tight"
    }

    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4('hf') # Uses default program (psi4)
    E = json_output['energies'][-1]
    nucenergy = json_output['trajectory'][-1]['properties']['nuclear_repulsion_energy']

    assert 'test_b' in json_output['keywords']
    assert 'test_derivative_b' in json_output['keywords']
    assert "g_convergence" in json_output['keywords']
    assert psi4.compare_values(refnucenergy, nucenergy, 4, "Nuclear repulsion energy") #TEST
    assert psi4.compare_values(refenergy, E , 8, "Reference energy")           #TEST


def test_maxiter():

    h2o = psi4.geometry("""
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """)

    psi4.core.clean_options()
    psi4options = {
        #Throw a bunch of options at psi4
        'diis': 0,
        'basis': 'STO-3G',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10,
        'scf_type': 'PK',
        'geom_maxiter': 2,
    }
    psi4.set_options(psi4options)

    json_output = optking.optimize_psi4('hf')

    assert 'geom_maxiter' in json_output['keywords']
    assert "Maximum number of steps exceeded" in json_output['error']['error_message']
    assert "OptError" in json_output['error']['error_type']
