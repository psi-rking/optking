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

    psi4options = {
      'basis': 'cc-pvdz',
      'g_convergence': 'gau_tight',
      'scf_type': 'pk'
    }

    psi4.set_options(psi4options)

    psi4.set_module_options('OPTKING', {'TEST_B': True, 'TEST_DERIVATIVE_B': True, "G_CONVERGENCE": "gau_tight"})

    json_output = optking.Psi4Opt('hf', psi4options)
    thisenergy = json_output['properties']['return_energy']
    nucenergy = json_output['properties']['nuclear_repulsion_energy']

    assert psi4.compare_values(refnucenergy, nucenergy, 4, "Nuclear repulsion energy") #TEST
    assert psi4.compare_values(refenergy, thisenergy, 8, "Reference energy")           #TEST


def test_maxiter():

    h2o = psi4.geometry("""
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """)

    psi4options = {
        'diis': 0,
        'basis': 'STO-3G',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10,
        'scf_type': 'PK',
      'geom_maxiter': 2,
    }
    psi4.set_options(psi4options)

    json_output = optking.Psi4Opt('hf', psi4options)

    assert "Maximum number of steps exceeded" in json_output['error']
