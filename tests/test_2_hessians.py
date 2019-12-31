import pytest
import psi4
import optking

final_energy = -150.786766850
hess_every = [(-1, final_energy), (0, final_energy), (1, final_energy), (3, final_energy)]
hess_guess = [('fischer', final_energy), ('lindH_simple', final_energy), ('simple', final_energy),
               ('lindh', final_energy)]
hess_update = [('MS', final_energy), ('powell', final_energy), ('bofill', final_energy)]


@pytest.mark.parametrize("every, expected", hess_every, ids=['None', 'First Step', 'Every', 'Every 3'])
def test_hess_every(every, expected):

    hooh = psi4.geometry("""
        H
        O 1 0.9
        O 2 1.4 1 100.0
        H 3 0.9 2 100.0 1 170.0
        """)
    
    psi4.core.clean_options()
    psi4_options = {
        'basis': 'cc-pvdz',
        'scf_type': 'pk',
        'g_convergence': 'gau_verytight',
        'full_hess_every': every
    }
    
    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4('hf')  # Uses default program (psi4)
    E = json_output['energies'][-1]
    
    assert psi4.compare_values(expected, E, 8, "Final energy, every step Hessian")  # TEST
    print(f"Number of steps taken {len(json_output['trajectory'])}")


@pytest.mark.parametrize("guess, expected", hess_guess)
def test_hess_guess(guess, expected):

    hooh = psi4.geometry("""
        H
        O 1 0.9
        O 2 1.4 1 100.0
        H 3 0.9 2 100.0 1 170.0
        """)

    psi4.core.clean_options()
    psi4_options = {
        'basis': 'cc-pvdz',
        'scf_type': 'pk',
        'g_convergence': 'gau_verytight',
        'intrafrag_hess': guess
    }

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4('hf') # Uses default program (psi4)
    E = json_output['energies'][-1]
    print(f"Number of steps taken {len(json_output['trajectory'])}")
    assert psi4.compare_values(expected, E, 8, "Final energy, every step Hessian")  # TEST


@pytest.mark.parametrize("update, expected", hess_update)
def test_hess_update(update, expected):

    hooh = psi4.geometry("""
        H
        O 1 0.9
        O 2 1.4 1 100.0
        H 3 0.9 2 100.0 1 170.0
        """)

    psi4.core.clean_options()
    psi4_options = {
        'basis': 'cc-pvdz',
        'scf_type': 'pk',
        'g_convergence': 'gau_verytight', 
        'hess_update': update
    }

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4('hf') # Uses default program (psi4)
    E = json_output['energies'][-1]
    
    print(f"Number of steps taken {len(json_output['trajectory'])}")
    assert psi4.compare_values(expected, E, 8, "Final energy, every step Hessian")  # TEST
