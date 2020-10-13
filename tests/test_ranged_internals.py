#! Test of 'ranged' coordinates.  Intcos that cannot move
#  out of a prescribed range.
import pytest
import psi4
import optking

conv_RHF_OO_at_135 = -150.7853238
init_OO_distance = ['1.25', '1.30', '1.325', '1.35', '1.40']

@pytest.mark.parametrize("option", init_OO_distance)
def test_ranged_stretch(option):
    geom_input_string = """
      H
      O 1 0.90
      O 2 """ + option + """ 1 100.0 
      H 3 0.90 2 100.0 1 115.0 """

    hooh = psi4.geometry(geom_input_string)

    psi4.core.clean_options()
    psi4options = {
      'basis': 'cc-PVDZ',
      'g_convergence': 'gau_tight',
      'geom_maxiter': 20
    }
    psi4.set_options(psi4options)

    xtra = {'ranged_distance' : "2 3 1.30 1.35"}
    json_output = optking.optimize_psi4('hf', **xtra)

    thisenergy = json_output['energies'][-1]
    assert psi4.compare_values(conv_RHF_OO_at_135, thisenergy, 6)

