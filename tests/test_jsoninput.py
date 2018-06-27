import runoptkingjson
import psi4
import os

refnucenergy = 8.9064890670
refenergy = -74.965901192 

def test_input_through_json():
    energy, nucenergy = runoptkingjson.run_optking_json(os.path.join(os.path.dirname(__file__), 'jsoninput.json'))

    assert psi4.compare_values(refnucenergy, nucenergy, 3, "Nuclear repulsion energy")
    assert psi4.compare_values(refenergy, energy, 6, "Reference energy") 
