import os
import json

import pytest
import optking
import psi4


def test_input_through_json_dict():
    refnucenergy = 8.9064890670
    refenergy = -74.965901192

    json_dict_in = {"schema_name": "qc_schema_input", "schema_version": 1, "molecule": { "geometry": [ 0.0, 0.0, 0.28100228, 0.0, 1.42674323, -0.8431958, 0.0, -1.42674323, -0.8431958 ], "symbols": [ "O", "H", "H" ], "masses": [ 15.994915, 1.007825, 1.007825 ] }, "driver": "optimize", "model": { "method": "hf", "basis": "sto-3g" }, "keywords": { "diis": False, "e_convergence": 10, "d_convergence": 10, "scf_type": "pk", "optimizer": { "output_type": "JSON" }}}

    json_dict = optking.run_json_dict(json_dict_in)

    assert psi4.compare_values(refnucenergy, json_dict['properties']['nuclear_repulsion_energy'], 3,
         "Nuclear repulsion energy")
    assert psi4.compare_values(refenergy, json_dict['properties']['return_energy'], 6,
        "Reference energy")


@pytest.mark.parametrize("inp,expected", [
    ('jsoninput.json', (8.9064890670, -74.965901192)),
    ('json_betapinene.json', (568.2219045869700267, -383.381055594356)),
    ('json_hooh_frozen.json', (37.969354880, -150.786372411)),
])
def test_input_through_json(inp, expected):
    with open(os.path.join(os.path.dirname(__file__), inp)) as input_data:
        input_copy = json.load(input_data)
    optking.run_json(os.path.join(os.path.dirname(__file__), inp))

    #For testing purposes. If this works, we have properly returned the output, and added the result
    #to the original file. In order to preserve the form of the test suite, we now resore the input
    #to its original state
    with open(os.path.join(os.path.dirname(__file__), inp)) as input_data:
        json_dict = json.load(input_data)
    assert psi4.compare_values(expected[0], json_dict['properties']['nuclear_repulsion_energy'], 3,
         "Nuclear repulsion energy")
    assert psi4.compare_values(expected[1], json_dict['properties']['return_energy'], 6,
        "Reference energy")

    with open(os.path.join(os.path.dirname(__file__), inp), 'r+') as input_data:
        input_data.seek(0)
        input_data.truncate()
        json.dump(input_copy, input_data, indent=2)
