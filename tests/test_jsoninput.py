import os
import json

import jsonoptwrapper
import psi4


def test_input_through_json(): 
    refnucenergy = 8.9064890670
    refenergy = -74.965901192
 
    with open(os.path.join(os.path.dirname(__file__), 'jsoninput.json')) as input_data:
        input_copy = json.load(input_data)
    jsonoptwrapper.run_optking_json(os.path.join(os.path.dirname(__file__), 'jsoninput.json'))
    
    #For testing purposes. If this works, we have properly returned the output, and added the result
    #to the original file. In order to preserve the form of the test suite, we now resore the input
    #to its original state
    with open(os.path.join(os.path.dirname(__file__), 'jsoninput.json')) as input_data:
        json_dict = json.load(input_data)
    assert psi4.compare_values(refnucenergy, json_dict['properties']['nuclear_repulsion_energy'], 3,
         "Nuclear repulsion energy")
    assert psi4.compare_values(refenergy, json_dict['properties']['return_energy'], 6,
        "Reference energy")
         
    with open(os.path.join(os.path.dirname(__file__), 'jsoninput.json'), 'r+') as input_data:
        input_data.seek(0)
        input_data.truncate()
        json.dump(input_copy, input_data, indent=2)

def test_betapinene_json():
    refenergy = -383.381055594356
    refnucenergy = 568.2219045869700267
    with open(os.path.join(os.path.dirname(__file__), 'json_betapinene.json')) as input_data:
        input_copy = json.load(input_data)
    jsonoptwrapper.run_optking_json(os.path.join(os.path.dirname(__file__), 'json_betapinene.json'))

    #For testing purposes. If this works, we have properly returned the output, and added the result
    #to the original file. In order to preserve the form of the test suite, we now resore the input
    #to its original state
    with open(os.path.join(os.path.dirname(__file__), 'json_betapinene.json')) as input_data:
        json_dict = json.load(input_data)
    assert psi4.compare_values(refnucenergy, json_dict['properties']['nuclear_repulsion_energy'], 3,
         "Nuclear repulsion energy")
    assert psi4.compare_values(refenergy, json_dict['properties']['return_energy'], 6,
        "Reference energy")

    with open(os.path.join(os.path.dirname(__file__), 'json_betapinene.json'), 'r+') as input_data:
        input_data.seek(0)
        input_data.truncate()
        json.dump(input_copy, input_data, indent=2)

def test_frozen_coords():
    refenergy = -150.786372411

    with open(os.path.join(os.path.dirname(__file__), 'json_hooh_frozen.json')) as input_data:
        input_copy = json.load(input_data)
    jsonoptwrapper.run_optking_json(os.path.join(os.path.dirname(__file__), 'json_hooh_frozen.json'))

    #For testing purposes. If this works, we have properly returned the output, and added the result
    #to the original file. In order to preserve the form of the test suite, we now resore the input
    #to its original state
    with open(os.path.join(os.path.dirname(__file__), 'json_hooh_frozen.json')) as input_data:
        json_dict = json.load(input_data)
    assert psi4.compare_values(refenergy, json_dict['properties']['return_energy'], 6, \
        "Reference energy")

    with open(os.path.join(os.path.dirname(__file__), 'json_hooh_frozen.json'), 'r+') as input_data:
        input_data.seek(0)
        input_data.truncate()
        json.dump(input_copy, input_data, indent=2)
 
