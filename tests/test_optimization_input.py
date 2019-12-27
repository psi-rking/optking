import pprint
import json
from qcelemental.models.procedures import OptimizationInput
from qcelemental.util.serialization import json_dumps

from optking.optwrapper import optimize_qcengine
from optking.molsys import Molsys
from optking.optimize import initialize_options, create_qcengine_computer, prepare_opt_output

from optking.compute_wrappers import QCEngineComputer


pp = pprint.PrettyPrinter(indent=2)


def test_optimization_input():

    opt_input_dict = {
        "schema_name": "qcschema_optimization_input",
        "schema_version": 1,
        "keywords": {
            "program": "psi4"
        },
        "initial_molecule": {
            "geometry": [
                 0.90,  0.80,  0.5,
                 0.00,  0.70,  0.0,
                 0.00, -0.70,  0.0,
                -0.90, -0.80,  0.5],
            "symbols": [
                "H",
                "O",
                "O",
                "H"
            ],
        },
        "input_specification": {
            "schema_name": "qcschema_input",
            "schema_version": 1,
            "driver": "gradient",
            "model": {
                "method": "HF",
                "basis": "sto-3g"
            },
            "keywords": {
                "soscf": True
            }
        }
    }
    
    opt_in = OptimizationInput(**opt_input_dict)  # Create Pydantic Model Fills all fields (including non-required)

    full_model = json.loads(json_dumps(opt_in))
    result_input = {'molecule': full_model['initial_molecule']}
    result_input.update(full_model['input_specification'])
    # Convert back to plain python dictionary
    oMolsys = Molsys.from_JSON_molecule(full_model['initial_molecule'])  # Create Optking's molecular system
    initialize_options({"OPTKING": full_model['keywords']}) 
    o_json = create_qcengine_computer(oMolsys, None, result_input)
    # Takes the o_json object and creates QCSchema formatted python dict
    opt_output = prepare_opt_output(oMolsys, o_json)

    print(json.dumps(opt_output))
