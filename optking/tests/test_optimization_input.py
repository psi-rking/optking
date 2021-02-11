import pprint
import json
from qcelemental.models.procedures import OptimizationInput
from qcelemental.util.serialization import json_dumps

from optking.optwrapper import optimize_qcengine, optimize_psi4, initialize_options, make_computer
from optking.molsys import Molsys
from optking.optimize import prepare_opt_output

from optking.compute_wrappers import QCEngineComputer


pp = pprint.PrettyPrinter(indent=2)


def test_optimization_input():

    opt_input_dict = {
        "schema_name": "qcschema_optimization_input",
        "schema_version": 1,
        "keywords": {"program": "psi4"},
        "initial_molecule": {
            "geometry": [0.90, 0.80, 0.5, 0.00, 0.70, 0.0, 0.00, -0.70, 0.0, -0.90, -0.80, 0.5],
            "symbols": ["H", "O", "O", "H"],
        },
        "input_specification": {
            "schema_name": "qcschema_input",
            "schema_version": 1,
            "driver": "gradient",
            "model": {"method": "HF", "basis": "sto-3g"},
            "keywords": {"soscf": True},
        },
    }

    # Create Pydantic Model Fills all fields (including non-required)
    opt_in = OptimizationInput(**opt_input_dict)

    # Convert back to plain python dictionary
    full_model = json.loads(json_dumps(opt_in))

    oMolsys = Molsys.from_schema(full_model["initial_molecule"])  # Create Optking's molecular system
    initialize_options(full_model["keywords"])
    computer = make_computer(full_model, "qc")
    # Takes the o_json object and creates QCSchema formatted python dict. Has numpy elements
    opt_output = prepare_opt_output(oMolsys, computer)

    assert "success" in opt_output

    psi_computer = make_computer(full_model, "psi4")
    opt_output = prepare_opt_output(oMolsys, computer)
