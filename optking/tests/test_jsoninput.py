import os
import json

import pytest
import optking
import psi4

from qcelemental.models import OptimizationInput
from .utils import utils

# Varying number of repulsion energy decimals to check.
@pytest.mark.parametrize(
    "inp,expected,num_steps",
    [
        ("json_h2o.json", (8.9064890670, -74.965901192, 3), 5),
        ("json_betapinene.json", (568.2219045869, -383.38105559, 1), 4),
        ("json_hooh_frozen.json", (37.969354880, -150.786372411, 2), 6),
    ],
)
def test_input_through_json(inp, expected, num_steps, check_iter):
    with open(os.path.join(os.path.dirname(__file__), inp)) as input_data:
        input_copy = json.load(input_data)
        opt_schema = OptimizationInput(**input_copy)

    # optking.run_json_file(os.path.join(os.path.dirname(__file__), inp))
    json_dict = optking.optimize_qcengine(input_copy)

    # For testing purposes. If this works, we have properly returned the output, and added the result
    # to the original file. In order to preserve the form of the test suite, we now resore the input
    # to its original state
    # with open(os.path.join(os.path.dirname(__file__), inp)) as input_data:
    #    json_dict = json.load(input_data)
    assert psi4.compare_values(
        expected[0],
        json_dict["trajectory"][-1]["properties"]["nuclear_repulsion_energy"],
        expected[2],
        "Nuclear repulsion energy",
    )
    assert psi4.compare_values(
        expected[1], json_dict["trajectory"][-1]["properties"]["return_energy"], 6, "Reference energy"
    )
    utils.compare_iterations(json_dict, num_steps, check_iter)

    # with open(os.path.join(os.path.dirname(__file__), inp), 'r+') as input_data:
    #    input_data.seek(0)
    #    input_data.truncate()
    #    json.dump(input_copy, input_data, indent=2)
