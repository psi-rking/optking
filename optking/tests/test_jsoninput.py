import os
import json
from packaging import version

import pytest
import optking
import psi4

from qcelemental.models import OptimizationInput
from qcelemental.testing import compare_values
from .utils import utils
from .psi4_helper import using_qcmanybody

# Varying number of repulsion energy decimals to check.
@pytest.mark.parametrize(
    "inp,expected,num_steps",
    [
        pytest.param("json_h2o.json", (8.9064890670, -74.965901192, 3), 5),
        pytest.param("json_betapinene.json", (568.2219045869, -383.38105559, 1), 4),
        pytest.param("json_hooh_frozen.json", (37.969354880, -150.786372411, 2), 6),
        pytest.param("json_lif_cp.json", (8.95167, -106.8867587, 2, 3.016), 4, marks=using_qcmanybody),
        pytest.param("json_lif_nocp.json", (9.09281, -106.9208785, 2, 2.969), 5, marks=using_qcmanybody),
    ],
)
def test_input_through_json(inp, expected, num_steps, check_iter):
    with open(os.path.join(os.path.dirname(__file__), inp)) as input_data:
        input_copy = json.load(input_data)
        if "lif" in inp:
            import qcmanybody
            if version.Version(qcmanybody.__version__) >= version.Version("0.5"):
                from qcmanybody.models.v1.generalized_optimization import GeneralizedOptimizationInput
            else:
                from qcmanybody.models.generalized_optimization import GeneralizedOptimizationInput
            opt_schema = GeneralizedOptimizationInput(**input_copy)
        else:
            opt_schema = OptimizationInput(**input_copy)

        # Note it's important to have `input_specification.schema_name = "qcschema_manybodyspecification"`
        #   in your json for a MBE optimization. Or you can explicitly construct a
        #   GeneralizedOptimizationInput like above.

    # optking.run_json_file(os.path.join(os.path.dirname(__file__), inp))
    json_dict = optking.optimize_qcengine(opt_schema)

    if "lif" in inp:
        assert inp, json_dict["trajectory"][-1]["schema_name"] == "qcschema_manybodyresult"
    else:
        assert inp, json_dict["trajectory"][-1]["schema_name"] == "qcschema_output"

    # For testing purposes. If this works, we have properly returned the output, and added the result
    # to the original file. In order to preserve the form of the test suite, we now resore the input
    # to its original state
    # with open(os.path.join(os.path.dirname(__file__), inp)) as input_data:
    #    json_dict = json.load(input_data)

    # LAB: for the MBE optimizations, psi4.compare_values strangely segfaults python, so using compare_values from qcel
    assert compare_values(
        expected[0],
        json_dict["trajectory"][-1]["properties"]["nuclear_repulsion_energy"],
        atol=1.0 * 10**-expected[2],
        label="Nuclear repulsion energy",
    )
    assert compare_values(
        expected[1], json_dict["trajectory"][-1]["properties"]["return_energy"], atol=1.e-6, label="Reference energy"
    )
    utils.compare_iterations(json_dict, num_steps, check_iter)

    if len(expected) > 3:
        assert compare_values(expected[3], json_dict["final_molecule"]["geometry"][5] - json_dict["final_molecule"]["geometry"][2], atol=1.e-3, label="bond length")

    # with open(os.path.join(os.path.dirname(__file__), inp), 'r+') as input_data:
    #    input_data.seek(0)
    #    input_data.truncate()
    #    json.dump(input_copy, input_data, indent=2)
