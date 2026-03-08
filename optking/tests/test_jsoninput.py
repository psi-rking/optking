import sys
import os
import json
from packaging import version
import pydantic

import pytest
import optking
import psi4

from qcelemental.testing import compare_values
from .utils import utils
from .psi4_helper import using_qcmanybody


_schver = 2 if utils.psi4_runs_v2_qcschema(psi4.__version__) else 1

# Varying number of repulsion energy decimals to check.
@pytest.mark.parametrize("schver", [1, 2])
@pytest.mark.parametrize(
    "inp,expected,num_steps",
    [
        pytest.param("json_h2o.json", (8.9064890670, -74.965901192, 3), 5),
        pytest.param("json_betapinene.json", (568.2219045869, -383.38105559, 1), 4),
        pytest.param("json_hooh_frozen.json", (37.969354880, -150.786372411, 2), 6),
        pytest.param(
            "json_lif_cp.json", (8.95167, -106.8867587, 2, 3.016), 4, marks=[using_qcmanybody, pytest.mark.long]
        ),
        pytest.param(
            "json_lif_nocp.json", (9.09281, -106.9208785, 2, 2.969), 5, marks=using_qcmanybody
        ),
    ],
)
def test_input_through_json(inp, expected, num_steps, check_iter, schver):
    if ((schver == 1 and sys.version_info >= (3, 14)) or
        (schver == 2 and not utils.qcel_impl_v2_qcschema())):
        pytest.skip()
    if schver == 1 and "lif" in inp:
        pytest.skip("ManyBody Optimization is only available for QCSchema v2. The experimental v1 GeneralizedOptimization is retired.")

    if schver == 1:
        from qcelemental.models import OptimizationInput
    elif schver == 2:
        from qcelemental.models.v2 import OptimizationInput
        inp = inp.replace(".json", ".v2.json")  # for generation (note below), use `v2_inp = inp...` and un-indent

    with open(os.path.join(os.path.dirname(__file__), inp)) as input_data:
        input_copy = json.load(input_data)
        opt_schema = OptimizationInput(**input_copy)
            # to generate v2 inp files from v1, uncomment below and run schver=1 (with qcel >=0.50 and not on py314)
            # * define v2_inp var (note above)
            # * for pretty formatting, add indent=4 to json_dumps in qcelemental/util/serialization.py
            # v2 = opt_schema.convert_v(2)
            # with open(os.path.join(os.path.dirname(__file__), v2_inp), "w") as input_data2:
            #     input_data2.write(v2.serialize("json"))
            # assert 0

    # optking.run_json_file(os.path.join(os.path.dirname(__file__), inp))
    json_dict = optking.optimize_qcengine(opt_schema)
    assert json_dict["success"] is True, json_dict["error"]["error_message"]

    if "lif" in inp:
        if schver == 1:
            assert inp, json_dict["trajectory"][-1]["schema_name"] == "qcschema_manybodyresult"
        elif schver == 2:
            assert inp, json_dict["trajectory_results"][-1]["schema_name"] == "qcschema_many_body_result"
    else:
        if schver == 1:
            assert inp, json_dict["trajectory"][-1]["schema_name"] == "qcschema_output"
        elif schver == 2:
            assert inp, json_dict["trajectory_results"][-1]["schema_name"] == "qcschema_atomic_result"

    # For testing purposes. If this works, we have properly returned the output, and added the result
    # to the original file. In order to preserve the form of the test suite, we now resore the input
    # to its original state
    # with open(os.path.join(os.path.dirname(__file__), inp)) as input_data:
    #    json_dict = json.load(input_data)

    # LAB: for the MBE optimizations, psi4.compare_values strangely segfaults python, so using compare_values from qcel
    if schver == 1:
        nre = json_dict["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]
        rete = json_dict["trajectory"][-1]["properties"]["return_energy"]
    elif schver == 2:
        nre = json_dict["trajectory_results"][-1]["properties"]["nuclear_repulsion_energy"]
        rete = json_dict["trajectory_results"][-1]["properties"]["return_energy"]
    assert compare_values(
        expected[0],
        nre,
        atol=1.0 * 10 ** -expected[2],
        label="Nuclear repulsion energy",
    )
    assert compare_values(
        expected[1],
        rete,
        atol=1.0e-6,
        label="Reference energy",
    )
    if schver == 2:
        assert compare_values(
            expected[1],
            json_dict["properties"]["return_energy"],
            atol=1.0e-6,
            label="Reference energy",
        )
    utils.compare_iterations(json_dict, num_steps, check_iter)

    if len(expected) > 3:
        assert compare_values(
            expected[3],
            json_dict["final_molecule"]["geometry"][5] - json_dict["final_molecule"]["geometry"][2],
            atol=1.0e-3,
            label="bond length",
        )

    # with open(os.path.join(os.path.dirname(__file__), inp), 'r+') as input_data:
    #    input_data.seek(0)
    #    input_data.truncate()
    #    json.dump(input_copy, input_data, indent=2)
