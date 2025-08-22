import copy
from pydantic import ValidationError
import qcelemental as qcel
import numpy as np
import optking
import pathlib
import pytest

"""
These tests attempt to verify that validation as well as saving the options object and reloading
occurs correctly
"""

def build_basic_mol():
    """ The exact molecule doesn't matter just need something """

    mol_string = """0 1
    O      0.000000    0.000000    0.000000
    O     -0.000000    0.000000    1.453275
    H     -0.968999    0.038209    1.565010
    H      0.968989   -0.038209   -0.111740
    """

    qc_mol = qcel.models.Molecule.from_data(mol_string, dtype="psi4")

    print(qc_mol.dict())

    qcel.models.OptimizationInput
    input_spec = {
            "driver": "gradient",
            "model": {"method": "hf", "basis": "STO-3G"},
            "keywords": {"scf_type": "pk"}  # just including a random keyword
    }
    opt_input = qcel.models.OptimizationInput(
        initial_molecule=qc_mol.dict(),
        input_specification=input_spec
    )
    return opt_input

def custom_helper_with_reload(params):
    """ Create an opthelper, take a step (just use unit forces), dump optimization to dict, and
    then reload it. Allows for testing changes in optking's internal state """

    opt_input = build_basic_mol()
    optking.optwrapper.initialize_options(params, silent=False)
    cust_helper = optking.CustomHelper(opt_input, params=params)

    cust_helper.gX = np.ones(12)
    stashed_opt = cust_helper.to_dict()
    initial_params = copy.deepcopy(stashed_opt.get('params'))

    cust_helper = optking.CustomHelper.from_dict(stashed_opt)
    new_params = copy.deepcopy(cust_helper.params.to_dict(by_alias=False))
    return cust_helper, initial_params, new_params

def assert_options_match(initial_params, new_params):
    """ This will likely be done a number of times for various options that are added. I have
    not added an exhuastive list of all optking's options at this time """

    for key, item in initial_params.items():
        assert key in new_params
        assert new_params[key] == item

def test_hessians():

    test_dir = pathlib.Path(__file__).parent

    params = {"hessian_file": f"{test_dir}/test_data/H2O2_irc.hess"}
    custom_helper, initial_params, reloaded_params = custom_helper_with_reload(params)
    print(initial_params)

    HESSIAN = np.asarray([
       [-0.0071978505, -0.0000000000,  0.0000000000],
       [ 0.0100999283,  0.0000000000, -0.0000000000],
       [-0.0100959745,  0.0000000000,  0.0000000000],
       [ 0.0071938967, -0.0000000000,  0.0000000000],
       [-0.0000000000,  0.1271202709, -0.1498069469],
       [ 0.0000000000, -0.0956752237,  0.0974019028],
       [ 0.0000000000, -0.0327566715,  0.0501829401],
       [-0.0000000000,  0.0013116242,  0.0022221040],
       [ 0.0000000000, -0.1498069469,  0.5784023616],
       [-0.0000000000,  0.1735739659, -0.5951420043],
       [-0.0000000000, -0.0215449150,  0.0094301107],
       [-0.0000000000, -0.0022221040,  0.0073095319],
       [ 0.0100999283,  0.0000000000, -0.0000000000],
       [-0.0141481332,  0.0000000000,  0.0000000000],
       [ 0.0141441793, -0.0000000000, -0.0000000000],
       [-0.0100959745,  0.0000000000,  0.0000000000],
       [ 0.0000000000, -0.0956752237,  0.1735739659],
       [ 0.0000000000,  0.5065101753, -0.1211689218],
       [-0.0000000000, -0.3780782801, -0.0739499591],
       [ 0.0000000000, -0.0327566715,  0.0215449150],
       [-0.0000000000,  0.0974019028, -0.5951420043],
       [ 0.0000000000, -0.1211689218,  0.6863986889],
       [ 0.0000000000,  0.0739499591, -0.1006867954],
       [-0.0000000000, -0.0501829401,  0.0094301107],
       [-0.0100959745,  0.0000000000, -0.0000000000],
       [ 0.0141441793, -0.0000000000,  0.0000000000],
       [-0.0141481332,  0.0000000000, -0.0000000000],
       [ 0.0100999283,  0.0000000000,  0.0000000000],
       [ 0.0000000000, -0.0327566715, -0.0215449150],
       [-0.0000000000, -0.3780782801,  0.0739499591],
       [ 0.0000000000,  0.5065101753,  0.1211689218],
       [ 0.0000000000, -0.0956752237, -0.1735739659],
       [ 0.0000000000,  0.0501829401,  0.0094301107],
       [-0.0000000000, -0.0739499591, -0.1006867954],
       [-0.0000000000,  0.1211689218,  0.6863986889],
       [ 0.0000000000, -0.0974019028, -0.5951420043],
       [ 0.0071938967, -0.0000000000, -0.0000000000],
       [-0.0100959745,  0.0000000000, -0.0000000000],
       [ 0.0100999283,  0.0000000000,  0.0000000000],
       [-0.0071978505, -0.0000000000, -0.0000000000],
       [-0.0000000000,  0.0013116242, -0.0022221040],
       [ 0.0000000000, -0.0327566715, -0.0501829401],
       [ 0.0000000000, -0.0956752237, -0.0974019028],
       [-0.0000000000,  0.1271202709,  0.1498069469],
       [ 0.0000000000,  0.0022221040,  0.0073095319],
       [ 0.0000000000,  0.0215449150,  0.0094301107],
       [ 0.0000000000, -0.1735739659, -0.5951420043],
       [-0.0000000000,  0.1498069469,  0.5784023616]])

    # TEST reading the hessian with the original value that was stored in params
    H = optking.hessian.from_file(initial_params.get("hessian_file"))
    ncart = 12
    hess = np.reshape(HESSIAN, (ncart, ncart))
    assert np.allclose(hess, H, atol=1e-10, rtol=0)

    # TEST reading the hessian after dumping opt_input to dict and reloading. Make sure that none
    # processing hasn't impacted the filename.
    H = optking.hessian.from_file(custom_helper.params.hessian_file)
    assert np.allclose(hess, H, atol=1e-10, rtol=0)

    # TEST confirm that the original option set Params.to_dict() matches the option set after
    # reloading and calling to_dict again()
    assert_options_match(initial_params, reloaded_params)

@pytest.mark.parametrize("option, keyword, valid", [
    ("frozen_bend", "1 2 3", True),
    ("frozen_bend", "1 2 3 4 5 6", True),
    ("frozen_bend", "(1 2 3) (4 5 6)", True),
    ("frozen_bend", "(1 2 3), (4 5)", False),
    ("frozen_bend", "(1 2), (4 5)", False),
    ("frozen_distance", "1 2", True),
    ("frozen_distance", "1 2 3 4 5 6", True),
    ("frozen_distance", "1 2, 3 4", True),
    ("frozen_distance", "(1 2) (3 4) (5 6)", True),
    ("frozen_distance", "(1 2), (4 5)", True),
    ("frozen_distance", "1 2 3 4 AB", False),
    ("frozen_distance", "(1 2) (3 4 5) (6)", False),
    ("frozen_dihedral", "1 2 3 4", True),
    ("frozen_dihedral", "(1 2 3 4) (4 5 6 7)", True),
    ("frozen_dihedral", "(1 2 3 4), (4 5 6 7)", True),
    ("frozen_dihedral", "1 2 3 4 5 6", False),
    ("frozen_dihedral", "(1 2 3 4), (4 5)", False),
    ("frozen_oofp", "1 2 3 4", True),
    ("frozen_oofp", "1 2 3 4 5 6", False),
    ("frozen_oofp", "(1 2 3 10) (10 4 5 6)", True),
    ("frozen_oofp", "(1 2 3 10), (10 4 5 6)", True),
    ("frozen_oofp", "(1 2 3) (4 5 6)", False),
    ("frozen_oofp", "(1 2 3), (4 5)", False),
    ("frozen_oofp", "(1 2), (4 5)", False),
    ("frozen_oofp", "(1 2, 4 5)", True),
    ("ranged_distance", "1 2 2.0 3.0", True),
    ("ranged_distance", "1 2 2 3", False),  # Needs to be float
    ("ranged_distance", "1 2 2.0 3.0 3 4 2.0 3.0 5 6 2.0 3.0", True),
    ("ranged_distance", "1 2 2.0 3.0, 3 4 2.0 10.0", True),
    ("ranged_distance", "(1 2 2.0 3.0) (3 4 2.0 3.0) (5 6 2.0 3.0)", True),
    ("ranged_distance", "(1 2 2.0 3.0), (4 5 2.0 3.0)", True),
    ("ranged_distance", "1 2 2.0 3.0 3 4 A B", False),
    ("ranged_distance", "(1 2) (3 4 5) (6)", False),
    ("ranged_dihedral", "1 2 3 4 2.0 3.0", True),
    ("ranged_dihedral", "(1 2 3 4 30.0 40.0),", True),
    ("ranged_dihedral", "(1 2 3 4 2.0 3.0) (4 5 6 7 2.0 3.0)", True),
    ("ranged_dihedral", "(1 2 3 4 2.0 3.0), (4 5 6 7 110.0 115.0)", True),
    ("ranged_dihedral", "1 2 3 4 5 6", False),
    ("ranged_dihedral", "(1 2 3 4), (4 5)", False),
    ("ranged_dihedral", "(1 2 3 4 3 4),", False),
    ("ext_force_bend", "1 2 3 'Sin(x)'", True),
    ("ext_force_bend", "1 2 3 Sin(x)", False),
    ("ext_force_bend", "1 2 3 4 'Sin(x)'", False),
    ("ext_force_bend", "1 2 3 'Sin(x)' 4 5 6 'Sin(x)'", True),
    ("ext_force_bend", "(1 2 3 'Sin(x)') (4 5 6 'Sin(x)')", True),
    ("ext_force_bend", "(1 2 3 'Sin(x)'), (4 5)", False),
    ("ext_force_bend", "(1 2) 'Sin(x)', (4 5 6) 'Sin(x)'", False),
    ("ext_force_bend", "(1 2 3 'Sin(x)'), (4 5 6) 'Sin(x)'", True),
    ("frozen_cartesian", "1 XYZ 2 X 3 Y 2 Y", True),
    ("frozen_cartesian", "(1 XYZ), (2 XYZ)", True),
    ("frozen_cartesian", "1 2 XYZ", False),
    ("frozen_cartesian", "1", False),
    ("ranged_cartesian", "(1 XYZ 2.0 3.0)", True),
    ("ranged_cartesian", "1 XYZ 3.0 4.0, 2 X 3.0 4.0", True),
    ("ranged_cartesian", "(1 XYZ 2.0 3.0 4.0)", False),
    ("ranged_cartesian", "1 XYZ 3.0 4.0 2 X 3.0 4.0, 2 M 3.0 4.0", False),
    ("ext_force_cartesian", "10 XYZ 'Sin(x)'", True),
    ("ext_force_cartesian", "10 XYZ 'Sin(x)', 4 XYZ 'Sin(x)', 3 X 'Sin(x)'", True),
    ("ext_force_cartesian", "10 XYZ 'Sin(x)', 4 XYZ 'Sin(x)', 3 X Sin(x)", False),
    ("G_CONVERGENCE", "GAU_TIGHT", True),
    ("G_CONVERGENCE", "QCHEM", True),
    ("G_CONVERGENCE", "GAUSSIAN", False),
    ("OPT_TPYE", "IRC", True),
    ("OPT_TYPE", "IRC2", False),
    ("OPT_TYPE", "2IRC", False),
    ("OPT_TYPE", "Random", False)
])
def test_regex_validation(option, keyword, valid):

    if not valid:
        with pytest.raises(ValidationError):
            params = optking.op.OptParams(**{option: keyword})
    else:
        optking.op.OptParams(**{option: keyword})
