import optking
import psi4
import qcelemental as qcel
import numpy as np

GEOMETRY = np.asarray(
    [
        0.000000,
        0.000000,
        0.000000,
        -0.000000,
        0.000000,
        1.453275,
        -0.968999,
        0.038209,
        1.565010,
        0.968989,
        -0.038209,
        -0.111740,
    ]
).reshape(12)

MOL_STRING = """0 1
O      0.000000    0.000000    0.000000
O     -0.000000    0.000000    1.453275
H     -0.968999    0.038209    1.565010
H      0.968989   -0.038209   -0.111740
units bohr
"""

def check_basic_mol(mol_dict):

    if isinstance(mol_dict['symbols'], np.ndarray):
        mol_dict.update({'symbols': mol_dict["symbols"].tolist()})
    assert ["O", "O", "H", "H"] == mol_dict["symbols"]

    if isinstance(mol_dict["geometry"], np.ndarray):
        mol_dict.update({'geometry': mol_dict["geometry"].reshape(12)})
        # Sometime coming from Psi4? The geometry is actually 4x3. Didn't know that was allowed.
    assert np.allclose(GEOMETRY, mol_dict["geometry"], atol=1e-7, rtol=0.0)


def check_basic_input_spec(opt_input):
    input_spec = {
        "driver": "gradient",
        "model": {"method": "hf", "basis": "STO-3G"},
        "keywords": {"scf_type": "pk"},  # just including a random keyword
    }

    for val in input_spec.keys():
        assert val in opt_input["input_specification"]
        assert input_spec[val] == opt_input["input_specification"][val]


def test_psi4_core_mol():
    psi4_mol = psi4.geometry(MOL_STRING + "\nNOCOM\nNOREORIENT")
    assert isinstance(psi4_mol, psi4.core.Molecule)
    _, opt_input = optking.opt_helper.from_psi4(psi4_mol)
    check_basic_mol(opt_input["initial_molecule"])


def test_psi4_qcdb_mol():
    psi4_qcdb = psi4.qcdb.molecule.Molecule(MOL_STRING + "\nNOCOM\nNOREORIENT")
    assert isinstance(psi4_qcdb, psi4.qcdb.Molecule)
    _, opt_input = optking.opt_helper.from_psi4(psi4_qcdb)
    check_basic_mol(opt_input["initial_molecule"])


def test_psi4_active_mol():
    psi4_mol = psi4.geometry(MOL_STRING + "\nNOCOM\nNOREORIENT")
    assert isinstance(psi4_mol, psi4.core.Molecule)
    _, opt_input = optking.opt_helper.from_psi4("")
    check_basic_mol(opt_input["initial_molecule"])


def build_qc():
    qc_mol = qcel.models.Molecule.from_data(MOL_STRING, dtype="psi4", fix_com=True, fix_orientation=True)
    input_spec = {
        "driver": "gradient",
        "model": {"method": "hf", "basis": "STO-3G"},
        "keywords": {"scf_type": "pk"},  # just including a random keyword
    }

    opt_input = qcel.models.OptimizationInput(
        initial_molecule=qc_mol.dict(), input_specification=input_spec
    )
    return opt_input, qc_mol


def test_qcel_mol():
    opt_input, qc_mol = build_qc()

    opt_dict1 = optking.opt_helper.from_schema(opt_input)
    check_basic_mol(opt_dict1["initial_molecule"])
    check_basic_input_spec(opt_dict1)

    opt_dict2 = optking.opt_helper.from_schema(qc_mol)
    check_basic_mol(opt_dict2["initial_molecule"])

    opt_dict1a = optking.opt_helper.from_dict(opt_input.dict())
    check_basic_mol(opt_dict1a["initial_molecule"])
    check_basic_input_spec(opt_dict1a)

    opt_dict2a = optking.opt_helper.from_dict(qc_mol.dict())
    check_basic_mol(opt_dict2a["initial_molecule"])


def test_molsys_creation():
    opt_input, _ = build_qc()
    opt_molsys = optking.molsys.Molsys.from_schema(opt_input.dict()["initial_molecule"])
    assert np.allclose(opt_molsys.geom.reshape(12), GEOMETRY, atol=1e-7, rtol=0.0)
