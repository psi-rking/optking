import json
import os
import pathlib
import qcelemental as qcel
from optking import misc

test_dir = pathlib.Path(__file__).parent

# Unit tests for writing trajectory files from OptimizationResults.
# Integration tests are in test_irc_hooh and test_hf_g_opt

def create_molecules(traj_file: pathlib.Path, natom, symbols):
    with traj_file.open() as f:
        lines = f.readlines()

    # Remove before assertions. Makes sure we don't accumulate files
    os.system(f'rm {str(traj_file)}')

    first = "".join(lines[:natom + 2])
    last = "".join(lines[-(natom + 2):])
    first_mol = qcel.molparse.from_string(first, dtype='xyz')
    last_mol = qcel.molparse.from_string(last, dtype='xyz')

    assert first_mol.get('qm').get('elem').tolist() == symbols
    assert last_mol.get('qm').get('elem').tolist() == symbols

def test_irc_trajectory():
    with open(f"{test_dir}/test_data/irc_history.json", "r") as f:
        opt_result = json.load(f)

    misc.write_irc_xyz_trajectory(opt_result)
    traj_file = pathlib.Path(f"irc_traj.{os.getpid()}.xyz")
    create_molecules(traj_file, natom=4, symbols=['H', 'O', 'O', 'H'])

def test_opt_trajectory():
    with open(f"{test_dir}/test_data/opt_history.json", "r") as f:
        opt_result = json.load(f)

    misc.write_opt_xyz_trajectory(opt_result)
    traj_file = pathlib.Path(f"opt_traj.{os.getpid()}.xyz")
    create_molecules(traj_file, natom=3, symbols=['O', 'H', 'H'])
