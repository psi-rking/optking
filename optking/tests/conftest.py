import pathlib
import pytest
import os


def pytest_configure(config):
    # Register marks to avoid warnings in installed testing
    # sync with setup.cfg
    config.addinivalue_line("markers", "long")
    config.addinivalue_line("markers", "pubchem")
    config.addinivalue_line("markers", "dimers")
    config.addinivalue_line("markers", "multimers")
    config.addinivalue_line("markers", "dftd3")


@pytest.fixture(scope="function", autouse=True)
def set_up():
    import psi4
    import optking
    yield
    mol = psi4.core.get_active_molecule()
    if not mol:
        return
    psi4_hess = pathlib.Path(psi4.core.get_writer_file_prefix(mol.name()) + ".hess")
    psi4_vibrec = pathlib.Path(psi4.core.get_writer_file_prefix(mol.name()) + ".vibrec")
    if psi4_hess.exists():
        os.remove(str(psi4_hess))
    if psi4_vibrec.exists():
        os.remove(str(psi4_vibrec))
    psi4.core.clean_options()
    psi4.core.clean()

@pytest.fixture(scope="function", autouse=True)
def psi4_setup(psi4_threads, psi4_mem):
    import psi4
    psi4.set_num_threads(int(psi4_threads))
    psi4.set_memory(f'{psi4_mem} GB')

def pytest_addoption(parser):
    parser.addoption(
        "--check_iter",
        action="store",
        default=0,
        help="1 -- raise error if # of steps taken doesn't match expected. 0 (default) -- ignore mismatch but warn user in log",
    )

    parser.addoption(
        "--psi4_threads",
        action="store",
        default=1,
        help="1 -- raise error if # of steps taken doesn't match expected. 0 (default) -- ignore mismatch but warn user in log",
    )

    parser.addoption(
        "--psi4_mem",
        action="store",
        default=0.5,
        help="1 -- raise error if # of steps taken doesn't match expected. 0 (default) -- ignore mismatch but warn user in log",
    )

    parser.addoption(
        "--keep-files",
        action="store",
        default=False,
        help="if True, don't remove files after last test for a successful run"
    )

@pytest.fixture()
def irc_cleanup():
    import psi4
    import os
    import pathlib

    yield
    irc_traj = pathlib.Path(f"irc_traj.{os.getpid()}.xyz")
    if irc_traj.exists():
        os.remove(str(irc_traj))

@pytest.hookimpl(trylast=True)
def cleanup_output_files(request):
    cleanup = [
        pathlib.Path("output.log"),
        pathlib.Path("output.dat"),
        pathlib.Path("opt_log.out"),
        pathlib.Path("irc_progress.log"),
        pathlib.Path("timer.dat"),
        pathlib.Path("ijk.dat")]
    if not request.config.getoption("--keep-files"):
        for f in cleanup:
            if f.exists():
                os.remove(str(f))

@pytest.fixture
def check_iter(request):
    return request.config.getoption("--check_iter")

@pytest.fixture
def psi4_threads(request):
    return request.config.getoption("--psi4_threads")

@pytest.fixture
def psi4_mem(request):
    return request.config.getoption("--psi4_mem")

