import pytest


def pytest_configure(config):
    # Register marks to avoid warnings in installed testing
    # sync with setup.cfg
    config.addinivalue_line("markers", "long")
    config.addinivalue_line("markers", "dimers")
    config.addinivalue_line("markers", "multimers")
    config.addinivalue_line("markers", "dftd3")


@pytest.fixture(scope="function", autouse=True)
def set_up():
    import psi4
    import optking
    yield
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

@pytest.fixture
def check_iter(request):
    return request.config.getoption("--check_iter")

@pytest.fixture
def psi4_threads(request):
    return request.config.getoption("--psi4_threads")

@pytest.fixture
def psi4_mem(request):
    return request.config.getoption("--psi4_mem")
