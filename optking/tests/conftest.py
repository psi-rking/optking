import pytest


def pytest_configure(config):
    # Register marks to avoid warnings in installed testing
    # sync with setup.cfg
    config.addinivalue_line("markers", "long")
    config.addinivalue_line("markers", "dimers")
    config.addinivalue_line("markers", "multimers")


@pytest.fixture(scope="function", autouse=True)
def set_up():
    import optking

def pytest_addoption(parser):
    parser.addoption(
        "--check_iter",
        action="store",
        default=0,
        help="1 -- raise error if # of steps taken doesn't match expected. 0 (default) -- ignore mismatch but warn user in log",
    )


@pytest.fixture
def check_iter(request):
    return request.config.getoption("--check_iter")
