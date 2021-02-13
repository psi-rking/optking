import pytest


def pytest_configure(config):
    # Register marks to avoid warnings in installed testing
    # sync with setup.cfg
    config.addinivalue_line("markers", "long")
    config.addinivalue_line("markers", "dimers")


@pytest.fixture(scope="function", autouse=True)
def set_up():
    import optking

    optking.optparams.Params = optking.optparams.OptParams({})
