import pytest


@pytest.fixture(scope="function", autouse=True)
def set_up():
    import optking

    optking.optparams.Params = optking.optparams.OptParams({})
