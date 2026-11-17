import pytest
from fakes import FakeBackend, fast_settings
from fastapi.testclient import TestClient

from icp_api.main import create_app


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def smoke_errors() -> dict[str, str]:
    """endpoint -> error message the fake test completion should return."""
    return {}


@pytest.fixture
def client(backend, smoke_errors):
    async def smoke(endpoint: str, model: str, timeout: float) -> str | None:
        return smoke_errors.get(endpoint)

    app = create_app(settings=fast_settings(), backends={"kubernetes": backend}, smoke=smoke)
    with TestClient(app) as c:
        yield c
