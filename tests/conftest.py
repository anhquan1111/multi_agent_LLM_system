import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client. Calls the real app in-process — no network, no server."""
    return TestClient(app)
