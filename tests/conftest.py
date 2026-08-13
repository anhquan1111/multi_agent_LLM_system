from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.rag.chunking import Chunk

FAKE_ANSWER = "Damaged items can be returned within 45 days [returns.md]."
FAKE_CHUNKS = [
    Chunk("Damaged items may be returned within 45 days.", "returns.md", "Damaged", "returns.md#2"),
]


@pytest.fixture(autouse=True)
def no_live_calls(monkeypatch, request):
    """Cut every network dependency out of the test suite.

    Retrieval and the LLM are replaced globally, so no test can reach Chroma or Gemini by
    accident. Tests marked `live` opt out.
    """
    if request.node.get_closest_marker("live"):
        return

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=FAKE_ANSWER)
    monkeypatch.setattr("src.agents.policy_agent.search", lambda q, k=4: FAKE_CHUNKS)
    monkeypatch.setattr("src.agents.policy_agent.get_llm", lambda: fake_llm)


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client. Calls the real app in-process — no network, no server."""
    from src.api.main import app

    return TestClient(app)
