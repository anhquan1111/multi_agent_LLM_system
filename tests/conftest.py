"""Shared test fixtures.

The suite never touches the network. Retrieval and every LLM entry point are replaced by
default, so a newly written test cannot reach Chroma or Gemini by accident — it is opt-out
(`@pytest.mark.live`), not opt-in.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.agents.supervisor import RoutingDecision
from src.rag.chunking import Chunk

FAKE_ANSWER = "Damaged items can be returned within 45 days [returns.md]."
FAKE_CHUNKS = [
    Chunk("Damaged items may be returned within 45 days.", "returns.md", "Damaged", "returns.md#2"),
]

LLM_ENTRY_POINTS = [
    "src.agents.supervisor.get_llm",
    "src.agents.policy_agent.get_llm",
    "src.agents.order_agent.get_llm",
    "src.agents.action_agent.get_llm",
]


class _Structured:
    """Stands in for `llm.with_structured_output(RoutingDecision)`."""

    def __init__(self, parent: "FakeLLM"):
        self._parent = parent

    def invoke(self, _messages):
        return RoutingDecision(route=self._parent.next_route())


class FakeLLM:
    """A deterministic stand-in for a chat model.

    Routes to `route` on the first supervisor turn and to `finish` afterwards, which is what a
    real supervisor does once a specialist has answered. Without that, every graph test would
    loop until the step cap.
    """

    def __init__(self, route: str = "policy_agent", answer: str = FAKE_ANSWER, tool_calls=None):
        self.route = route
        self.answer = answer
        self.tool_calls = tool_calls or []
        self.routed = False
        self.calls: list = []

    def next_route(self) -> str:
        if self.routed:
            return "finish"
        self.routed = True
        return self.route

    def with_structured_output(self, _schema):
        return _Structured(self)

    def bind_tools(self, tools):
        self.bound_tools = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.answer, tool_calls=self.tool_calls)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture(autouse=True)
def no_live_calls(monkeypatch, request, fake_llm):
    """Cut every network dependency out of the test suite."""
    if request.node.get_closest_marker("live"):
        return

    monkeypatch.setattr("src.agents.policy_agent.search", lambda q, k=4: FAKE_CHUNKS)
    for target in LLM_ENTRY_POINTS:
        monkeypatch.setattr(target, lambda: fake_llm)


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client. Calls the real app in-process — no network, no server."""
    from src.api.main import app

    return TestClient(app)
