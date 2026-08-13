"""Policy agent with the LLM and retrieval both faked — fast and deterministic."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.policy_agent import NO_CONTEXT, policy_agent_node
from src.rag.chunking import Chunk

CHUNKS = [
    Chunk("Damaged items may be returned within 45 days.", "returns.md", "Damaged", "returns.md#2"),
    Chunk("Laptops carry a 24 month warranty.", "warranty.md", "Coverage", "warranty.md#0"),
]


@pytest.fixture
def fake_llm():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="You have 45 days for damaged items [returns.md].")
    return llm


def state(question: str) -> dict:
    return {"messages": [HumanMessage(question)], "step_count": 0}


def test_answers_from_retrieved_chunks(monkeypatch, fake_llm):
    monkeypatch.setattr("src.agents.policy_agent.search", lambda q, k=4: CHUNKS)

    result = policy_agent_node(state("damaged item return window?"), llm=fake_llm)

    assert "45 days" in result["messages"][0].content
    assert result["step_count"] == 1


def test_cites_only_the_sources_the_answer_mentions(monkeypatch, fake_llm):
    monkeypatch.setattr("src.agents.policy_agent.search", lambda q, k=4: CHUNKS)

    result = policy_agent_node(state("damaged item return window?"), llm=fake_llm)

    assert [c["source"] for c in result["citations"]] == ["returns.md"]


def test_retrieved_chunks_are_actually_put_in_the_prompt(monkeypatch, fake_llm):
    monkeypatch.setattr("src.agents.policy_agent.search", lambda q, k=4: CHUNKS)

    policy_agent_node(state("anything"), llm=fake_llm)

    system_prompt = fake_llm.invoke.call_args[0][0][0].content
    assert "45 days" in system_prompt
    assert "returns.md" in system_prompt


def test_empty_index_does_not_call_the_llm(monkeypatch, fake_llm):
    """No context must mean no API call — answering anyway is how hallucination starts."""
    monkeypatch.setattr("src.agents.policy_agent.search", lambda q, k=4: [])

    result = policy_agent_node(state("anything"), llm=fake_llm)

    fake_llm.invoke.assert_not_called()
    assert result["messages"][0].content == NO_CONTEXT
    assert result.get("citations", []) == []


def test_declining_to_answer_cites_nothing(monkeypatch):
    """An unused source attached to a refusal is worse than no source at all."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="I do not have that information.")
    monkeypatch.setattr("src.agents.policy_agent.search", lambda q, k=4: CHUNKS)

    result = policy_agent_node(state("what is the capital of France?"), llm=llm)

    assert result.get("citations", []) == []


def test_handles_content_returned_as_blocks(monkeypatch):
    """Models that emit reasoning signatures put a list in .content, not a string."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content=[{"type": "text", "text": "45 days for damaged items [returns.md]."}]
    )
    monkeypatch.setattr("src.agents.policy_agent.search", lambda q, k=4: CHUNKS)

    result = policy_agent_node(state("damaged?"), llm=llm)

    assert result["messages"][0].content.startswith("45 days")
    assert [c["source"] for c in result["citations"]] == ["returns.md"]
