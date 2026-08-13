"""Graph-level tests: routing and state plumbing, called directly with no HTTP layer."""

from langchain_core.messages import HumanMessage

from src.agents.graph import build_graph


def invoke(step_count: int = 0, message: str = "what is the return policy?") -> dict:
    return build_graph().invoke(
        {"messages": [HumanMessage(message)], "customer_id": "C007", "step_count": step_count}
    )


def test_graph_returns_an_answer():
    assert invoke()["messages"][-1].text


def test_graph_keeps_the_incoming_message():
    """add_messages appends rather than replaces, so the question survives the round trip."""
    result = invoke()

    assert result["messages"][0].content == "what is the return policy?"
    assert len(result["messages"]) >= 2


def test_step_count_counts_supervisor_and_specialist_turns():
    """supervisor → policy_agent → supervisor → END is three turns, summed by operator.add."""
    assert invoke()["step_count"] == 3


def test_citations_reach_the_final_state():
    citations = invoke()["citations"]

    assert citations
    assert set(citations[0]) == {"source", "chunk_id"}


def test_step_cap_ends_the_run_without_calling_the_model(fake_llm):
    """At the cap the supervisor must stop before spending another API call."""
    result = invoke(step_count=99)

    assert result["error"] == "step_cap"
    assert fake_llm.calls == []
