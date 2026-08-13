"""Graph-level tests: the state plumbing, called directly with no HTTP layer."""

from langchain_core.messages import HumanMessage

from src.agents.graph import build_graph


def invoke(step_count: int = 0) -> dict:
    return build_graph().invoke(
        {"messages": [HumanMessage("what is the return policy?")], "step_count": step_count}
    )


def test_graph_returns_an_answer():
    assert invoke()["messages"][-1].content


def test_graph_keeps_the_incoming_message():
    """add_messages appends rather than replaces, so the question survives the round trip."""
    result = invoke()

    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "what is the return policy?"


def test_step_count_reducer_accumulates():
    """step_count uses operator.add: the node returns 1 and it sums onto the initial value."""
    assert invoke(step_count=5)["step_count"] == 6


def test_citations_reach_the_final_state():
    citations = invoke()["citations"]

    assert citations
    assert set(citations[0]) == {"source", "chunk_id"}
