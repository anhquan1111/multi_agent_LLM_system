"""Unit tests for the graph, called directly — no HTTP layer involved."""

from langchain_core.messages import HumanMessage

from src.agents.graph import SKELETON_ANSWER, build_graph


def test_graph_returns_an_answer():
    graph = build_graph()

    result = graph.invoke({"messages": [HumanMessage("hello")], "step_count": 0})

    assert result["messages"][-1].content == SKELETON_ANSWER


def test_graph_keeps_the_incoming_message():
    """add_messages appends rather than replaces, so the question survives the round trip."""
    graph = build_graph()

    result = graph.invoke({"messages": [HumanMessage("hello")], "step_count": 0})

    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "hello"


def test_step_count_reducer_accumulates():
    """step_count uses operator.add: the node returns 1 and it sums onto the initial value."""
    graph = build_graph()

    result = graph.invoke({"messages": [HumanMessage("hello")], "step_count": 5})

    assert result["step_count"] == 6
