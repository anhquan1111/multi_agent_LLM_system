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


def test_a_conversation_survives_more_than_three_turns(fake_llm):
    """The cap must bound one request, not the whole conversation.

    With operator.add the count persisted across turns of the same thread, so the third
    message on a thread hit the cap however simple it was.
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    graph = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "same-thread"}}

    for turn in range(1, 6):
        fake_llm.routed = False
        result = graph.invoke(
            {
                "messages": [HumanMessage("what is the return policy?")],
                "step_count": 0,
                "citations": [],
                "last_agent_output": "",
                "error": None,
            },
            config=config,
        )
        assert result.get("error") is None, f"turn {turn} hit the cap"
        assert result["step_count"] <= 3


def test_citations_do_not_leak_between_turns(fake_llm):
    """Sources from an earlier question must not be attached to a later answer."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    graph = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "same-thread"}}
    turn = {
        "messages": [HumanMessage("return policy?")],
        "step_count": 0,
        "citations": [],
        "last_agent_output": "",
        "error": None,
    }

    first = graph.invoke(turn, config=config)
    fake_llm.routed = False  # let the stand-in route to a specialist again
    second = graph.invoke(turn, config=config)

    assert len(first["citations"]) == 1
    assert len(second["citations"]) == 1, "the previous turn's sources came along"


def test_asking_the_same_question_twice_is_not_stuck(fake_llm):
    """Stuck detection compares hops inside one request, not messages across a conversation."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    graph = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "repeat"}}
    turn = {
        "messages": [HumanMessage("return policy?")],
        "step_count": 0,
        "citations": [],
        "last_agent_output": "",
        "error": None,
    }

    graph.invoke(turn, config=config)
    fake_llm.routed = False
    second = graph.invoke(turn, config=config)

    assert second.get("error") is None
