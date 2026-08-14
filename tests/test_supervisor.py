"""Routing and guardrails. `route_from_supervisor` is pure, so most of this needs no mocks."""

from langchain_core.messages import HumanMessage

from src.agents.supervisor import (
    detect_stuck,
    route_from_supervisor,
    supervisor_node,
)
from src.config import settings


def test_routes_to_whatever_the_supervisor_chose():
    assert route_from_supervisor({"route": "order_agent", "step_count": 1}) == "order_agent"


def test_an_error_ends_the_run_regardless_of_route():
    state = {"route": "order_agent", "error": "stuck", "step_count": 1}

    assert route_from_supervisor(state) == "finish"


def test_the_step_cap_overrides_the_route():
    state = {"route": "order_agent", "step_count": settings.max_agent_steps}

    assert route_from_supervisor(state) == "finish"


def test_missing_route_defaults_to_finish():
    """A malformed state must stop, not loop."""
    assert route_from_supervisor({}) == "finish"


def test_stuck_detection_fires_on_an_identical_repeat():
    assert detect_stuck({"last_agent_output": "no orders found"}, "no orders found")


def test_stuck_detection_ignores_the_first_turn():
    """There is nothing to compare against yet, so it must not fire."""
    assert not detect_stuck({}, "no orders found")


def test_stuck_detection_allows_progress():
    assert not detect_stuck({"last_agent_output": "no orders found"}, "order 1042 is shipped")


def test_supervisor_asks_the_model_and_returns_a_label(fake_llm):
    state = {"messages": [HumanMessage("where is order 1042?")], "step_count": 0}

    result = supervisor_node(state, llm=fake_llm)

    assert result["route"] == "policy_agent"
    assert result["step_count"] == 1


def test_supervisor_stops_at_the_cap_without_calling_the_model(fake_llm):
    state = {"messages": [HumanMessage("hi")], "step_count": settings.max_agent_steps}

    result = supervisor_node(state, llm=fake_llm)

    assert result["route"] == "finish"
    assert result["error"] == "step_cap"
    assert fake_llm.calls == []


def test_transcript_keeps_answers_but_drops_tool_plumbing():
    """The supervisor is bound to no tools; sending it tool calls is noise and risk."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from src.agents.messages import transcript

    state = {
        "messages": [
            HumanMessage("status of order 1006?"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_order", "args": {}, "id": "c1", "type": "tool_call"}],
            ),
            ToolMessage(content="{'status': 'processing'}", tool_call_id="c1"),
            AIMessage("Order 1006 is processing."),
        ]
    }

    result = transcript(state, "which specialist next?")

    kinds = [type(m).__name__ for m in result]
    assert kinds == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert result[-1].content == "which specialist next?"
    assert not any(getattr(m, "tool_calls", None) for m in result)


def test_a_greeting_gets_an_answer_not_an_empty_string(fake_llm):
    """Finishing with nothing said sends the customer an empty response."""
    from src.agents.supervisor import CAPABILITIES_MESSAGE

    fake_llm.route = "finish"
    fake_llm.routed = True  # force "finish" on the first turn

    result = supervisor_node({"messages": [HumanMessage("hello")], "step_count": 0}, llm=fake_llm)

    assert result["route"] == "finish"
    assert result["messages"][0].content == CAPABILITIES_MESSAGE


def test_finishing_after_a_specialist_answered_adds_nothing(fake_llm):
    """The capability message must not be appended to a real answer."""
    from langchain_core.messages import AIMessage

    fake_llm.route = "finish"
    fake_llm.routed = True
    state = {
        "messages": [HumanMessage("return policy?"), AIMessage("You have 30 days [returns.md].")],
        "step_count": 2,
    }

    result = supervisor_node(state, llm=fake_llm)

    assert "messages" not in result


def test_a_specialist_is_never_asked_twice_in_one_turn():
    """order_agent reporting "no such order" is an answer, not a failure to retry.

    Four apologies deep the run hit the step cap, and stuck detection could not catch it
    because the model reworded the refusal every time. This guard does not depend on the model
    reading the prompt.
    """
    state = {"route": "order_agent", "visited": ["order_agent"], "step_count": 2}

    assert route_from_supervisor(state) == "finish"


def test_a_different_specialist_is_still_allowed():
    """A two-part question legitimately needs two different specialists."""
    state = {"route": "policy_agent", "visited": ["action_agent"], "step_count": 2}

    assert route_from_supervisor(state) == "policy_agent"
