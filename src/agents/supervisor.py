"""Supervisor: reads the request and picks one specialist. Nothing else.

It never touches data and never writes prose — it emits one label. Constraining the output
to a `Literal` via structured output removes a whole class of string-parsing bugs.

The routing decision and the dispatch are deliberately separate: `supervisor_node` makes the
LLM call, `route_from_supervisor` is a pure function of state. That makes the guardrails
unit-testable without mocking anything.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.llm import get_llm, with_retry
from src.agents.messages import transcript
from src.agents.state import AgentState, Route
from src.config import settings

STUCK_MESSAGE = (
    "I am not making progress on this request. Please rephrase it, or contact support."
)
CAP_MESSAGE = (
    "This request needed more steps than I am allowed to take. "
    "Please ask a narrower question, or contact support."
)

SYSTEM_PROMPT = """You route customer support requests to exactly one specialist.

- policy_agent — questions about shop rules: returns, shipping, warranty, payment, privacy,
  membership tiers. Anything answerable from a policy document.
- order_agent — questions about this customer's own orders or about products: status,
  totals, dates, what was in an order, product price or stock. READ ONLY.
- action_agent — requests to CHANGE something: cancel an order, open a support ticket.
- finish — the conversation already contains a complete answer, or the request is outside
  what this shop's support can help with.

Route to exactly one. If a request needs both a policy answer and an order lookup, pick the
one that is still missing from the conversation so far."""


class RoutingDecision(BaseModel):
    route: Route = Field(description="Which specialist should handle the request next")


ROUTING_QUESTION = (
    "Given the conversation above, which specialist should handle the request next? "
    "Answer with one label."
)


def supervisor_node(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """Choose the next specialist, or finish.

    Two guardrails run before the LLM is consulted, because both mean the request is already
    doomed and spending another API call on it would only cost money.
    """
    if state.get("step_count", 0) >= settings.max_agent_steps:
        return {
            "route": "finish",
            "error": "step_cap",
            "messages": [AIMessage(content=CAP_MESSAGE)],
            "step_count": 1,
        }

    model = llm or get_llm()
    decision = with_retry(model.with_structured_output(RoutingDecision).invoke)(
        [SystemMessage(content=SYSTEM_PROMPT), *transcript(state, ROUTING_QUESTION)]
    )

    return {"route": decision.route, "step_count": 1}


def route_from_supervisor(state: AgentState) -> str:
    """Pure dispatch: no LLM call, no side effects, trivially testable."""
    if state.get("error"):
        return "finish"
    if state.get("step_count", 0) >= settings.max_agent_steps:
        return "finish"
    return state.get("route", "finish")


def detect_stuck(state: AgentState, new_output: str) -> bool:
    """True when a specialist has just repeated itself word for word.

    The step cap always stops a runaway loop eventually, but it costs the full budget to do
    so. This catches the common case — an agent that cannot make progress and says the same
    thing again — on the second iteration instead of the eighth.
    """
    previous = state.get("last_agent_output", "")
    return bool(previous) and previous.strip() == new_output.strip()


def stuck_result() -> dict:
    return {
        "route": "finish",
        "error": "stuck",
        "messages": [AIMessage(content=STUCK_MESSAGE)],
        "step_count": 1,
    }
