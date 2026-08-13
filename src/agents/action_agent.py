"""Write path, split across two nodes on purpose.

`propose_action_node` asks the model what to change and writes the proposal into state.
`confirm_action_node` pauses on `interrupt()` and, once resumed, executes *that stored
proposal*.

The split is not cosmetic. LangGraph re-runs an interrupted node from the top when it
resumes, so a single node holding both the model call and the interrupt would ask the model
again after the customer clicked approve — and could then execute a *different* action from
the one that was shown to them. Splitting means the proposal is checkpointed before the
pause, so what runs is exactly what was approved. It also halves the model calls.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.agents.llm import get_llm, with_retry
from src.agents.messages import last_user_question
from src.agents.state import AgentState
from src.tools.write_tools import WRITE_TOOLS

TOOLS_BY_NAME = {t.name: t for t in WRITE_TOOLS}

SYSTEM_PROMPT = """You carry out changes a customer asks for: cancelling an order, or opening
a support ticket.

The customer you are serving is {customer_id}. Always pass that exact customer_id.

Call exactly one tool for the change that was asked for. Do not call a tool if the customer
is only asking a question — in that case reply in words instead. Never claim a change has
been made; the system reports the result after the customer confirms."""

DECLINED = "Understood — I have not made any changes."
NOTHING_TO_DO = "I could not identify a change to make. Could you say what you would like done?"
LOST_PROPOSAL = "That request expired before it was confirmed. Please ask again."


def describe(tool_name: str, args: dict[str, Any]) -> str:
    """A sentence the customer reads before approving. Never show raw arguments."""
    if tool_name == "cancel_order":
        return f"Cancel order {args.get('order_id')} and refund it in full."
    if tool_name == "create_ticket":
        return f"Open a support ticket for order {args.get('order_id')}: {args.get('reason')}"
    return f"Run {tool_name} with {args}"


def propose_action_node(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """Decide what change to make. Reads only — nothing is written to the shop database."""
    customer_id = state.get("customer_id", "")
    model = (llm or get_llm()).bind_tools(WRITE_TOOLS)

    response = with_retry(model.invoke)(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(customer_id=customer_id)),
            HumanMessage(content=last_user_question(state)),
        ]
    )

    calls = getattr(response, "tool_calls", None)
    if not calls:
        text = response.text.strip() or NOTHING_TO_DO
        return {
            "messages": [AIMessage(content=text)],
            "route": "finish",
            "step_count": 1,
            "last_agent_output": text,
        }

    # One write per turn, deliberately — unlike the read agent, which loops over every tool
    # call the model makes. Approval is a single yes/no, so batching several writes behind
    # one click would ask the customer to approve things individually invisible to them.
    # Extra proposals are dropped; the customer can ask again for the next change.
    call = calls[0]
    args = dict(call["args"])
    args["customer_id"] = customer_id  # from state, never from the model

    return {
        "pending_action": {
            "tool": call["name"],
            "args": args,
            "description": describe(call["name"], args),
        },
        "step_count": 1,
    }


def confirm_action_node(state: AgentState) -> dict:
    """Pause for approval, then execute exactly what was proposed.

    On resume LangGraph re-runs this node from the top, but everything it needs is read from
    state rather than regenerated, so the second pass is deterministic.
    """
    proposed = state.get("pending_action")
    if not proposed:
        return _finish(LOST_PROPOSAL)

    approved = interrupt(proposed)
    if not approved:
        return _finish(DECLINED)

    tool = TOOLS_BY_NAME.get(proposed["tool"])
    if tool is None:
        return _finish(f"Unknown action {proposed['tool']}. Nothing changed.")

    try:
        result = str(tool.invoke(proposed["args"]))
    except Exception as exc:  # noqa: BLE001 — reported rather than crashing the request
        result = f"That change could not be completed ({type(exc).__name__}). Nothing changed."

    return _finish(result)


def _finish(message: str) -> dict:
    return {
        "messages": [AIMessage(content=message)],
        "route": "finish",
        "pending_action": None,
        "step_count": 1,
        "last_agent_output": message,
    }


def route_after_proposal(state: AgentState) -> str:
    """Skip the approval step entirely when the model chose not to change anything."""
    return "confirm_action" if state.get("pending_action") else "supervisor"
