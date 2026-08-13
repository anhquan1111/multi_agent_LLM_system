"""Action agent: the only node that can change data, and it never does so unasked.

The model chooses a write tool and its arguments, then `interrupt()` pauses the graph and
checkpoints the state. Nothing has been written at that point. Execution resumes only when
`POST /confirm` sends `Command(resume=True)`, and continues from the line after `interrupt()`.

This is the difference between a demo and something you would let near real data.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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


def _describe(name: str, args: dict[str, Any]) -> str:
    if name == "cancel_order":
        return f"Cancel order {args.get('order_id')} and refund it in full."
    if name == "create_ticket":
        return f"Open a support ticket for order {args.get('order_id')}: {args.get('reason')}"
    return f"Run {name} with {args}"


def action_agent_node(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """Propose one write, pause for approval, and only then execute it."""
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

    call = calls[0]
    args = dict(call["args"])
    args["customer_id"] = customer_id  # from state, never from the model
    description = _describe(call["name"], args)

    # Everything above this line is read-only. The graph stops here and the state is
    # checkpointed; the HTTP response carries `pending_action` and the thread id.
    approved = interrupt(
        {"tool": call["name"], "args": args, "description": description}
    )

    if not approved:
        return {
            "messages": [AIMessage(content=DECLINED)],
            "route": "finish",
            "pending_action": None,
            "step_count": 1,
            "last_agent_output": DECLINED,
        }

    tool = TOOLS_BY_NAME.get(call["name"])
    if tool is None:
        result = f"Unknown tool {call['name']}. Nothing changed."
    else:
        try:
            result = str(tool.invoke(args))
        except Exception as exc:  # noqa: BLE001 — reported rather than crashing the request
            result = f"That change could not be completed ({type(exc).__name__}). Nothing changed."

    return {
        "messages": [
            response,
            ToolMessage(content=result, tool_call_id=call["id"]),
            AIMessage(content=result),
        ],
        "route": "finish",
        "pending_action": None,
        "step_count": 1,
        "last_agent_output": result,
    }
