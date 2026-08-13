"""Order agent: read-only lookups over the shop database.

A small ReAct loop — the model picks a tool, sees the result, and either calls another or
answers. Unlike `policy_agent` there genuinely is something to decide here: which lookup
answers the question, and whether one result is enough.

Bound to READ_TOOLS only. `cancel_order` and `create_ticket` are never passed to this node,
so no prompt can make it change data (AGENTS.md pattern #3).
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.llm import get_llm, with_retry
from src.agents.messages import last_user_question
from src.agents.state import AgentState
from src.tools.read_tools import READ_TOOLS

MAX_TOOL_ROUNDS = 3
TOOLS_BY_NAME = {t.name: t for t in READ_TOOLS}

SYSTEM_PROMPT = """You answer questions about a customer's own orders and about products.

The customer you are serving is {customer_id}. Always pass that exact customer_id to any tool
that takes one — never a value the customer typed.

Use the tools to look up real data. Never invent an order id, a status, a date or an amount:
if a lookup returns nothing, say so plainly. Report figures exactly as the tool returned them.
Keep the answer under 120 words."""


def order_agent_node(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """Answer from the database, looping over tool calls until the model stops asking."""
    customer_id = state.get("customer_id", "")
    model = (llm or get_llm()).bind_tools(READ_TOOLS)

    conversation = [
        SystemMessage(content=SYSTEM_PROMPT.format(customer_id=customer_id)),
        HumanMessage(content=last_user_question(state)),
    ]
    new_messages: list = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = with_retry(model.invoke)(conversation)
        conversation.append(response)
        new_messages.append(response)

        if not getattr(response, "tool_calls", None):
            break

        for call in response.tool_calls:
            tool = TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                output = f"Unknown tool {call['name']}."
            else:
                # The customer id comes from state, never from the model's arguments: a model
                # that guesses someone else's id must not be able to read their orders.
                args = dict(call["args"])
                if "customer_id" in tool.args:
                    args["customer_id"] = customer_id
                output = _run(tool, args)

            message = ToolMessage(content=str(output), tool_call_id=call["id"])
            conversation.append(message)
            new_messages.append(message)
    else:
        # Loop finished without break: the model kept asking for tools past the cap.
        note = AIMessage(content="I could not finish that lookup. Please contact support.")
        new_messages.append(note)

    answer = _final_text(new_messages)
    return {
        "messages": new_messages,
        "route": "finish",
        "step_count": 1,
        "last_agent_output": answer,
    }


def _run(tool, args: dict) -> str:
    """Invoke a tool, turning a validation failure into a message the model can recover from.

    Returning the error rather than raising lets the model correct its arguments on the next
    round instead of failing the whole request.
    """
    try:
        return str(tool.invoke(args))
    except Exception as exc:  # noqa: BLE001 — surfaced to the model on purpose
        return f"Tool error: {type(exc).__name__}: {exc}. Check the arguments and try again."


def _final_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.text.strip():
            return message.text.strip()
    return ""
