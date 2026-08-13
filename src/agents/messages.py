"""Helpers for turning graph state into a request a chat model will accept.

Some models reject a request whose final turn is an assistant message ("prefilling"). That
happens naturally here on the supervisor's second pass, when the previous turn was a
specialist's answer. Every prompt built in this project therefore ends with a user turn.
"""

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from src.agents.state import AgentState


def last_user_question(state: AgentState) -> str:
    """The most recent thing the customer actually asked."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def transcript(state: AgentState, closing_question: str) -> list[AnyMessage]:
    """What was asked and what has been answered, always ending with a user turn.

    Assistant answers are kept: seeing them is what lets the supervisor decide whether a
    second specialist is still needed, or whether the request is already satisfied.

    Tool plumbing is dropped — the tool-call messages and their raw results. The supervisor
    is bound to no tools, so handing it calls for tools it does not have adds tokens and
    noise to every routing decision, and some providers reject the combination outright.
    It needs to know *what* was answered, not *how* it was looked up.
    """
    kept = [
        message
        for message in state.get("messages", [])
        if isinstance(message, HumanMessage)
        or (isinstance(message, AIMessage) and message.text.strip())
    ]
    return [*kept, HumanMessage(content=closing_question)]
