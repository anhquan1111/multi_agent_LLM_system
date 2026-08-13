"""Helpers for turning graph state into a request a chat model will accept.

Some models reject a request whose final turn is an assistant message ("prefilling"). That
happens naturally here on the supervisor's second pass, when the previous turn was a
specialist's answer. Every prompt built in this project therefore ends with a user turn.
"""

from langchain_core.messages import AnyMessage, HumanMessage

from src.agents.state import AgentState


def last_user_question(state: AgentState) -> str:
    """The most recent thing the customer actually asked."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def transcript(state: AgentState, closing_question: str) -> list[AnyMessage]:
    """Conversation so far, always ending with a user turn.

    Assistant messages are kept so the supervisor can see what has already been answered,
    which is what lets it decide whether a second specialist is still needed.
    """
    return [*state.get("messages", []), HumanMessage(content=closing_question)]
