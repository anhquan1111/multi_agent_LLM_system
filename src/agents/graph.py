"""Graph assembly: supervisor routing over three specialists.

Every specialist returns to the supervisor rather than to END, because one request can need
two of them — "cancel order 1042, and what is the refund policy?" is not answerable in a
single hop. The cost is one extra LLM call per hop, bounded by MAX_AGENT_STEPS.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agents.action_agent import action_agent_node
from src.agents.order_agent import order_agent_node
from src.agents.policy_agent import policy_agent_node
from src.agents.state import AgentState
from src.agents.supervisor import (
    detect_stuck,
    route_from_supervisor,
    stuck_result,
    supervisor_node,
)

SPECIALISTS = {
    "policy_agent": policy_agent_node,
    "order_agent": order_agent_node,
    "action_agent": action_agent_node,
}


def _guarded(node):
    """Wrap a specialist so a repeated answer ends the run instead of looping to the cap."""

    def wrapped(state: AgentState) -> dict:
        result = node(state)
        if detect_stuck(state, result.get("last_agent_output", "")):
            return stuck_result()
        return result

    wrapped.__name__ = getattr(node, "__name__", "specialist")
    return wrapped


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    for name, node in SPECIALISTS.items():
        builder.add_node(name, _guarded(node))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {**{name: name for name in SPECIALISTS}, "finish": END},
    )
    for name in SPECIALISTS:
        builder.add_edge(name, "supervisor")

    return builder.compile(checkpointer=checkpointer)


def open_checkpointer(path: Path) -> SqliteSaver:
    """A SQLite checkpointer that lives for the life of the process.

    `check_same_thread=False` because FastAPI serves requests on a thread pool, and a
    conversation resumed by /confirm is usually handled by a different thread than the one
    that created it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)
