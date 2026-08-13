"""Graph assembly: supervisor routing over three specialists.

Every specialist returns to the supervisor rather than to END, because one request can need
two of them — "cancel order 1042, and what is the refund policy?" is not answerable in a
single hop. The cost is one extra LLM call per hop, bounded by MAX_AGENT_STEPS.

The write path is two nodes, not one: see `action_agent.py` for why splitting the proposal
from the approval is a correctness requirement rather than a style choice.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agents.action_agent import (
    confirm_action_node,
    propose_action_node,
    route_after_proposal,
)
from src.agents.order_agent import order_agent_node
from src.agents.policy_agent import policy_agent_node
from src.agents.state import AgentState
from src.agents.supervisor import (
    detect_stuck,
    route_from_supervisor,
    stuck_result,
    supervisor_node,
)

# Names the supervisor may route to. `action_agent` is the entry point of the write path.
SPECIALISTS = {
    "policy_agent": policy_agent_node,
    "order_agent": order_agent_node,
    "action_agent": propose_action_node,
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
    builder.add_node("confirm_action", confirm_action_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {**{name: name for name in SPECIALISTS}, "finish": END},
    )

    builder.add_edge("policy_agent", "supervisor")
    builder.add_edge("order_agent", "supervisor")

    # The write path forks: a proposal goes to approval, a plain answer skips it.
    builder.add_conditional_edges(
        "action_agent",
        route_after_proposal,
        {"confirm_action": "confirm_action", "supervisor": "supervisor"},
    )
    builder.add_edge("confirm_action", "supervisor")

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
