"""Shared graph state.

Read `docs/architecture.md` section 2 before changing anything here. Which fields carry a
reducer and which do not is deliberate: accumulated fields need one, latest-value-wins fields
must not have one.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["policy_agent", "order_agent", "action_agent", "finish"]


class Citation(TypedDict):
    source: str
    chunk_id: str


class PendingAction(TypedDict):
    tool: str
    args: dict[str, Any]
    description: str


class AgentState(TypedDict, total=False):
    # --- accumulated across nodes: reducer required ---
    messages: Annotated[list[AnyMessage], add_messages]
    citations: Annotated[list[Citation], operator.add]
    step_count: Annotated[int, operator.add]

    # --- latest value wins: deliberately no reducer ---
    customer_id: str
    route: Route
    pending_action: PendingAction | None
    last_agent_output: str
    error: str | None
