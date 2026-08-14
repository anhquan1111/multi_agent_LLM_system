"""Shared graph state.

Read `docs/architecture.md` section 2 before changing anything here. Which fields carry a
reducer and which do not is deliberate: accumulated fields need one, latest-value-wins fields
must not have one.
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["policy_agent", "order_agent", "action_agent", "finish"]


def count_steps(current: int, update: int) -> int:
    """Sum the hops inside one request, and let the caller start a fresh turn at zero.

    Plain `operator.add` cannot express the reset: passing 0 adds 0 rather than replacing the
    value, so the count survived into the next message on the same thread and the third turn
    of any conversation hit the cap no matter how simple it was. Nodes only ever return 1; a
    literal 0 comes from the API opening a new turn.
    """
    return 0 if update == 0 else current + update


def collect_citations(current: list, update: list) -> list:
    """Gather sources within one request, and start a fresh turn empty.

    Same reason as `count_steps`: with plain `operator.add` the sources of an earlier message
    were still attached to the next one, so a cancellation request came back citing
    returns.md from three questions ago. A node with nothing to cite omits the field
    entirely, so an explicit empty list means the API is opening a new turn.
    """
    return [] if update == [] else current + update


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
    citations: Annotated[list[Citation], collect_citations]
    step_count: Annotated[int, count_steps]

    # --- latest value wins: deliberately no reducer ---
    customer_id: str
    route: Route
    pending_action: PendingAction | None
    last_agent_output: str
    error: str | None
