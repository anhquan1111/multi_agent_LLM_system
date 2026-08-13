"""Walking skeleton graph: one node, one hardcoded answer.

Deliberately has no LLM call, no retrieval and no tools. Its only job is to prove that the
deploy pipeline works end to end — FastAPI can call the graph, Docker can serve FastAPI, and
Hugging Face can run the container. Intelligence arrives in later steps.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from src.agents.state import AgentState

SKELETON_ANSWER = (
    "This is the walking skeleton. The routing graph, policy retrieval and order lookups "
    "are not wired up yet — this endpoint exists to prove the deployment path works."
)


def responder_node(state: AgentState) -> dict:
    """Return a fixed answer. Replaced by the supervisor in step 5."""
    return {
        "messages": [AIMessage(content=SKELETON_ANSWER)],
        "route": "finish",
        "step_count": 1,
        "last_agent_output": SKELETON_ANSWER,
    }


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("responder", responder_node)
    builder.add_edge(START, "responder")
    builder.add_edge("responder", END)
    return builder.compile()
