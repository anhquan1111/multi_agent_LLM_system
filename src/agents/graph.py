"""Graph assembly.

Step 4: a single specialist, `policy_agent`. The supervisor and the order/action agents
arrive in step 5, at which point START points at the supervisor instead.
"""

from langgraph.graph import END, START, StateGraph

from src.agents.policy_agent import policy_agent_node
from src.agents.state import AgentState


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("policy_agent", policy_agent_node)
    builder.add_edge(START, "policy_agent")
    builder.add_edge("policy_agent", END)
    return builder.compile()
