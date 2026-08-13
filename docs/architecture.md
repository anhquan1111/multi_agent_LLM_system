# Architecture

Design decisions made before writing code. Companion to `docs/spec.md` (what) — this file covers
how. Verified against the installed LangGraph 1.2.11 API, not from memory.

## 1. Overview

```
                        POST /chat  ──┐
                        POST /confirm ─┤  src/api/main.py  (FastAPI — thin)
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │  graph.invoke(state)   │  src/agents/graph.py
                          └────────────────────────┘
                                       │
                                    START
                                       ▼
                              ┌─────────────────┐
                    ┌────────▶│   supervisor    │────── finish ──▶ END
                    │         └────────┬────────┘
                    │                  │ route
                    │      ┌───────────┼───────────┐
                    │      ▼           ▼           ▼
                    │ ┌─────────┐ ┌─────────┐ ┌──────────────┐
                    │ │ policy  │ │  order  │ │    action    │
                    │ │  agent  │ │  agent  │ │    agent     │
                    │ └────┬────┘ └────┬────┘ └──────┬───────┘
                    │      │           │             │ interrupt()
                    └──────┴───────────┴─────────────┘        │
                             back to supervisor           ┌───▼────┐
                                                          │ human  │
                                                          │approval│
                                                          └────────┘
```

Every specialist returns to the supervisor, which decides whether the question is fully answered or
another specialist is still needed. `"Can I cancel order 1042, and what's the refund policy?"`
legitimately needs two specialists — single-hop routing could not answer it.

The cost of that choice is one extra LLM call per hop, bounded by `MAX_AGENT_STEPS`.

## 2. State schema

The single most important design decision in the project. Getting the reducers wrong produces
silent data loss, not a crash.

```python
import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["policy_agent", "order_agent", "action_agent", "finish"]


class Citation(TypedDict):
    source: str  # "returns.md"
    chunk_id: str  # "returns.md#3"


class PendingAction(TypedDict):
    tool: str  # "cancel_order"
    args: dict[str, Any]  # {"order_id": 1042}
    description: str  # human-readable, shown to the user for approval


class AgentState(TypedDict):
    # --- accumulated across nodes (reducer required) ---
    messages: Annotated[list[AnyMessage], add_messages]
    citations: Annotated[list[Citation], operator.add]
    step_count: Annotated[int, operator.add]

    # --- overwritten deliberately (latest value is the only one that matters) ---
    customer_id: str
    route: Route
    pending_action: PendingAction | None
    last_agent_output: str
    error: str | None
```

### Why each reducer is what it is

| Field | Reducer | Reason |
|---|---|---|
| `messages` | `add_messages` | LangGraph's own reducer: appends, and de-duplicates/updates by message id. Never use plain `operator.add` here — it breaks tool-call message pairing. |
| `citations` | `operator.add` | Two specialists in one request must both keep their sources. Without the reducer the second node silently erases the first node's citations. |
| `step_count` | `operator.add` | Each node returns `{"step_count": 1}`; the reducer sums them. This is what makes the loop cap work — a plain assignment would let a node reset the counter. |
| `route` | none | Only the newest routing decision matters. Accumulating it would be wrong. |
| `pending_action` | none | Either there is one pending action or there is none. |
| `last_agent_output` | none | Stuck detection compares the current node's output to the previous one. Only the previous value is needed. |
| `error` | none | Latest error wins; the run terminates anyway. |

The un-annotated fields are **deliberately** un-annotated. This asymmetry is documented in
`AGENTS.md` so nobody "fixes" it for consistency.

## 3. Nodes

| Node | Reads from state | Writes to state | LLM call |
|---|---|---|---|
| `supervisor` | `messages`, `step_count`, `last_agent_output` | `route`, `step_count`, `error` | 1, structured output constrained to `Route` |
| `policy_agent` | `messages` | `messages`, `citations`, `step_count`, `last_agent_output` | 1 (+ retrieval, no LLM) |
| `order_agent` | `messages`, `customer_id` | `messages`, `step_count`, `last_agent_output` | 1–2 (tool call, then phrasing) |
| `action_agent` | `messages`, `customer_id` | `messages`, `pending_action`, `step_count` | 1 (tool selection), then `interrupt()` |

`supervisor` never touches data and never writes prose. It emits one label. Constraining it to a
`Literal` via structured output removes a whole class of parsing bugs.

## 4. Routing

```python
from langgraph.graph import END, START, StateGraph

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("policy_agent", policy_agent_node)
builder.add_node("order_agent", order_agent_node)
builder.add_node("action_agent", action_agent_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,  # pure function of state, no LLM call
    {
        "policy_agent": "policy_agent",
        "order_agent": "order_agent",
        "action_agent": "action_agent",
        "finish": END,
    },
)
for specialist in ("policy_agent", "order_agent", "action_agent"):
    builder.add_edge(specialist, "supervisor")
```

`route_from_supervisor` is a plain function reading `state["route"]` — the LLM call already happened
inside `supervisor_node`. Keeping the decision and the dispatch separate makes the routing logic
unit-testable without any mocking.

Guard inside the same function:

```python
def route_from_supervisor(state: AgentState) -> str:
    if state.get("error"):
        return "finish"
    if state["step_count"] >= settings.max_agent_steps:
        return "finish"
    return state["route"]
```

## 5. RAG pipeline (`policy_agent`)

```
data/policies/*.md
   → MarkdownHeaderTextSplitter        keep section headings as metadata
   → RecursiveCharacterTextSplitter    chunk_size=800, chunk_overlap=100
   → gemini-embedding-2 @ 768 dims     batched at build time
   → Chroma (persisted to data/chroma) collection "policies"
   → similarity search, k=4
   → prompt with numbered chunks + "answer only from these; say you don't know otherwise"
   → answer + citations
```

**Chunk size 800 / overlap 100.** Policy documents are short, sectioned prose. 800 characters holds
roughly one complete clause with its heading; the 100-character overlap keeps a sentence split
across a boundary readable in both chunks.

**Two-stage splitting.** Splitting on markdown headings first means a chunk never merges text from
"Returns" and "Warranty" — the near-duplicate passages in those two documents are exactly what makes
retrieval hard here, and heading metadata is what keeps them apart.

**Ingestion runs at Docker build time**, not at container start. Hugging Face Spaces cold starts
would otherwise pay for embedding every document on every boot.

Deliberately not implemented, listed as future work in the README: reranking, hybrid BM25 + dense
retrieval, query rewriting, Corrective RAG.

## 6. Tools and permissions

```
src/tools/read_tools.py     get_order, list_orders_for_customer, get_product
src/tools/write_tools.py    cancel_order, create_ticket
```

Separate modules, and no module exports a combined list. Binding is explicit per agent:

```python
policy_agent_llm = llm  # no tools; retrieval is a plain call
order_agent_llm = llm.bind_tools(READ_TOOLS)
action_agent_llm = llm.bind_tools(WRITE_TOOLS)
```

| Tool | Module | SQL | Requires approval |
|---|---|---|---|
| `get_order(order_id)` | read | `SELECT ... WHERE id = ? AND customer_id = ?` | no |
| `list_orders_for_customer()` | read | `SELECT ... WHERE customer_id = ?` | no |
| `get_product(product_id)` | read | `SELECT ... WHERE id = ?` | no |
| `cancel_order(order_id)` | write | `UPDATE orders SET status='cancelled' WHERE ...` | **yes** |
| `create_ticket(order_id, reason)` | write | `INSERT INTO tickets ...` | **yes** |

Every read tool filters by `customer_id` from state, so one customer cannot read another's orders
even if the LLM is talked into asking. All SQL is hand-written and parameterised; the LLM only
chooses the function and fills typed arguments validated by Pydantic.

## 7. Human-in-the-loop

```python
from langgraph.types import interrupt


def action_agent_node(state: AgentState) -> dict:
    call = pick_write_tool(state)  # LLM decides tool + args
    approved = interrupt(
        {  # graph pauses here; state is checkpointed
            "tool": call.name,
            "args": call.args,
            "description": describe(call),
        }
    )
    if not approved:
        return {"messages": [AIMessage("Cancelled — nothing was changed.")], "step_count": 1}
    result = WRITE_TOOLS_BY_NAME[call.name].invoke(call.args)
    return {"messages": [ToolMessage(str(result), tool_call_id=call.id)], "step_count": 1}
```

`interrupt()` raises out of the node; the checkpointer stores everything. `POST /confirm` resumes
with `graph.invoke(Command(resume=True), config={"configurable": {"thread_id": ...}})`, and
execution continues from the line after `interrupt()`.

This is why `thread_id` is part of the API contract rather than an implementation detail: without
it there is nothing to resume.

## 8. Persistence

```python
from langgraph.checkpoint.sqlite import SqliteSaver

graph = builder.compile(checkpointer=SqliteSaver.from_conn_string(settings.checkpoint_path))
```

One SQLite file for checkpoints, separate from `data/shop.db`. Two reasons to keep them apart:
checkpoints are disposable session state while shop data is the domain, and on Hugging Face Spaces
the checkpoint file lives on ephemeral storage and is expected to vanish on restart.

## 9. Guardrails — where each one lives

| Guardrail | Implemented in |
|---|---|
| Loop cap | `route_from_supervisor`, reading `step_count` |
| Stuck detection | `supervisor_node`, comparing `last_agent_output` |
| Retry + backoff | `src/agents/llm.py`, `tenacity` wrapper on the LLM client, transient errors only |
| Tool argument validation | Pydantic models in `src/tools/`; failures return a message to the agent |
| Token ceiling | `src/agents/llm.py`, checked before each call |
| Read/write separation | Module split + per-agent `bind_tools` |
| Approval before writes | `interrupt()` in `action_agent_node` |

## 10. Request flows

**Policy question** — `START → supervisor(policy_agent) → policy_agent → supervisor(finish) → END`.
Two LLM calls for supervision plus one for the answer.

**Order question** — same shape via `order_agent`, which additionally makes a tool call.

**Write request** — `START → supervisor(action_agent) → action_agent → interrupt`. The HTTP response
carries `pending_action` and `thread_id`. On `POST /confirm`:
`resume → action_agent (after interrupt) → supervisor(finish) → END`.

## 11. Decisions deferred

- Streaming responses — the API returns complete answers only.
- Reranking and query rewriting — measured against the gold set first; added only if retrieval is
  the actual bottleneck.
- Postgres checkpointer — SQLite is sufficient for a single-container deployment.
- Authentication — `customer_id` is trusted from the request body, as stated in the spec.
