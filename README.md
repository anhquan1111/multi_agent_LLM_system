# Multi-Agent E-Commerce Support Assistant

![CI](https://github.com/anhquan1111/multi_agent_LLM_system/actions/workflows/ci.yml/badge.svg)

**▶ Live demo: https://multi-agent-llm-system.onrender.com**

A support assistant for an online shop, built on the observation that customers ask three
fundamentally different kinds of question — and that answering all of them with one mechanism is
how you end up with an assistant that confidently invents order totals.

A supervisor reads each request and routes it to the specialist equipped to answer it.

| You ask | Handled by | How it answers |
|---|---|---|
| *"How long do I have to return a damaged item?"* | `policy_agent` | RAG over shop documents, **with the source shown** |
| *"What is the status of order 1001?"* | `order_agent` | Hand-written parameterised SQL. **Never retrieval** — a similarity score is not an order total |
| *"Cancel order 1006"* | `action_agent` | Describes the change and **waits for your approval** before touching anything |
| *"Status of order 1001, and the return policy?"* | **both**, in two hops | Two specialists answer; both halves come back |

---

## What is worth looking at

**The read/write boundary is structural, not a prompt instruction.** `policy_agent` and
`order_agent` are never handed the write tools, so no phrasing reaches them — there is no button
for them to press. A prompt saying *"do not cancel orders"* can be argued with; a missing tool
cannot.

**Nothing is written without a human saying yes.** `action_agent` proposes a change, the graph
pauses on `interrupt()` and checkpoints, and the response carries what it intends to do.
`POST /confirm` resumes it. The proposal and the approval are **separate graph nodes**, because
LangGraph re-runs an interrupted node from the top — one node would ask the model again after
you clicked approve, and could then execute something other than what you were shown.

**The model never writes SQL.** It picks a function and fills typed arguments validated by
Pydantic. Every query filters by `customer_id` taken from server state, never from the model's
arguments, so a model that guesses someone else's id still reads nothing.

**Guardrails, and what each one catches.**

| Guardrail | Catches |
|---|---|
| Step cap, checked *before* the model call | Runaway loops — without paying for the call that would have caused one |
| A specialist is never routed to twice per turn | "No such order" read as a failure and retried until the cap |
| Stuck detection on repeated output | An agent restating itself, on the second hop instead of the eighth |
| Retry with exponential backoff | Rate limits and 5xx — and **nothing else**: retrying a malformed request bills you twice for the same error |
| Tool errors returned as text | A wrong argument becomes feedback the model corrects, instead of a failed request |

---

## Architecture

```
                        POST /chat   POST /confirm
                              │            │
                              ▼            ▼
                     ┌──────────────────────────┐
                     │   FastAPI — thin layer   │
                     └────────────┬─────────────┘
                                  │  graph.invoke()
                                  ▼
                        ┌───────────────────┐
              ┌────────▶│    supervisor     │────── finish ──▶ END
              │         └─────────┬─────────┘
              │      ┌────────────┼────────────┐
              │      ▼            ▼            ▼
              │ ┌─────────┐ ┌─────────┐ ┌──────────────┐
              │ │ policy  │ │  order  │ │    action    │
              │ │  agent  │ │  agent  │ │    agent     │
              │ │  Chroma │ │ SQLite  │ │   proposes   │
              │ └────┬────┘ └────┬────┘ └──────┬───────┘
              │      │           │             │ pending_action
              │      │           │             ▼
              │      │           │      ┌──────────────┐
              │      │           │      │   confirm    │──▶ interrupt()
              │      │           │      │   action     │    waits for a human
              │      │           │      └──────┬───────┘
              └──────┴───────────┴─────────────┘
                        back to the supervisor
```

Specialists return to the supervisor rather than to `END`, because one request can need two of
them. The cost is one extra model call per hop, bounded by `MAX_AGENT_STEPS`.

Design decisions and their trade-offs: [`docs/architecture.md`](docs/architecture.md) ·
Scope and definition of done: [`docs/spec.md`](docs/spec.md) ·
Conventions and hard-won gotchas: [`AGENTS.md`](AGENTS.md)

---

## Evaluation

Twenty gold cases across all three routes plus one out-of-scope question, each with an expected
route, expected source, and a fact the answer must contain.

| Metric | Result | Target |
|---|---|---|
| **Routing accuracy** | **14/14 (100%)** on the cases scored so far | ≥ 90% |
| Correct source cited | 14/14 | — |
| Answer contains the expected fact | 14/14 | — |

```bash
uv run python scripts/run_eval.py --routing-only   # 1 model call per case
uv run python scripts/run_eval.py                  # full: ~3 calls per case
uv run python scripts/run_eval.py --report         # last report, no calls
```

The route is **observed** by streaming node execution, not inferred from the output. An earlier
version inferred it from citations and scored a correct policy answer as a routing failure,
because the model happened not to quote a filename in that answer.

Free-tier quota is 20 model calls per day per model and a full pass costs about 60, so results
are appended per case and a run resumes where it stopped. `--routing-only` scores the
supervisor's decision alone, which fits inside one day and covers the headline metric.

---

## Stack

Python 3.12 · LangGraph 1.2 · Gemini `gemini-3.1-flash-lite` + `gemini-embedding-2` (768 dims) ·
Chroma · SQLite · FastAPI · Docker · GitHub Actions · optional LangSmith tracing

| Endpoint | Purpose |
|---|---|
| `GET /` | Demo page — one static file, no build step, no runtime dependency |
| `GET /health` | Liveness, plus whether the API key reached the process — never its value |
| `POST /chat` | Ask a question; returns the answer, route, citations and any pending action |
| `POST /confirm` | Approve or decline a pending write, resuming the paused run |
| `GET /docs` | Swagger UI |

---

## Run it

```bash
uv sync --extra dev
cp .env.example .env                 # paste a Google AI Studio key
uv run python scripts/check_key.py   # verifies the key and lists real model names
uv run uvicorn src.api.main:app --reload --port 8000
```

Then open http://127.0.0.1:8000. The shop database and the vector index are built on first
start — both are generated artifacts, so a fresh clone needs no data download.

With Docker:

```bash
docker build -t masys .
docker run -p 7860:7860 --env-file .env masys      # -e PORT=10000 to mimic Render
```

## Tests

```bash
uv run pytest        # 61 tests, every model and retrieval call mocked
uv run ruff check .
```

Tests never reach the network, enforced by an autouse fixture — a newly written test cannot call
the real API by accident, because it is opt-out rather than opt-in. CI additionally builds the
image and starts it with a deliberately fake key, since a green build does not mean a container
that boots.

---

## Limitations

Written out rather than left to be discovered. Each of these is a decision, not an oversight.

- **No authentication.** `customer_id` comes from the request body and is trusted. In production
  it would come from a signed token; accepting it in the body is precisely the vulnerability.
- **Pending approvals do not survive a restart.** The checkpointer writes to the container's own
  disk, which the free tier wipes on deploy and on waking from sleep. A persistent disk or a
  Postgres checkpointer fixes it; neither is free.
- **Synthetic data**, generated from a fixed seed so the gold set stays valid. A real deployment
  would connect to an existing database and run migrations, not seed one at startup.
- **No reranking, hybrid retrieval or query rewriting.** Deferred until the gold set shows
  retrieval is the bottleneck; at 14/14 on sources, it is not yet.
- **Six of twenty gold cases are unscored.** Finishing the set needs more daily quota than the
  free tier gives. Each recorded result names the model that produced it.
- **English only.** One shop, one currency, no localisation.

## Given more time

- **Corrective RAG** in `policy_agent`: grade the retrieved chunks, rewrite the query and search
  again when they come back thin. Costs one extra model call per policy question and adds a loop
  needing its own cap — worth measuring against the gold set before adopting.
- **Postgres checkpointer and a managed vector store**, so approvals and the index survive a
  restart.
- **Authentication**, so `customer_id` arrives in a signed token instead of the request body.
