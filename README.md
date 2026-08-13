# Multi-Agent E-Commerce Support Assistant

**Live demo:** https://multi-agent-llm-system.onrender.com

A customer-support assistant that has to answer two fundamentally different kinds of question —
and route each one to a specialist that is actually equipped to answer it.

- **Policy questions** ("How long do I have to return a damaged item?") are answered by a RAG agent
  over shop policy documents, with citations.
- **Account questions** ("Where is order 1042?") are answered by a read-only agent that runs
  hand-written, parameterised SQL. Never by retrieval — a similarity score is not an order total.
- **Write requests** ("Cancel my order") are handled by a separate agent that is the only component
  holding write tools, and every call stops for human confirmation before it executes.

The read/write split is structural, not a prompt instruction: read-only agents are never given the
write tools, so no amount of prompt injection reaches them.

> **Status: the multi-agent flow is live.** A supervisor routes each question to a policy RAG
> agent, a read-only order agent, or a write agent that pauses for your confirmation before it
> changes anything.
>
> The demo runs on a free instance that sleeps after 15 minutes of inactivity; the first request
> may take ~50 seconds to wake it. Hit `/health` first.

## Architecture

```
                          POST /chat  ──┐
                          POST /confirm ─┤  FastAPI (thin layer)
                                         ▼
                                ┌─────────────────┐
                      ┌────────▶│   supervisor    │────── finish ──▶ END
                      │         └────────┬────────┘
                      │      ┌───────────┼───────────┐
                      │      ▼           ▼           ▼
                      │ ┌─────────┐ ┌─────────┐ ┌──────────────┐
                      │ │ policy  │ │  order  │ │    action    │
                      │ │  agent  │ │  agent  │ │    agent     │
                      │ │  (RAG)  │ │  (SQL)  │ │  (WRITE) ⚠   │
                      │ └────┬────┘ └────┬────┘ └──────┬───────┘
                      └──────┴───────────┴─────────────┘
                                back to supervisor      └─▶ human approval
```

Design decisions and their trade-offs: [`docs/architecture.md`](docs/architecture.md).
Scope and definition of done: [`docs/spec.md`](docs/spec.md).

## Tech stack

Python 3.12 · LangGraph 1.2 · Gemini (`gemini-3.5-flash-lite` + `gemini-embedding-2`) · Chroma ·
SQLite · FastAPI · Docker · GitHub Actions

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness plus whether the API key reached the process (never its value) |
| `POST` | `/chat` | Ask a question; returns the answer, route, citations and any pending action |
| `POST` | `/confirm` | Approve or decline a pending write, resuming the paused run |
| `GET` | `/docs` | Interactive Swagger UI (the bare URL redirects here) |

## Run locally

```bash
uv sync --extra dev
cp .env.example .env          # then paste your Google AI Studio key
uv run python scripts/check_key.py
uv run uvicorn src.api.main:app --reload --port 8000
```

Then open http://127.0.0.1:8000/docs.

With Docker:

```bash
docker build -t masys .
docker run -p 7860:7860 --env-file .env masys   # or -e PORT=10000 to mimic Render
```

## Tests

```bash
uv run pytest              # LLM is always mocked: fast and deterministic
uv run ruff check .
```

## Evaluation

```bash
uv run python scripts/run_eval.py            # score the gold set, resumable
uv run python scripts/run_eval.py --report   # print the last report
```

Twenty cases across all three routes plus one out-of-scope question. The harness records which
specialist actually ran, which sources were cited, and whether the answer contains the expected
fact. Free-tier quota is 20 model calls per day per model and a full pass costs roughly 60, so
results are appended per case and a run can be resumed.

## Roadmap

- [x] Walking skeleton: one-node graph → FastAPI → Docker → live URL
- [x] RAG over policy documents with citations
- [x] Supervisor routing with a read-only order agent
- [x] Write agent behind `interrupt()` human approval
- [x] Guardrails: loop cap, stuck detection, retry with backoff
- [x] 20-question gold evaluation set with a resumable scoring script
- [x] CI on GitHub Actions
- [x] Optional LangSmith tracing

## Limitations

- **No authentication.** `customer_id` is taken from the request body and trusted. In production it
  would come from a signed token; accepting it in the body is precisely the vulnerability.
- Synthetic data, generated from a fixed seed so the evaluation set stays valid.
- No reranking, hybrid retrieval or query rewriting — deferred until the gold set shows retrieval
  is actually the bottleneck.
- **Pending approvals do not survive a restart.** The checkpointer writes to the container's
  own disk, which the free tier wipes on every deploy and every wake from sleep. If the demo
  sleeps between `/chat` and `/confirm`, the paused run is gone and `/confirm` answers 409.
  A persistent disk or a Postgres checkpointer would fix it; neither is free.
- Scoring the full gold set needs more daily quota than the free tier allows, so the published
  numbers are partial. Each result records the model that produced it.
- English only.
