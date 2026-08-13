# AGENTS.md

Rules for AI coding agents working in this repo. Read this before touching code.

## Project

Multi-agent e-commerce support assistant. A supervisor routes each question to one of three
specialists: `policy_agent` (RAG over policy docs), `order_agent` (read-only SQLite lookups), and
`action_agent` (the only component allowed to write, gated behind human confirmation).

Full spec: `docs/spec.md`. Do not expand scope beyond it without asking.

## Stack (verified 13/08/2026, do not guess these)

| Component | Version |
|---|---|
| Python | 3.12.4 |
| langgraph | 1.2.11 |
| langchain-core | 1.5.4 |
| langchain-google-genai | 4.3.3 |
| langgraph-checkpoint-sqlite | 3.1.1 |
| chromadb / langchain-chroma | 1.5.9 / 1.1.0 |
| fastapi / uvicorn | 0.141.1 / 0.52.1 |
| pydantic | 2.13.4 |
| pytest / ruff | 9.1.1 / 0.16.2 |

Models: `gemini-3.6-flash` (LLM), `gemini-embedding-2` truncated to 768 dims (embeddings).
Google SDK is `google-genai` (import `from google import genai`), **not** the retired
`google-generativeai`.

## Commands

```bash
uv sync --extra dev                  # install; never use pip directly
uv run pytest                        # full suite, LLM always mocked
uv run pytest -m integration         # API-level tests
uv run ruff check . --fix
uv run ruff format .
uv run python scripts/check_key.py   # verify GOOGLE_API_KEY + list real model names
uv run python scripts/seed_data.py   # regenerate data/shop.db
uv run python scripts/ingest.py      # rebuild data/chroma from data/policies/
uv run uvicorn src.api.main:app --reload --port 8000
docker build -t masys . && docker run -p 7860:7860 --env-file .env masys
```

## Layout

```
src/agents/   supervisor, the three specialist agents, graph + state definitions
src/rag/      chunking, embeddings, Chroma read/write
src/tools/    plain Python functions exposed to agents (read tools and write tools live in
              SEPARATE modules — see "Non-obvious patterns")
src/api/      FastAPI app, request/response Pydantic schemas
scripts/      one-off runnable scripts (seed, ingest, eval, key check)
tests/        pytest
data/policies/  source markdown — committed
data/shop.db, data/chroma/  generated artifacts — gitignored, never hand-edit
```

`src/` must not import FastAPI anywhere except `src/api/`. The graph has to stay callable from a
plain Python script.

## Code style

Type-hint everything. Pydantic models for tool arguments and API schemas. Docstrings only where
the intent is not obvious from the signature. No bare `except:`.

```python
from pydantic import BaseModel, Field


class GetOrderArgs(BaseModel):
    order_id: int = Field(description="Numeric order id, e.g. 1042")


def get_order(args: GetOrderArgs) -> dict[str, Any] | None:
    """Return one order row, or None when the id does not exist.

    Returning None rather than raising lets the agent tell the customer the order
    was not found instead of failing the whole request.
    """
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, status, total, placed_at FROM orders WHERE id = ?",
            (args.order_id,),
        ).fetchone()
    return dict(row) if row else None
```

SQL is always parameterised and always written by hand. The LLM picks which function to call and
fills its arguments; it never authors SQL.

## Testing

- Every test mocks the LLM. A test that hits the real Gemini API is a bug — it is slow, costs
  money, and is not deterministic. Mark the rare exception with `@pytest.mark.live`; CI skips it.
- Test tools and retrieval as plain functions before testing them through the graph.
- At least one `@pytest.mark.integration` test asserts the `/chat` response schema. That schema is
  a contract — if you change it, update `docs/spec.md` in the same commit.

## Permission tiers

> **REVIEW THIS SECTION YOURSELF** — these are the rules you are delegating, so make sure you
> agree with them before trusting them.

✅ **Do without asking**
- Write and edit code, add tests, add logging
- Run tests, ruff, the dev server, the scripts above
- Rebuild generated artifacts (`data/shop.db`, `data/chroma/`)
- Commit working increments using Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`,
  `test:`, `refactor:`)

⚠️ **Ask first**
- Changing the state schema or the `/chat` / `/confirm` response shape
- Adding or removing a dependency
- Moving or renaming directories
- Changing the embedding model or its dimension (invalidates the whole index)
- Anything not covered by `docs/spec.md`

🚫 **Never**
- Commit `.env` or any real API key; never print a key to logs or terminal output
- `git push --force`, rewrite pushed history, or delete branches
- Give a read-only agent a write tool (see below)
- Delete `data/policies/` or any file under `src/` without being asked
- Add a text-to-SQL path that lets the LLM author raw SQL

## Non-obvious patterns — do not "clean these up"

1. **LangGraph is 1.x, not 0.x.** Most tutorials, blog posts, and model priors target 0.x. Import
   paths, `interrupt()`, and checkpointer setup all differ. Read the installed source in `.venv`
   before writing graph code; do not write from memory.

2. **`Annotated[list, add]` on state fields is deliberate.** Without the reducer, LangGraph
   *replaces* the field when a node returns it, so a second node silently wipes the first node's
   `citations`. Fields like `route` are deliberately left un-annotated because only the latest
   value matters. Both choices are intentional.

3. **Read-only agents do not receive write tools — structurally, not by prompt.** Do not create a
   shared `ALL_TOOLS` list "for consistency". `policy_agent` and `order_agent` must never have
   `cancel_order` or `create_ticket` in their bound tool list. This is the project's main safety
   boundary; a prompt instruction is not an acceptable substitute.

4. **A model appearing in `models.list()` does not mean it works.** `gemini-2.5-flash` is still
   listed by the API but returns 404 on a real call. Always smoke-test a model with an actual
   request before switching to it.

5. **Windows console is cp1252.** Scripts that print emoji or Vietnamese text need
   `sys.stdout.reconfigure(encoding="utf-8")` or they crash with `UnicodeEncodeError`.

6. **`pydantic-settings` reads `.env` directly and never populates `os.environ`.** Application
   code must always use `settings.*`, never `os.getenv`. But third-party libraries that read
   `os.environ` themselves see nothing: verified — `os.getenv("LLM_MODEL")` is `None` while
   `settings.llm_model` is set. Pass credentials explicitly where the library accepts a parameter
   (`ChatGoogleGenerativeAI(google_api_key=settings.google_api_key)`). LangSmith accepts no
   parameter — it hooks in implicitly — so enabling tracing requires `load_dotenv()` at startup or
   real environment variables.

7. **Never put a trailing comment on a value line in `.env`.** Two parsers disagree: python-dotenv
   (used locally by pydantic-settings) strips `VALUE  # note`, but Docker's `--env-file` keeps it,
   so the value becomes `"8  # loop cap"`. This crashed the container on first run while the app
   worked perfectly locally. Put comments on their own line.

8. **Docker must bind `0.0.0.0`, port `7860`.** Hugging Face Spaces requires 7860, and binding
   `127.0.0.1` inside a container makes the service unreachable from outside while still working
   locally.

9. **Embeddings are truncated to 768 dims** from a native 3072. Changing this silently invalidates
   every stored vector — the index must be rebuilt, or retrieval returns nonsense.

10. **`data/chroma/` and `data/shop.db` are build outputs.** Regenerate them with the scripts; never
   hand-edit them and never commit them. `data/policies/*.md` is the source of truth.

## Keeping this file current

When an architectural decision changes (vector store, model, state schema, a new agent), update
this file in the same commit. A stale AGENTS.md is worse than none — it teaches the next session
to act on assumptions that are no longer true.
