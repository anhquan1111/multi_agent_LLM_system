"""FastAPI layer.

Intentionally thin: parse the request, call the graph, shape the response. No business logic
lives here, so the graph stays callable from a plain script or a future CLI.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agents.graph import build_graph, open_checkpointer
from src.agents.state import new_turn
from src.api.schemas import ChatRequest, ChatResponse, ConfirmRequest, HealthResponse
from src.config import settings

VERSION = "0.3.0"
POLICIES = Path("data/policies")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Build the vector index once at boot if it is missing.

    Deliberately not fatal: a missing index makes policy answers unavailable, but /health
    must still respond so the failure is diagnosable instead of showing a dead container.
    """
    from src.observability import enable_tracing

    logger.info("LangSmith tracing: %s", "on" if enable_tracing() else "off")

    try:
        from src.tools.db import ensure_database

        logger.info("Shop database ready: %s", ensure_database())
    except Exception:
        logger.exception("Could not build the shop database; order lookups will fail")

    try:
        from src.rag.store import ensure_index

        logger.info("Vector index ready: %d chunks", ensure_index(POLICIES))
    except Exception:
        logger.exception("Could not build the vector index; policy answers will be unavailable")
    yield


app = FastAPI(
    title="Multi-Agent E-Commerce Support Assistant",
    version=VERSION,
    description="Supervisor-routed support assistant over shop policies and order data.",
    lifespan=lifespan,
)

# Built once at import time, not per request: compiling the graph on every call would add
# latency for no benefit. The checkpointer is what makes /confirm able to resume a paused run.
graph = build_graph(checkpointer=open_checkpointer(settings.checkpoint_path))


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _answer_for_turn(messages: list) -> str:
    """Every answer produced since the customer last spoke, oldest first.

    A two-part question is answered by two specialists in two hops, so taking only the last
    message would silently drop half of what was asked for. Stopping at the previous
    `HumanMessage` keeps earlier turns of the conversation out of this response.

    `.text`, not `.content`: models that return reasoning signatures put a list of blocks in
    `.content`, and that would fail ChatResponse validation with a 500.
    """
    answers: list[str] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.text.strip():
            answers.append(message.text.strip())
    return "\n\n".join(reversed(answers))


def _to_response(result: dict[str, Any], thread_id: str) -> ChatResponse:
    """Shape a graph result, whether it ran to completion or paused for approval.

    A paused run returns only `__interrupt__` — none of the state fields are present — so the
    two cases cannot share one code path.
    """
    pending = result.get("__interrupt__")
    if pending:
        proposed = pending[0].value
        return ChatResponse(
            answer=f"{proposed['description']} Please confirm.",
            route="action_agent",
            pending_action=proposed,
            thread_id=thread_id,
            steps=0,
        )

    return ChatResponse(
        answer=_answer_for_turn(result.get("messages", [])),
        route=result.get("route", "finish"),
        citations=result.get("citations", []),
        pending_action=None,
        thread_id=thread_id,
        steps=result.get("step_count", 0),
    )


UI_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """The demo page. One static file, no build step and no runtime dependency.

    Swagger at /docs shows the API; this shows the product. The approval flow is the reason
    it exists — on /docs it looks like copying an id between two endpoints.
    """
    return FileResponse(UI_PAGE)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Reports whether the API key reached the process, never its value."""
    return HealthResponse(
        status="ok",
        version=VERSION,
        llm_model=settings.llm_model,
        google_api_key_configured=settings.has_api_key,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or str(uuid4())

    result = graph.invoke(
        new_turn(HumanMessage(content=request.message), request.customer_id),
        config=_config(thread_id),
    )
    return _to_response(result, thread_id)


@app.post("/confirm", response_model=ChatResponse)
def confirm(request: ConfirmRequest) -> ChatResponse:
    """Approve or decline a pending write, resuming the paused run.

    Resuming a thread that is not waiting on anything is a client mistake, not a server
    error, so it answers 409 rather than pretending something happened.
    """
    config = _config(request.thread_id)

    if not graph.get_state(config).next:
        raise HTTPException(status_code=409, detail="No pending action for this thread_id.")

    result = graph.invoke(Command(resume=request.approved), config=config)
    return _to_response(result, request.thread_id)
