"""FastAPI layer.

Intentionally thin: parse the request, call the graph, shape the response. No business logic
lives here, so the graph stays callable from a plain script or a future CLI.
"""

from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.messages import HumanMessage

from src.agents.graph import build_graph
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.config import settings

VERSION = "0.1.0"

app = FastAPI(
    title="Multi-Agent E-Commerce Support Assistant",
    version=VERSION,
    description="Supervisor-routed support assistant over shop policies and order data.",
)

# Built once at import time, not per request: compiling the graph on every call would add
# latency for no benefit.
graph = build_graph()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send the bare URL somewhere useful. Anyone opening the deployed link lands here."""
    return RedirectResponse(url="/docs")


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
        {
            "messages": [HumanMessage(content=request.message)],
            "customer_id": request.customer_id,
            "step_count": 0,
        }
    )

    return ChatResponse(
        answer=result["messages"][-1].content,
        route=result.get("route", "finish"),
        citations=result.get("citations", []),
        pending_action=result.get("pending_action"),
        thread_id=thread_id,
        steps=result.get("step_count", 0),
    )
