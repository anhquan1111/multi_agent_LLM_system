"""Request/response contracts for the HTTP layer.

These shapes are part of the spec (`docs/spec.md` section 3) and are asserted by an
integration test. Changing them means changing the spec in the same commit.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    customer_id: str = Field(default="C001", max_length=32)
    thread_id: str | None = Field(
        default=None,
        description="Omit to start a new conversation; the response returns the new id.",
    )


class ConfirmRequest(BaseModel):
    thread_id: str = Field(description="Thread id returned alongside a pending_action")
    approved: bool = Field(description="True to execute the pending action, false to discard it")


class Citation(BaseModel):
    source: str
    chunk_id: str


class PendingAction(BaseModel):
    tool: str
    args: dict[str, Any]
    description: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    citations: list[Citation] = []
    pending_action: PendingAction | None = None
    thread_id: str
    steps: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    llm_model: str
    google_api_key_configured: bool
