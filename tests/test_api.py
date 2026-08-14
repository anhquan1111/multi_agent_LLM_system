"""API-level tests. The response schema is a contract — these tests defend it."""

import pytest


def test_root_redirects_to_docs(client):
    """A bare URL is what people paste into a browser; it must not 404."""
    resp = client.get("/", follow_redirects=False)

    assert resp.status_code in (307, 308)
    assert resp.headers["location"] == "/docs"


def test_health_reports_ok(client):
    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["google_api_key_configured"], bool)


def test_health_reports_key_presence_without_revealing_it(client):
    """A health endpoint that echoes secrets is a security bug, not a feature."""
    body = client.get("/health").json()

    assert "google_api_key_configured" in body
    assert not any(isinstance(v, str) and v.startswith("AIza") for v in body.values())


@pytest.mark.integration
def test_chat_returns_the_agreed_schema(client):
    resp = client.post("/chat", json={"message": "what is the return policy?"})

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "answer",
        "route",
        "citations",
        "pending_action",
        "thread_id",
        "steps",
    }
    assert body["answer"]
    assert body["thread_id"]


@pytest.mark.integration
def test_chat_mints_a_thread_id_when_absent(client):
    first = client.post("/chat", json={"message": "hi"}).json()
    second = client.post("/chat", json={"message": "hi"}).json()

    assert first["thread_id"] != second["thread_id"]


@pytest.mark.integration
def test_chat_echoes_back_a_supplied_thread_id(client):
    resp = client.post("/chat", json={"message": "hi", "thread_id": "fixed-123"})

    assert resp.json()["thread_id"] == "fixed-123"


@pytest.mark.integration
def test_chat_rejects_an_empty_message(client):
    resp = client.post("/chat", json={"message": ""})

    assert resp.status_code == 422


def test_a_two_part_answer_keeps_both_halves():
    """Two specialists answering in two hops must both reach the customer."""
    from langchain_core.messages import AIMessage, HumanMessage

    from src.api.main import _answer_for_turn

    messages = [
        HumanMessage("earlier question"),
        AIMessage("earlier answer"),
        HumanMessage("status of order 1006, and the return policy?"),
        AIMessage("Order 1006 is processing, total $2,471.78."),
        AIMessage("Damaged items can be returned within 45 days [returns.md]."),
    ]

    answer = _answer_for_turn(messages)

    assert "Order 1006 is processing" in answer
    assert "45 days" in answer
    assert "earlier answer" not in answer, "previous turns must not leak into this response"
    assert answer.index("Order 1006") < answer.index("45 days"), "oldest first"


def test_tracing_stays_off_without_a_key(monkeypatch):
    """The deployed demo has no LangSmith key; it must run identically without one."""
    from src.config import settings
    from src.observability import enable_tracing

    monkeypatch.setattr(settings, "langchain_tracing_v2", True)
    monkeypatch.setattr(settings, "langchain_api_key", "   ")

    assert enable_tracing() is False


def test_tracing_copies_settings_into_environ(monkeypatch):
    """pydantic-settings never populates os.environ, and LangSmith reads only from there."""
    import os

    from src.config import settings
    from src.observability import enable_tracing

    monkeypatch.setattr(settings, "langchain_tracing_v2", True)
    monkeypatch.setattr(settings, "langchain_api_key", "ls-fake")
    monkeypatch.setattr(settings, "langchain_project", "test-project")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    assert enable_tracing() is True
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-fake"
    assert os.environ["LANGCHAIN_PROJECT"] == "test-project"


def test_the_default_model_matches_the_documented_one():
    """A host that sets no LLM_MODEL runs the default, so it must be the evaluated model.

    Render sets no LLM_MODEL and silently ran a different model from the one the gold set was
    scored against, which made the published accuracy describe a system nobody was using.
    """
    from src.config import Settings

    assert Settings.model_fields["llm_model"].default == "gemini-3.1-flash-lite"
