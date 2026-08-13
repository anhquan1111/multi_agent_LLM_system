"""API-level tests. The response schema is a contract — these tests defend it."""

import pytest


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
