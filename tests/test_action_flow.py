"""The human-in-the-loop flow: /chat pauses, /confirm executes.

This is the behaviour that separates this project from a demo, so it is tested end to end
through the HTTP layer rather than by calling the node directly.
"""

import shutil
from pathlib import Path

import pytest

from src.tools import db
from tests.conftest import FakeLLM

SHOP_DB = Path("data/shop.db")


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    copy = tmp_path / "shop.db"
    shutil.copy(SHOP_DB, copy)
    monkeypatch.setattr(db.settings, "sqlite_path", copy)
    return copy


@pytest.fixture
def processing_order(temp_db) -> dict:
    return db.query_all(
        "SELECT id, customer_id FROM orders WHERE status = 'processing' ORDER BY id LIMIT 1", ()
    )[0]


@pytest.fixture
def cancelling_llm(monkeypatch, processing_order):
    """A model that routes to action_agent and asks to cancel the order."""
    llm = FakeLLM(
        route="action_agent",
        answer="",
        tool_calls=[
            {
                "name": "cancel_order",
                "args": {"order_id": processing_order["id"], "customer_id": "ignored-by-design"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    for target in ("src.agents.supervisor.get_llm", "src.agents.action_agent.get_llm"):
        monkeypatch.setattr(target, lambda: llm)
    return llm


def status_of(order_id: int) -> str:
    return db.query_one("SELECT status FROM orders WHERE id = ?", (order_id,))["status"]


@pytest.mark.integration
def test_chat_pauses_and_changes_nothing(client, cancelling_llm, processing_order):
    resp = client.post(
        "/chat",
        json={"message": "cancel that order", "customer_id": processing_order["customer_id"]},
    )

    body = resp.json()
    assert body["pending_action"]["tool"] == "cancel_order"
    assert body["thread_id"]
    assert status_of(processing_order["id"]) == "processing"  # nothing written yet


@pytest.mark.integration
def test_confirming_executes_the_write(client, cancelling_llm, processing_order):
    thread_id = client.post(
        "/chat",
        json={"message": "cancel that order", "customer_id": processing_order["customer_id"]},
    ).json()["thread_id"]

    resp = client.post("/confirm", json={"thread_id": thread_id, "approved": True})

    assert resp.status_code == 200
    assert status_of(processing_order["id"]) == "cancelled"
    assert resp.json()["pending_action"] is None


@pytest.mark.integration
def test_declining_leaves_the_data_untouched(client, cancelling_llm, processing_order):
    thread_id = client.post(
        "/chat",
        json={"message": "cancel that order", "customer_id": processing_order["customer_id"]},
    ).json()["thread_id"]

    resp = client.post("/confirm", json={"thread_id": thread_id, "approved": False})

    assert resp.status_code == 200
    assert status_of(processing_order["id"]) == "processing"


@pytest.mark.integration
def test_the_customer_id_comes_from_state_not_from_the_model(
    client, cancelling_llm, processing_order
):
    """The model was told to pass 'ignored-by-design'; the node must override it.

    If the model's value reached the tool, the cancel would fail the ownership check and the
    order would stay 'processing'.
    """
    thread_id = client.post(
        "/chat",
        json={"message": "cancel that order", "customer_id": processing_order["customer_id"]},
    ).json()["thread_id"]
    client.post("/confirm", json={"thread_id": thread_id, "approved": True})

    assert status_of(processing_order["id"]) == "cancelled"


@pytest.mark.integration
def test_confirming_an_unknown_thread_is_a_client_error(client):
    resp = client.post("/confirm", json={"thread_id": "never-existed", "approved": True})

    assert resp.status_code == 409


@pytest.mark.integration
def test_the_model_is_asked_once_not_twice(client, cancelling_llm, processing_order):
    """LangGraph re-runs an interrupted node from the top.

    With the model call and the interrupt in one node, approving would ask the model again
    and could execute a different action from the one shown to the customer. The proposal is
    therefore made in its own node and checkpointed before the pause.
    """
    thread_id = client.post(
        "/chat",
        json={"message": "cancel that order", "customer_id": processing_order["customer_id"]},
    ).json()["thread_id"]
    calls_after_chat = len(cancelling_llm.calls)

    client.post("/confirm", json={"thread_id": thread_id, "approved": True})

    assert calls_after_chat == 1
    assert len(cancelling_llm.calls) == 1, "confirming must not re-ask the model"


@pytest.mark.integration
def test_what_runs_is_what_was_shown(client, cancelling_llm, processing_order):
    """The executed action must match the description the customer approved."""
    body = client.post(
        "/chat",
        json={"message": "cancel that order", "customer_id": processing_order["customer_id"]},
    ).json()
    shown = body["pending_action"]

    client.post("/confirm", json={"thread_id": body["thread_id"], "approved": True})

    assert shown["tool"] == "cancel_order"
    assert shown["args"]["order_id"] == processing_order["id"]
    assert str(processing_order["id"]) in shown["description"]
    assert status_of(processing_order["id"]) == "cancelled"
