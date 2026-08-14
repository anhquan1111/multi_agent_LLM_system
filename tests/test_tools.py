"""Tools tested as plain functions against a throwaway copy of the database."""

import shutil
from pathlib import Path

import pytest

from src.tools import db
from src.tools.read_tools import get_order, list_orders_for_customer
from src.tools.write_tools import cancel_order, create_ticket

SHOP_DB = Path("data/shop.db")


@pytest.fixture
def temp_db(tmp_path, monkeypatch, shop_db):
    """Each test gets its own copy, so write tests cannot affect anything else."""
    copy = tmp_path / "shop.db"
    shutil.copy(shop_db, copy)
    monkeypatch.setattr(db.settings, "sqlite_path", copy)
    return copy


def one_order(status: str) -> dict:
    return db.query_all(
        "SELECT id, customer_id FROM orders WHERE status = ? ORDER BY id LIMIT 1", (status,)
    )[0]


def test_get_order_returns_the_order_with_its_items(temp_db):
    order = one_order("delivered")

    result = get_order.invoke({"order_id": order["id"], "customer_id": order["customer_id"]})

    assert result["id"] == order["id"]
    assert isinstance(result["items"], list)


def test_a_customer_cannot_read_another_customers_order(temp_db):
    """The customer filter is the only thing standing between two customers' data."""
    order = one_order("delivered")

    result = get_order.invoke({"order_id": order["id"], "customer_id": "C999"})

    assert isinstance(result, str)
    assert "No order" in result


def test_listing_orders_only_returns_that_customers_orders(temp_db):
    order = one_order("delivered")

    rows = list_orders_for_customer.invoke({"customer_id": order["customer_id"]})

    assert rows
    ids = [r["id"] for r in rows]
    owned = {
        r["id"]
        for r in db.query_all(
            "SELECT id FROM orders WHERE customer_id = ?", (order["customer_id"],)
        )
    }
    assert set(ids) <= owned


def test_cancelling_a_processing_order_updates_the_row(temp_db):
    order = one_order("processing")

    message = cancel_order.invoke({"order_id": order["id"], "customer_id": order["customer_id"]})

    assert "cancelled" in message
    assert db.query_one("SELECT status FROM orders WHERE id = ?", (order["id"],))["status"] == (
        "cancelled"
    )


def test_a_shipped_order_cannot_be_cancelled(temp_db):
    """shipping.md allows cancellation only while processing — enforced in code, not by prompt."""
    order = one_order("shipped")

    message = cancel_order.invoke({"order_id": order["id"], "customer_id": order["customer_id"]})

    assert "no longer be cancelled" in message
    assert db.query_one("SELECT status FROM orders WHERE id = ?", (order["id"],))["status"] == (
        "shipped"
    )


def test_cancelling_someone_elses_order_changes_nothing(temp_db):
    order = one_order("processing")

    message = cancel_order.invoke({"order_id": order["id"], "customer_id": "C999"})

    assert "Nothing changed" in message
    assert db.query_one("SELECT status FROM orders WHERE id = ?", (order["id"],))["status"] == (
        "processing"
    )


def test_creating_a_ticket_inserts_a_row(temp_db):
    order = one_order("delivered")

    create_ticket.invoke(
        {"order_id": order["id"], "customer_id": order["customer_id"], "reason": "item is faulty"}
    )

    rows = db.query_all("SELECT reason, status FROM tickets WHERE order_id = ?", (order["id"],))
    assert rows[0]["reason"] == "item is faulty"
    assert rows[0]["status"] == "open"


def test_read_and_write_tools_live_in_separate_modules():
    """The safety boundary is structural: no module exports a combined list."""
    from src.tools import read_tools, write_tools

    read_names = {t.name for t in read_tools.READ_TOOLS}
    write_names = {t.name for t in write_tools.WRITE_TOOLS}

    assert read_names.isdisjoint(write_names)
    assert not hasattr(read_tools, "WRITE_TOOLS")
    assert not hasattr(read_tools, "ALL_TOOLS")


def test_the_database_is_built_when_missing(tmp_path, monkeypatch):
    """A deployed container has no data/shop.db: it is gitignored and not in the image."""
    from src.tools.db import ensure_database

    monkeypatch.setattr(db.settings, "sqlite_path", tmp_path / "fresh.db")

    assert ensure_database() is True
    assert (tmp_path / "fresh.db").exists()
