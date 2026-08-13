"""Shared SQLite access.

One place that opens connections, so no tool module invents its own convention.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from src.config import settings


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open the shop database with rows accessible by column name."""
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def query_one(sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def query_all(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def execute(sql: str, params: tuple[Any, ...]) -> int:
    """Run a write statement and return the number of affected rows."""
    with connect() as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.rowcount
