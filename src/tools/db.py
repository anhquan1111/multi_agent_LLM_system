"""Shared SQLite access.

One place that opens connections, so no tool module invents its own convention.
"""

import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.config import settings

SEED_SCRIPT = Path("scripts/seed_data.py")


def ensure_database() -> bool:
    """Build the shop database if it is missing. Returns whether it exists afterwards.

    `data/shop.db` is a generated artifact: gitignored, and not baked into the image. Without
    this the deployed service starts happily and then fails every order lookup, which the
    model reports as "a technical issue with the database" — a symptom several layers away
    from the cause. Same reasoning as `ensure_index` for the vector store.
    """
    if settings.sqlite_path.exists():
        return True
    if not SEED_SCRIPT.exists():
        return False
    # The target is passed through the environment rather than inherited: the child process
    # builds its own Settings, and a real environment variable outranks the .env file.
    subprocess.run(
        [sys.executable, str(SEED_SCRIPT)],
        check=True,
        env={**os.environ, "SQLITE_PATH": str(settings.sqlite_path)},
    )
    return settings.sqlite_path.exists()


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
