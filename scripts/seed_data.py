"""Generate the synthetic shop database.

Deterministic: a fixed seed means the same rows every run, so the gold evaluation set
stays valid. Safe to re-run — it drops and rebuilds every table.

Run:  uv run python scripts/seed_data.py
"""

import random
import sqlite3
import sys
from datetime import date, timedelta

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402

SEED = 20260814
N_CUSTOMERS, N_PRODUCTS, N_ORDERS = 30, 60, 200
TODAY = date(2026, 8, 14)

TIERS = ["standard", "gold", "platinum"]
STATUSES = ["processing", "shipped", "delivered", "cancelled"]
CATEGORIES = {
    "laptops": ("Laptop", 699, 2499),
    "tablets": ("Tablet", 249, 1299),
    "audio": ("Headphones", 29, 449),
    "accessories": ("Cable", 9, 99),
}
FIRST = ["An", "Binh", "Chi", "Dung", "Giang", "Ha", "Khanh", "Lan", "Minh", "Nam"]
LAST = ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Vu"]

SCHEMA = """
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS tickets;

CREATE TABLE customers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    tier        TEXT NOT NULL,
    joined_at   TEXT NOT NULL
);

CREATE TABLE products (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    price       REAL NOT NULL,
    in_stock    INTEGER NOT NULL
);

CREATE TABLE orders (
    id           INTEGER PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customers(id),
    status       TEXT NOT NULL,
    total        REAL NOT NULL,
    placed_at    TEXT NOT NULL,
    delivered_at TEXT
);

CREATE TABLE order_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL,
    unit_price  REAL NOT NULL
);

CREATE TABLE tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    customer_id TEXT NOT NULL REFERENCES customers(id),
    reason      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_items_order ON order_items(order_id);
"""


def build() -> None:
    rng = random.Random(SEED)
    path = settings.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)

    customers = []
    for i in range(1, N_CUSTOMERS + 1):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        customers.append(
            (
                f"C{i:03d}",
                name,
                f"{name.lower().replace(' ', '.')}{i}@example.com",
                rng.choices(TIERS, weights=[6, 3, 1])[0],
                str(TODAY - timedelta(days=rng.randint(30, 1200))),
            )
        )
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    products = []
    for i in range(1, N_PRODUCTS + 1):
        category = rng.choice(list(CATEGORIES))
        label, low, high = CATEGORIES[category]
        products.append(
            (
                i,
                f"{label} Model {chr(65 + i % 26)}{i}",
                category,
                round(rng.uniform(low, high), 2),
                rng.randint(0, 40),
            )
        )
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    orders, items = [], []
    for order_id in range(1001, 1001 + N_ORDERS):
        customer_id = f"C{rng.randint(1, N_CUSTOMERS):03d}"
        placed = TODAY - timedelta(days=rng.randint(0, 180))
        status = rng.choices(STATUSES, weights=[2, 2, 6, 1])[0]

        total = 0.0
        for _ in range(rng.randint(1, 3)):
            pid, _, _, price, _ = products[rng.randrange(N_PRODUCTS)]
            qty = rng.randint(1, 2)
            items.append((order_id, pid, qty, price))
            total += price * qty

        delivered = None
        if status == "delivered":
            delivered = str(placed + timedelta(days=rng.randint(2, 7)))
        orders.append((order_id, customer_id, status, round(total, 2), str(placed), delivered))

    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders)
    conn.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?,?,?,?)",
        items,
    )
    conn.commit()

    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("customers", "products", "orders", "order_items", "tickets")
    }
    conn.close()

    print(f"✅ Built {path} (seed={SEED})")
    for table, n in counts.items():
        print(f"   {table:12} {n:>5}")


if __name__ == "__main__":
    build()
