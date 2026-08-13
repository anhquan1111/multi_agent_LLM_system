"""Read-only tools for `order_agent`.

Every query is hand-written and parameterised, and every one filters by `customer_id`, so
one customer cannot read another's orders even if the model is talked into asking. The LLM
chooses which function to call and fills typed arguments; it never authors SQL.

This module must never gain a write statement. Writes live in `write_tools.py` and are bound
to a different agent — see AGENTS.md pattern #3.
"""

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.tools.db import query_all, query_one

MAX_ROWS = 20


class GetOrderArgs(BaseModel):
    order_id: int = Field(description="Numeric order id, for example 1042")
    customer_id: str = Field(description="Id of the customer asking, for example C007")


class ListOrdersArgs(BaseModel):
    customer_id: str = Field(description="Id of the customer asking, for example C007")


class GetProductArgs(BaseModel):
    product_id: int = Field(description="Numeric product id")


@tool(args_schema=GetOrderArgs)
def get_order(order_id: int, customer_id: str) -> dict[str, Any] | str:
    """Look up one order belonging to this customer, including the items on it."""
    order = query_one(
        """
        SELECT id, status, total, placed_at, delivered_at
        FROM orders
        WHERE id = ? AND customer_id = ?
        """,
        (order_id, customer_id),
    )
    if order is None:
        return f"No order {order_id} was found for customer {customer_id}."

    order["items"] = query_all(
        """
        SELECT p.name, p.category, oi.quantity, oi.unit_price
        FROM order_items oi JOIN products p ON p.id = oi.product_id
        WHERE oi.order_id = ?
        """,
        (order_id,),
    )
    return order


@tool(args_schema=ListOrdersArgs)
def list_orders_for_customer(customer_id: str) -> list[dict[str, Any]] | str:
    """List this customer's most recent orders, newest first."""
    orders = query_all(
        """
        SELECT id, status, total, placed_at
        FROM orders
        WHERE customer_id = ?
        ORDER BY placed_at DESC
        LIMIT ?
        """,
        (customer_id, MAX_ROWS),
    )
    return orders or f"Customer {customer_id} has no orders."


@tool(args_schema=GetProductArgs)
def get_product(product_id: int) -> dict[str, Any] | str:
    """Look up a product's name, category, price and stock level."""
    product = query_one(
        "SELECT id, name, category, price, in_stock FROM products WHERE id = ?",
        (product_id,),
    )
    return product or f"No product {product_id} was found."


READ_TOOLS = [get_order, list_orders_for_customer, get_product]
