"""Write tools for `action_agent` — the only module in the project that mutates data.

Kept in its own module on purpose. There is no combined `ALL_TOOLS` list anywhere, so a
read-only agent cannot accidentally be handed one of these: `policy_agent` and `order_agent`
are never passed this list. That is the project's safety boundary, and it is structural
rather than a prompt instruction (AGENTS.md pattern #3).

Nothing here executes without human confirmation — `action_agent` calls `interrupt()` first.
These functions still validate independently, because a tool must be safe on its own terms
rather than trusting the caller to have asked.
"""

from datetime import date

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.tools.db import execute, query_one

CANCELLABLE_STATUSES = {"processing"}


class CancelOrderArgs(BaseModel):
    order_id: int = Field(description="Numeric order id to cancel")
    customer_id: str = Field(description="Id of the customer asking, for example C007")


class CreateTicketArgs(BaseModel):
    order_id: int = Field(description="Order the ticket is about")
    customer_id: str = Field(description="Id of the customer asking")
    reason: str = Field(min_length=3, max_length=500, description="Why the ticket is opened")


@tool(args_schema=CancelOrderArgs)
def cancel_order(order_id: int, customer_id: str) -> str:
    """Cancel an order. Only possible while the order is still being processed."""
    order = query_one(
        "SELECT status FROM orders WHERE id = ? AND customer_id = ?",
        (order_id, customer_id),
    )
    if order is None:
        return f"No order {order_id} was found for customer {customer_id}. Nothing changed."

    # Policy is enforced here, not left to the model: shipping.md says cancellation is only
    # allowed while the order is `processing`.
    if order["status"] not in CANCELLABLE_STATUSES:
        return (
            f"Order {order_id} is '{order['status']}', so it can no longer be cancelled. "
            "Once an order has shipped it must be returned instead. Nothing changed."
        )

    execute(
        "UPDATE orders SET status = 'cancelled' WHERE id = ? AND customer_id = ?",
        (order_id, customer_id),
    )
    return f"Order {order_id} has been cancelled and will be refunded in full."


@tool(args_schema=CreateTicketArgs)
def create_ticket(order_id: int, customer_id: str, reason: str) -> str:
    """Open a support ticket about an order."""
    order = query_one(
        "SELECT id FROM orders WHERE id = ? AND customer_id = ?",
        (order_id, customer_id),
    )
    if order is None:
        return f"No order {order_id} was found for customer {customer_id}. Nothing changed."

    execute(
        "INSERT INTO tickets (order_id, customer_id, reason, status, created_at) "
        "VALUES (?, ?, ?, 'open', ?)",
        (order_id, customer_id, reason, str(date.today())),
    )
    return (
        f"Support ticket opened for order {order_id}. Support will reply within two business days."
    )


WRITE_TOOLS = [cancel_order, create_ticket]
