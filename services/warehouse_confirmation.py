import os
import secrets
from datetime import datetime, timedelta, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.first_mile import CONFIRMATION_TOKENS, ConfirmationAction
from services.email_service import send_warehouse_arrival_email

# ── Set BASE_URL in .env to your deployed API root (no trailing slash) ───────
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
_TOKEN_TTL_HOURS = 48


async def trigger_warehouse_arrival_email(package: dict, db: AsyncIOMotorDatabase) -> None:
    """
    Call this whenever a package status transitions to a destination warehouse.

    What it does:
      1. Generates two single-use tokens (one confirm, one reject).
      2. Persists both to the confirmation_tokens collection.
      3. Emails the customer with buttons that hit /confirm/<token> or /reject/<token>.

    Raises ValueError if package["customer_email"] is not set.
    """
    if not package.get("customer_email"):
        raise ValueError(
            f"Package {package['_id']} has no customer_email — cannot send confirmation email"
        )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    expires_at = now + timedelta(hours=_TOKEN_TTL_HOURS)

    confirm_tok = secrets.token_urlsafe(32)
    reject_tok = secrets.token_urlsafe(32)

    await db[CONFIRMATION_TOKENS].insert_many([
        {
            "package_id": package["_id"],
            "token": confirm_tok,
            "action": ConfirmationAction.confirm.value,
            "expires_at": expires_at,
            "used": False,
            "created_at": now,
        },
        {
            "package_id": package["_id"],
            "token": reject_tok,
            "action": ConfirmationAction.reject.value,
            "expires_at": expires_at,
            "used": False,
            "created_at": now,
        },
    ])

    send_warehouse_arrival_email(
        to_email=package["customer_email"],
        package_id=str(package["_id"]),
        confirm_url=f"{BASE_URL}/api/v1/confirmation/confirm/{confirm_tok}",
        reject_url=f"{BASE_URL}/api/v1/confirmation/reject/{reject_tok}",
    )
