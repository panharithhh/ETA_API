import os
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from models.first_mile import ConfirmationAction, ConfirmationToken, Package
from services.email_service import send_warehouse_arrival_email

# ── Set BASE_URL in .env to your deployed API root (no trailing slash) ───────
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
_TOKEN_TTL_HOURS = 48


def trigger_warehouse_arrival_email(package: Package, db: Session) -> None:
    """
    Call this from your routing logic whenever a package status transitions
    to `arrived_at_warehouse`.

    What it does:
      1. Generates two single-use tokens (one confirm, one reject).
      2. Persists both to confirmation_tokens.
      3. Emails the customer with buttons that hit /confirm/<token> or /reject/<token>.

    Raises ValueError if package.customer_email is not set.
    Does NOT commit — caller owns the transaction so the status change and
    token inserts land in the same commit.
    """
    if not package.customer_email:
        raise ValueError(
            f"Package {package.id} has no customer_email — cannot send confirmation email"
        )

    expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=_TOKEN_TTL_HOURS)

    confirm_tok = secrets.token_urlsafe(32)
    reject_tok = secrets.token_urlsafe(32)

    db.add(ConfirmationToken(
        package_id=package.id,
        token=confirm_tok,
        action=ConfirmationAction.confirm,
        expires_at=expires_at,
        used=False,
    ))
    db.add(ConfirmationToken(
        package_id=package.id,
        token=reject_tok,
        action=ConfirmationAction.reject,
        expires_at=expires_at,
        used=False,
    ))
    db.flush()  # write tokens before handing URLs to the email sender

    send_warehouse_arrival_email(
        to_email=package.customer_email,
        package_id=package.id,
        confirm_url=f"{BASE_URL}/api/v1/confirmation/confirm/{confirm_tok}",
        reject_url=f"{BASE_URL}/api/v1/confirmation/reject/{reject_tok}",
    )
