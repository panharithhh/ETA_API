from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from database import get_db
from models.first_mile import ConfirmationAction, ConfirmationToken, Package, PackageEvent, PackageStatus

router = APIRouter(prefix="/confirmation", tags=["Warehouse Confirmation"])

# ── Minimal response pages ────────────────────────────────────────────────────

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; display: flex; align-items: center; justify-content: center;
            min-height: 100vh; background: #f4f4f5; font-family: Arial, sans-serif; }}
    .card {{ background: #fff; border-radius: 12px; padding: 48px 40px; text-align: center;
             box-shadow: 0 2px 12px rgba(0,0,0,.08); max-width: 440px; }}
    .icon {{ font-size: 48px; margin-bottom: 16px; }}
    h2 {{ margin: 0 0 12px; color: #1e293b; }}
    p  {{ color: #64748b; line-height: 1.6; margin: 0; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">{icon}</div>
    <h2>{title}</h2>
    <p>{message}</p>
  </div>
</body>
</html>"""


def _page(title: str, message: str, icon: str) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(title=title, message=message, icon=icon))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/confirm/{token}", include_in_schema=False)
def confirm_delivery(token: str, db: Session = Depends(get_db)) -> HTMLResponse:
    return _handle_token(token, ConfirmationAction.confirm, db)


@router.get("/reject/{token}", include_in_schema=False)
def reject_delivery(token: str, db: Session = Depends(get_db)) -> HTMLResponse:
    return _handle_token(token, ConfirmationAction.reject, db)


# ── Shared token handler ──────────────────────────────────────────────────────

def _handle_token(token: str, expected_action: ConfirmationAction, db: Session) -> HTMLResponse:
    row: ConfirmationToken | None = (
        db.query(ConfirmationToken).filter(ConfirmationToken.token == token).first()
    )

    if not row:
        return _page("Invalid Link", "This confirmation link is not valid or does not exist.", "❌")

    if row.used:
        return _page("Already Used", "This link has already been used. No further action is needed.", "⚠️")

    # expires_at stored as naive UTC
    if datetime.utcnow() > row.expires_at:
        return _page("Link Expired", "This confirmation link expired 48 hours after it was sent.", "⏰")

    if row.action != expected_action:
        # Prevent using a confirm token on the reject endpoint and vice-versa
        return _page("Invalid Link", "This link cannot be used for that action.", "❌")

    row.used = True

    package: Package = db.get(Package, row.package_id)

    if expected_action == ConfirmationAction.confirm:
        package.status = PackageStatus.confirmed
        db.add(PackageEvent(
            package_id=package.id,
            status=PackageStatus.confirmed,
            notes="Customer confirmed delivery via email link — package queued for routing",
        ))
        db.commit()
        return _page(
            "Delivery Confirmed",
            f"Package #{package.id} is confirmed. We will proceed with routing your delivery.",
            "✅",
        )
    else:
        package.status = PackageStatus.rejected
        db.add(PackageEvent(
            package_id=package.id,
            status=PackageStatus.rejected,
            notes="Customer rejected delivery via email link — routing halted",
        ))
        db.commit()
        return _page(
            "Delivery Rejected",
            f"Package #{package.id} has been rejected. Our team will follow up with you shortly.",
            "🚫",
        )
