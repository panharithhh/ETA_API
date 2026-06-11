from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from db import get_db
from models.first_mile import (
    CONFIRMATION_TOKENS,
    PACKAGE_EVENTS,
    PACKAGES,
    ConfirmationAction,
    PackageStatus,
)

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
async def confirm_delivery(token: str, db: AsyncIOMotorDatabase = Depends(get_db)) -> HTMLResponse:
    return await _handle_token(token, ConfirmationAction.confirm, db)


@router.get("/reject/{token}", include_in_schema=False)
async def reject_delivery(token: str, db: AsyncIOMotorDatabase = Depends(get_db)) -> HTMLResponse:
    return await _handle_token(token, ConfirmationAction.reject, db)


# ── Shared token handler ──────────────────────────────────────────────────────

async def _handle_token(
    token: str, expected_action: ConfirmationAction, db: AsyncIOMotorDatabase
) -> HTMLResponse:
    row = await db[CONFIRMATION_TOKENS].find_one({"token": token})

    if not row:
        return _page("Invalid Link", "This confirmation link is not valid or does not exist.", "❌")

    if row.get("used"):
        return _page("Already Used", "This link has already been used. No further action is needed.", "⚠️")

    # expires_at stored as naive UTC
    if datetime.utcnow() > row["expires_at"]:
        return _page("Link Expired", "This confirmation link expired 48 hours after it was sent.", "⏰")

    if row["action"] != expected_action.value:
        # Prevent using a confirm token on the reject endpoint and vice-versa
        return _page("Invalid Link", "This link cannot be used for that action.", "❌")

    await db[CONFIRMATION_TOKENS].update_one({"_id": row["_id"]}, {"$set": {"used": True}})

    pkg_id = row["package_id"]
    now = datetime.utcnow()

    if expected_action == ConfirmationAction.confirm:
        await db[PACKAGES].update_one(
            {"_id": pkg_id}, {"$set": {"status": PackageStatus.confirmed.value}}
        )
        await db[PACKAGE_EVENTS].insert_one({
            "package_id": pkg_id,
            "status": PackageStatus.confirmed.value,
            "notes": "Customer confirmed delivery via email link — package queued for routing",
            "created_at": now,
        })
        return _page(
            "Delivery Confirmed",
            f"Package #{pkg_id} is confirmed. We will proceed with routing your delivery.",
            "✅",
        )
    else:
        await db[PACKAGES].update_one(
            {"_id": pkg_id}, {"$set": {"status": PackageStatus.rejected.value}}
        )
        await db[PACKAGE_EVENTS].insert_one({
            "package_id": pkg_id,
            "status": PackageStatus.rejected.value,
            "notes": "Customer rejected delivery via email link — routing halted",
            "created_at": now,
        })
        return _page(
            "Delivery Rejected",
            f"Package #{pkg_id} has been rejected. Our team will follow up with you shortly.",
            "🚫",
        )
