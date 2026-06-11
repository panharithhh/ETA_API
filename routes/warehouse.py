from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from db import get_db
from models.first_mile import (
    BRANCHES,
    PACKAGE_EVENTS,
    PACKAGES,
    PackageStatus,
    to_object_id,
)
from services.warehouse_confirmation import trigger_warehouse_arrival_email

try:
    from routing.osrm_client import get_route as _osrm_route
    _OSRM_AVAILABLE = True
except ImportError:
    _OSRM_AVAILABLE = False

router = APIRouter(prefix="/warehouse", tags=["Warehouse — Mid-Mile"])

# ── Valid incoming statuses for each transition ───────────────────────────────
_ARRIVE_FROM = {PackageStatus.at_origin_branch.value, PackageStatus.in_transit_w2w.value}
_DEPART_FROM = {PackageStatus.at_warehouse.value}


# ── Request bodies ────────────────────────────────────────────────────────────

class ArriveBody(BaseModel):
    branch_id: str

class DepartBody(BaseModel):
    to_branch_id: Optional[str] = None   # None → final leg (out_for_delivery)


# ── W2W travel time helper ────────────────────────────────────────────────────

def _w2w_minutes(origin: dict, dest: dict) -> float:
    """
    Real road distance from OSRM → minutes = (distance_km / 25) × 60.
    Falls back to haversine on OSRM error so a network blip never blocks dispatch.
    """
    if _OSRM_AVAILABLE:
        try:
            route = _osrm_route([(origin["lat"], origin["lng"]), (dest["lat"], dest["lng"])])
            distance_km = route["distance"] / 1000
            return (distance_km / 25) * 60
        except Exception:
            pass

    # Haversine fallback
    from math import radians, sin, cos, sqrt, asin
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [origin["lat"], origin["lng"], dest["lat"], dest["lng"]])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    distance_km = 2 * R * asin(sqrt(a))
    return (distance_km / 25) * 60


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/{package_id}/arrive")
async def arrive(package_id: str, body: ArriveBody, db: AsyncIOMotorDatabase = Depends(get_db)):
    pkg_oid = to_object_id(package_id)
    package = await db[PACKAGES].find_one({"_id": pkg_oid}) if pkg_oid else None
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")

    if package["status"] not in _ARRIVE_FROM:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot arrive: package is '{package['status']}', expected one of {sorted(_ARRIVE_FROM)}",
        )

    branch_oid = to_object_id(body.branch_id)
    branch = await db[BRANCHES].find_one({"_id": branch_oid}) if branch_oid else None
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch {body.branch_id} not found")

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    await db[PACKAGES].update_one(
        {"_id": pkg_oid},
        {"$set": {
            "status": PackageStatus.at_warehouse.value,
            "current_branch_id": branch_oid,
            "warehouse_arrived_at": now,
        }},
    )
    await db[PACKAGE_EVENTS].insert_one({
        "package_id": pkg_oid,
        "status": PackageStatus.at_warehouse.value,
        "branch_id": branch_oid,
        "notes": f"Arrived at {branch['name']}",
        "created_at": now,
    })

    email_sent = False
    is_destination = (
        package.get("destination_branch_id") is not None
        and package["destination_branch_id"] == branch_oid
    )
    if is_destination and package.get("customer_email"):
        # Re-read so the email service sees the updated status
        package = await db[PACKAGES].find_one({"_id": pkg_oid})
        try:
            await trigger_warehouse_arrival_email(package, db)
            email_sent = True
        except Exception as exc:
            # Email failure must not undo the warehouse arrival
            await db[PACKAGE_EVENTS].insert_one({
                "package_id": pkg_oid,
                "status": PackageStatus.at_warehouse.value,
                "branch_id": branch_oid,
                "notes": f"Confirmation email failed: {exc}",
                "created_at": now,
            })

    return {
        "package_id": package_id,
        "status": PackageStatus.at_warehouse.value,
        "branch_id": body.branch_id,
        "is_destination_warehouse": is_destination,
        "confirmation_email_sent": email_sent,
    }


@router.post("/{package_id}/depart")
async def depart(package_id: str, body: DepartBody, db: AsyncIOMotorDatabase = Depends(get_db)):
    pkg_oid = to_object_id(package_id)
    package = await db[PACKAGES].find_one({"_id": pkg_oid}) if pkg_oid else None
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")

    if package["status"] not in _DEPART_FROM:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot depart: package is '{package['status']}', expected 'at_warehouse'",
        )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    current_branch_id = package.get("current_branch_id")

    # Final leg (out_for_delivery) when to_branch_id is omitted
    if body.to_branch_id is None:
        await db[PACKAGES].update_one(
            {"_id": pkg_oid},
            {"$set": {"status": PackageStatus.out_for_delivery.value},
             "$unset": {"warehouse_arrived_at": ""}},
        )
        await db[PACKAGE_EVENTS].insert_one({
            "package_id": pkg_oid,
            "status": PackageStatus.out_for_delivery.value,
            "branch_id": current_branch_id,
            "notes": "Dispatched for last-mile delivery to customer",
            "created_at": now,
        })
        return {"package_id": package_id, "status": PackageStatus.out_for_delivery.value}

    # Warehouse-to-warehouse leg
    dest_oid = to_object_id(body.to_branch_id)
    dest_branch = await db[BRANCHES].find_one({"_id": dest_oid}) if dest_oid else None
    if not dest_branch:
        raise HTTPException(status_code=404, detail=f"Branch {body.to_branch_id} not found")

    origin_branch = await db[BRANCHES].find_one({"_id": current_branch_id}) if current_branch_id else None
    travel_min = _w2w_minutes(origin_branch, dest_branch) if origin_branch else None

    await db[PACKAGES].update_one(
        {"_id": pkg_oid},
        {"$set": {"status": PackageStatus.in_transit_w2w.value},
         "$unset": {"warehouse_arrived_at": ""}},
    )
    await db[PACKAGE_EVENTS].insert_one({
        "package_id": pkg_oid,
        "status": PackageStatus.in_transit_w2w.value,
        "branch_id": current_branch_id,
        "notes": (
            f"In transit to {dest_branch['name']}"
            + (f" — estimated {travel_min:.0f} min" if travel_min is not None else "")
        ),
        "created_at": now,
    })

    return {
        "package_id": package_id,
        "status": PackageStatus.in_transit_w2w.value,
        "to_branch_id": body.to_branch_id,
        "to_branch_name": dest_branch["name"],
        "estimated_travel_min": round(travel_min, 1) if travel_min is not None else None,
    }


@router.get("/{branch_id}/inventory")
async def inventory(branch_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    branch_oid = to_object_id(branch_id)
    branch = await db[BRANCHES].find_one({"_id": branch_oid}) if branch_oid else None
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch {branch_id} not found")

    # Live query — packages physically sitting at this warehouse right now.
    cursor = db[PACKAGES].find({
        "status": PackageStatus.at_warehouse.value,
        "current_branch_id": branch_oid,
    })
    packages = await cursor.to_list(length=None)

    return {
        "branch_id": branch_id,
        "branch_name": branch["name"],
        "count": len(packages),
        "packages": [
            {
                "inventory_id": str(p["_id"]),
                "package_id": str(p["_id"]),
                "arrived_at": p["warehouse_arrived_at"].isoformat() if p.get("warehouse_arrived_at") else None,
                "status": p["status"],
                "destination_branch_id": (
                    str(p["destination_branch_id"]) if p.get("destination_branch_id") else None
                ),
                "receiver_lat": p["receiver_lat"],
                "receiver_lng": p["receiver_lng"],
            }
            for p in packages
        ],
    }
