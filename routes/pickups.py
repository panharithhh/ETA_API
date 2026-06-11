from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from db import get_db
from models.first_mile import (
    PACKAGE_EVENTS,
    PACKAGES,
    PackageStatus,
    to_object_id,
)
from schemas.pickups import DropoffRequest, PickupBookRequest, ScanRequest
from services.first_mile import assign_pickup

router = APIRouter(prefix="/pickups", tags=["First Mile — Pickups"])


@router.post("")
async def book_pickup(data: PickupBookRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    package = {
        "customer_phone": data.customer_phone,
        "customer_email": data.customer_email,
        "receiver_phone": data.receiver_phone,
        "receiver_lat": data.receiver_lat,
        "receiver_lng": data.receiver_lng,
        "weight_kg": data.weight_kg,
        "max_dimension_cm": data.max_dimension_cm,
        "origin_branch_id": to_object_id(data.origin_branch_id),
        "destination_branch_id": to_object_id(data.destination_branch_id),
        "current_branch_id": None,
        "assigned_driver_id": None,
        "assigned_vehicle_id": None,
        "pickup_window_start": data.pickup_window_start,
        "pickup_window_end": data.pickup_window_end,
        "status": PackageStatus.created.value,
        "created_at": now,
    }
    result = await db[PACKAGES].insert_one(package)
    package["_id"] = result.inserted_id

    return await assign_pickup(package, db)


@router.post("/{package_id}/scan")
async def scan_qr(
    package_id: str,
    _: ScanRequest = ScanRequest(),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    pkg_oid = to_object_id(package_id)
    package = await db[PACKAGES].find_one({"_id": pkg_oid}) if pkg_oid else None
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
    if package["status"] != PackageStatus.pending_pickup.value:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot scan: package is '{package['status']}', expected 'pending_pickup'",
        )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    await db[PACKAGES].update_one(
        {"_id": pkg_oid}, {"$set": {"status": PackageStatus.picked_up.value}}
    )
    await db[PACKAGE_EVENTS].insert_one({
        "package_id": pkg_oid,
        "status": PackageStatus.picked_up.value,
        "driver_id": package.get("assigned_driver_id"),
        "notes": "QR scanned by driver at pickup location",
        "created_at": now,
    })

    return {"package_id": package_id, "status": PackageStatus.picked_up.value}


@router.post("/{package_id}/dropoff")
async def dropoff_at_branch(
    package_id: str, data: DropoffRequest, db: AsyncIOMotorDatabase = Depends(get_db)
):
    pkg_oid = to_object_id(package_id)
    package = await db[PACKAGES].find_one({"_id": pkg_oid}) if pkg_oid else None
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
    if package["status"] != PackageStatus.picked_up.value:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot drop off: package is '{package['status']}', expected 'picked_up'",
        )

    branch_oid = to_object_id(data.branch_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    await db[PACKAGES].update_one(
        {"_id": pkg_oid},
        {"$set": {
            "status": PackageStatus.at_origin_branch.value,
            "current_branch_id": branch_oid,
        }},
    )
    await db[PACKAGE_EVENTS].insert_one({
        "package_id": pkg_oid,
        "status": PackageStatus.at_origin_branch.value,
        "driver_id": package.get("assigned_driver_id"),
        "branch_id": branch_oid,
        "notes": "Package checked in at origin branch — ready for middle mile",
        "created_at": now,
    })

    return {
        "package_id": package_id,
        "status": PackageStatus.at_origin_branch.value,
        "branch_id": data.branch_id,
    }
