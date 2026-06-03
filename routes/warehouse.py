from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models.first_mile import (
    Branch,
    Package,
    PackageEvent,
    PackageStatus,
    WarehouseInventory,
)
from services.warehouse_confirmation import trigger_warehouse_arrival_email

try:
    from routing.osrm_client import get_route as _osrm_route
    _OSRM_AVAILABLE = True
except ImportError:
    _OSRM_AVAILABLE = False

router = APIRouter(prefix="/warehouse", tags=["Warehouse — Mid-Mile"])

# ── Valid incoming statuses for each transition ───────────────────────────────
_ARRIVE_FROM = {PackageStatus.at_origin_branch, PackageStatus.in_transit_w2w}
_DEPART_FROM = {PackageStatus.at_warehouse}


# ── Request bodies ────────────────────────────────────────────────────────────

class ArriveBody(BaseModel):
    branch_id: int

class DepartBody(BaseModel):
    to_branch_id: Optional[int] = None   # None → final leg (out_for_delivery)


# ── W2W travel time helper ────────────────────────────────────────────────────

def _w2w_minutes(origin: Branch, dest: Branch) -> float:
    """
    Real road distance from OSRM → minutes = (distance_km / 25) × 60.
    Falls back to haversine on OSRM error so a network blip never blocks dispatch.
    """
    if _OSRM_AVAILABLE:
        try:
            route = _osrm_route([(origin.lat, origin.lng), (dest.lat, dest.lng)])
            distance_km = route["distance"] / 1000
            return (distance_km / 25) * 60
        except Exception:
            pass

    # Haversine fallback
    from math import radians, sin, cos, sqrt, asin
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [origin.lat, origin.lng, dest.lat, dest.lng])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    distance_km = 2 * R * asin(sqrt(a))
    return (distance_km / 25) * 60


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/{package_id}/arrive")
def arrive(package_id: int, body: ArriveBody, db: Session = Depends(get_db)):
    package = db.get(Package, package_id)
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")

    if package.status not in _ARRIVE_FROM:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot arrive: package is '{package.status}', expected one of {[s.value for s in _ARRIVE_FROM]}",
        )

    branch = db.get(Branch, body.branch_id)
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch {body.branch_id} not found")

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    package.status = PackageStatus.at_warehouse
    package.current_branch_id = body.branch_id
    db.add(PackageEvent(
        package_id=package.id,
        status=PackageStatus.at_warehouse,
        branch_id=body.branch_id,
        notes=f"Arrived at {branch.name}",
    ))
    db.add(WarehouseInventory(
        package_id=package.id,
        branch_id=body.branch_id,
        arrived_at=now,
    ))

    email_sent = False
    is_destination = (
        package.destination_branch_id is not None
        and package.destination_branch_id == body.branch_id
    )
    if is_destination and package.customer_email:
        try:
            trigger_warehouse_arrival_email(package, db)
            email_sent = True
        except Exception as exc:
            # Email failure must not roll back the warehouse arrival
            db.add(PackageEvent(
                package_id=package.id,
                status=PackageStatus.at_warehouse,
                branch_id=body.branch_id,
                notes=f"Confirmation email failed: {exc}",
            ))

    db.commit()

    return {
        "package_id": package_id,
        "status": PackageStatus.at_warehouse,
        "branch_id": body.branch_id,
        "is_destination_warehouse": is_destination,
        "confirmation_email_sent": email_sent,
    }


@router.post("/{package_id}/depart")
def depart(package_id: int, body: DepartBody, db: Session = Depends(get_db)):
    package = db.get(Package, package_id)
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")

    if package.status not in _DEPART_FROM:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot depart: package is '{package.status}', expected 'at_warehouse'",
        )

    # Mark current inventory record as shipped
    inv = (
        db.query(WarehouseInventory)
        .filter(
            WarehouseInventory.package_id == package_id,
            WarehouseInventory.is_active == True,
        )
        .first()
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if inv:
        inv.is_active = False
        inv.departed_at = now

    # Final leg (out_for_delivery) when to_branch_id is omitted
    if body.to_branch_id is None:
        package.status = PackageStatus.out_for_delivery
        db.add(PackageEvent(
            package_id=package.id,
            status=PackageStatus.out_for_delivery,
            branch_id=package.current_branch_id,
            notes="Dispatched for last-mile delivery to customer",
        ))
        db.commit()
        return {"package_id": package_id, "status": PackageStatus.out_for_delivery}

    # Warehouse-to-warehouse leg
    dest_branch = db.get(Branch, body.to_branch_id)
    if not dest_branch:
        raise HTTPException(status_code=404, detail=f"Branch {body.to_branch_id} not found")

    origin_branch = db.get(Branch, package.current_branch_id) if package.current_branch_id else None
    travel_min = _w2w_minutes(origin_branch, dest_branch) if origin_branch else None

    package.status = PackageStatus.in_transit_w2w
    db.add(PackageEvent(
        package_id=package.id,
        status=PackageStatus.in_transit_w2w,
        branch_id=package.current_branch_id,
        notes=(
            f"In transit to {dest_branch.name}"
            + (f" — estimated {travel_min:.0f} min" if travel_min is not None else "")
        ),
    ))
    db.commit()

    return {
        "package_id": package_id,
        "status": PackageStatus.in_transit_w2w,
        "to_branch_id": body.to_branch_id,
        "to_branch_name": dest_branch.name,
        "estimated_travel_min": round(travel_min, 1) if travel_min is not None else None,
    }


@router.get("/{branch_id}/inventory")
def inventory(branch_id: int, db: Session = Depends(get_db)):
    branch = db.get(Branch, branch_id)
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch {branch_id} not found")

    rows = (
        db.query(WarehouseInventory)
        .filter(
            WarehouseInventory.branch_id == branch_id,
            WarehouseInventory.is_active == True,
        )
        .all()
    )

    return {
        "branch_id": branch_id,
        "branch_name": branch.name,
        "count": len(rows),
        "packages": [
            {
                "inventory_id": r.id,
                "package_id": r.package_id,
                "arrived_at": r.arrived_at.isoformat(),
                "status": r.package.status,
                "destination_branch_id": r.package.destination_branch_id,
                "receiver_lat": r.package.receiver_lat,
                "receiver_lng": r.package.receiver_lng,
            }
            for r in rows
        ],
    }
