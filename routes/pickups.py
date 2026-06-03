from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.first_mile import Package, PackageEvent, PackageStatus
from schemas.pickups import DropoffRequest, PickupBookRequest, ScanRequest
from services.first_mile import assign_pickup

router = APIRouter(prefix="/pickups", tags=["First Mile — Pickups"])


@router.post("")
def book_pickup(data: PickupBookRequest, db: Session = Depends(get_db)):
    package = Package(
        customer_phone=data.customer_phone,
        customer_email=data.customer_email,
        receiver_phone=data.receiver_phone,
        receiver_lat=data.receiver_lat,
        receiver_lng=data.receiver_lng,
        weight_kg=data.weight_kg,
        max_dimension_cm=data.max_dimension_cm,
        origin_branch_id=data.origin_branch_id,
        destination_branch_id=data.destination_branch_id,
        pickup_window_start=data.pickup_window_start,
        pickup_window_end=data.pickup_window_end,
        status=PackageStatus.created,
    )
    db.add(package)
    db.flush()  # get package.id before assign_pickup writes events

    result = assign_pickup(package, db)
    return result


@router.post("/{package_id}/scan")
def scan_qr(package_id: int, _: ScanRequest = ScanRequest(), db: Session = Depends(get_db)):
    package = db.get(Package, package_id)
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
    if package.status != PackageStatus.pending_pickup:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot scan: package is '{package.status}', expected 'pending_pickup'",
        )

    package.status = PackageStatus.picked_up
    db.add(PackageEvent(
        package_id=package.id,
        status=PackageStatus.picked_up,
        driver_id=package.assigned_driver_id,
        notes="QR scanned by driver at pickup location",
    ))
    db.commit()

    return {"package_id": package_id, "status": PackageStatus.picked_up}


@router.post("/{package_id}/dropoff")
def dropoff_at_branch(package_id: int, data: DropoffRequest, db: Session = Depends(get_db)):
    package = db.get(Package, package_id)
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
    if package.status != PackageStatus.picked_up:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot drop off: package is '{package.status}', expected 'picked_up'",
        )

    package.status = PackageStatus.at_origin_branch
    package.current_branch_id = data.branch_id
    db.add(PackageEvent(
        package_id=package.id,
        status=PackageStatus.at_origin_branch,
        driver_id=package.assigned_driver_id,
        branch_id=data.branch_id,
        notes="Package checked in at origin branch — ready for middle mile",
    ))
    db.commit()

    return {
        "package_id": package_id,
        "status": PackageStatus.at_origin_branch,
        "branch_id": data.branch_id,
    }
