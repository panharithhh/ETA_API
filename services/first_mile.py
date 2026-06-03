from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from models.first_mile import Branch, Driver, Package, PackageEvent, PackageStatus, Vehicle, VehicleType
from utils.geo import haversine

PP_TZ = ZoneInfo("Asia/Phnom_Penh")
RADIUS_KM = 15.0

VEHICLE_SPECS: dict[VehicleType, dict] = {
    VehicleType.motorbike: {"max_kg": 15,    "max_cm": 400,  "cost_per_km": 0.05, "scheduled": False},
    VehicleType.tuktuk:    {"max_kg": 80,    "max_cm": 1000, "cost_per_km": 0.12, "scheduled": False},
    VehicleType.van:       {"max_kg": 500,   "max_cm": 2500, "cost_per_km": 0.25, "scheduled": False},
    VehicleType.container: {"max_kg": 10000, "max_cm": 6000, "cost_per_km": 0.90, "scheduled": True},
}

# Cheapest → most expensive, real-time only
_REALTIME_TIERS = [VehicleType.motorbike, VehicleType.tuktuk, VehicleType.van]


def _eligible_tiers(weight_kg: float, max_cm: float) -> list[VehicleType]:
    return [
        t for t, spec in VEHICLE_SPECS.items()
        if weight_kg <= spec["max_kg"] and max_cm <= spec["max_cm"]
    ]


def _is_rush_hour(dt: datetime) -> bool:
    local = dt.astimezone(PP_TZ)
    return 17 <= local.hour < 19  # 5–7 pm Phnom Penh


def _score(dist_km: float, pending: int, tier: VehicleType, dt: datetime) -> float:
    """Lower is better. Motorbike gets 1.5× cost multiplier during 5–7 pm."""
    cost = VEHICLE_SPECS[tier]["cost_per_km"]
    multiplier = 1.5 if (tier == VehicleType.motorbike and _is_rush_hour(dt)) else 1.0
    return dist_km * cost * multiplier + pending * 2


def _pending_count(driver_id: int, db: Session) -> int:
    return db.scalar(
        select(func.count(Package.id)).where(
            Package.assigned_driver_id == driver_id,
            Package.status.in_([PackageStatus.pending_pickup, PackageStatus.picked_up]),
        )
    ) or 0


def _drivers_within_radius(
    lat: float, lng: float, tier: VehicleType, db: Session
) -> list[tuple[Driver, float]]:
    rows = db.execute(
        select(Driver).join(Vehicle, Driver.vehicle_id == Vehicle.id).where(
            Vehicle.vehicle_type == tier,
            Driver.is_available == True,
            Vehicle.is_available == True,
            Driver.current_lat.isnot(None),
            Driver.current_lng.isnot(None),
        )
    ).scalars().all()

    return [
        (d, dist)
        for d in rows
        if (dist := haversine(lat, lng, d.current_lat, d.current_lng)) <= RADIUS_KM
    ]


def _nearest_branch(lat: float, lng: float, db: Session) -> Branch | None:
    branches = db.execute(select(Branch)).scalars().all()
    if not branches:
        return None
    return min(branches, key=lambda b: haversine(lat, lng, b.lat, b.lng))


def assign_pickup(package: Package, db: Session, now: datetime | None = None) -> dict:
    if now is None:
        now = datetime.now(PP_TZ)

    eligible = _eligible_tiers(package.weight_kg, package.max_dimension_cm)
    if not eligible:
        raise ValueError("Package exceeds all vehicle capacities")

    realtime_eligible = [t for t in _REALTIME_TIERS if t in eligible]

    # Container-only path — package too heavy/large for any real-time vehicle
    if not realtime_eligible:
        return {
            "type": "scheduled",
            "vehicle_type": VehicleType.container,
            "pickup_window_start": package.pickup_window_start,
            "pickup_window_end": package.pickup_window_end,
            "message": "Package requires container — 12-hour lead time, scheduled booking",
        }

    # Fallback chain: motorbike → tuktuk → van
    for tier in realtime_eligible:
        candidates = _drivers_within_radius(package.receiver_lat, package.receiver_lng, tier, db)
        if not candidates:
            continue

        best_driver, best_dist = min(
            candidates,
            key=lambda x: _score(x[1], _pending_count(x[0].id, db), tier, now),
        )

        package.assigned_driver_id = best_driver.id
        package.assigned_vehicle_id = best_driver.vehicle_id
        package.status = PackageStatus.pending_pickup
        db.add(PackageEvent(
            package_id=package.id,
            status=PackageStatus.pending_pickup,
            driver_id=best_driver.id,
            notes=f"Auto-assigned {tier.value}, dist={best_dist:.2f}km",
        ))
        db.commit()
        db.refresh(package)

        return {
            "type": "assigned",
            "package_id": package.id,
            "status": package.status,
            "vehicle_type": tier,
            "driver_id": best_driver.id,
            "driver_phone": best_driver.phone,
            "distance_km": round(best_dist, 2),
        }

    # All real-time tiers exhausted → drop at branch
    branch = _nearest_branch(package.receiver_lat, package.receiver_lng, db)
    return {
        "type": "drop_at_branch",
        "package_id": package.id,
        "message": "No driver available within 15 km. Please drop the package at the nearest branch.",
        "nearest_branch": {
            "id": branch.id if branch else None,
            "name": branch.name if branch else "N/A",
            "lat": branch.lat if branch else None,
            "lng": branch.lng if branch else None,
        },
    }
