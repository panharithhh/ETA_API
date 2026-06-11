from datetime import datetime
from zoneinfo import ZoneInfo

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.first_mile import (
    BRANCHES,
    DRIVERS,
    PACKAGE_EVENTS,
    PACKAGES,
    VEHICLES,
    PackageStatus,
    VehicleType,
)
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


async def _pending_count(driver_id, db: AsyncIOMotorDatabase) -> int:
    return await db[PACKAGES].count_documents({
        "assigned_driver_id": driver_id,
        "status": {"$in": [PackageStatus.pending_pickup.value, PackageStatus.picked_up.value]},
    })


async def _drivers_within_radius(
    lat: float, lng: float, tier: VehicleType, db: AsyncIOMotorDatabase
) -> list[tuple[dict, float]]:
    vehicle_ids = await db[VEHICLES].distinct("_id", {
        "vehicle_type": tier.value,
        "is_available": True,
    })
    if not vehicle_ids:
        return []

    cursor = db[DRIVERS].find({
        "vehicle_id": {"$in": vehicle_ids},
        "is_available": True,
        "current_lat": {"$ne": None},
        "current_lng": {"$ne": None},
    })
    drivers = await cursor.to_list(length=None)

    return [
        (d, dist)
        for d in drivers
        if (dist := haversine(lat, lng, d["current_lat"], d["current_lng"])) <= RADIUS_KM
    ]


async def _nearest_branch(lat: float, lng: float, db: AsyncIOMotorDatabase) -> dict | None:
    branches = await db[BRANCHES].find().to_list(length=None)
    if not branches:
        return None
    return min(branches, key=lambda b: haversine(lat, lng, b["lat"], b["lng"]))


async def assign_pickup(package: dict, db: AsyncIOMotorDatabase, now: datetime | None = None) -> dict:
    if now is None:
        now = datetime.now(PP_TZ)

    eligible = _eligible_tiers(package["weight_kg"], package["max_dimension_cm"])
    if not eligible:
        raise ValueError("Package exceeds all vehicle capacities")

    realtime_eligible = [t for t in _REALTIME_TIERS if t in eligible]

    # Container-only path — package too heavy/large for any real-time vehicle
    if not realtime_eligible:
        return {
            "type": "scheduled",
            "vehicle_type": VehicleType.container,
            "pickup_window_start": package.get("pickup_window_start"),
            "pickup_window_end": package.get("pickup_window_end"),
            "message": "Package requires container — 12-hour lead time, scheduled booking",
        }

    # Fallback chain: motorbike → tuktuk → van
    for tier in realtime_eligible:
        candidates = await _drivers_within_radius(
            package["receiver_lat"], package["receiver_lng"], tier, db
        )
        if not candidates:
            continue

        scored = [
            (driver, dist, _score(dist, await _pending_count(driver["_id"], db), tier, now))
            for driver, dist in candidates
        ]
        best_driver, best_dist, _ = min(scored, key=lambda x: x[2])

        await db[PACKAGES].update_one(
            {"_id": package["_id"]},
            {"$set": {
                "assigned_driver_id": best_driver["_id"],
                "assigned_vehicle_id": best_driver.get("vehicle_id"),
                "status": PackageStatus.pending_pickup.value,
            }},
        )
        await db[PACKAGE_EVENTS].insert_one({
            "package_id": package["_id"],
            "status": PackageStatus.pending_pickup.value,
            "driver_id": best_driver["_id"],
            "notes": f"Auto-assigned {tier.value}, dist={best_dist:.2f}km",
            "created_at": now.replace(tzinfo=None),
        })

        return {
            "type": "assigned",
            "package_id": str(package["_id"]),
            "status": PackageStatus.pending_pickup.value,
            "vehicle_type": tier,
            "driver_id": str(best_driver["_id"]),
            "driver_phone": best_driver["phone"],
            "distance_km": round(best_dist, 2),
        }

    # All real-time tiers exhausted → drop at branch
    branch = await _nearest_branch(package["receiver_lat"], package["receiver_lng"], db)
    return {
        "type": "drop_at_branch",
        "package_id": str(package["_id"]),
        "message": "No driver available within 15 km. Please drop the package at the nearest branch.",
        "nearest_branch": {
            "id": str(branch["_id"]) if branch else None,
            "name": branch["name"] if branch else "N/A",
            "lat": branch["lat"] if branch else None,
            "lng": branch["lng"] if branch else None,
        },
    }
