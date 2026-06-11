"""
First-mile assignment tests.

Uses an in-memory mongomock_motor DB so no real MongoDB server is needed.
Run with: pytest tests/test_first_mile.py -v
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from bson import ObjectId

from models.first_mile import (
    BRANCHES,
    DRIVERS,
    PACKAGE_EVENTS,
    PACKAGES,
    VEHICLES,
    PackageStatus,
    VehicleType,
)
from services.first_mile import assign_pickup

PP_TZ = ZoneInfo("Asia/Phnom_Penh")

# Phnom Penh city centre — used as the receiver location in all tests
PP_LAT, PP_LNG = 11.5564, 104.9282

# Nearby coordinates (≈ 2 km from PP_LAT/PP_LNG)
NEARBY_LAT, NEARBY_LNG = 11.5600, 104.9300

# Far coordinates (≈ 60 km away — outside 15 km radius)
FAR_LAT, FAR_LNG = 12.0, 105.5

# A weekday morning — no rush hour
MORNING = datetime(2026, 1, 5, 9, 0, tzinfo=PP_TZ)


# ── helpers ────────────────────────────────────────────────────────────────────

async def _vehicle(db, vtype: VehicleType, plate: str) -> ObjectId:
    res = await db[VEHICLES].insert_one({
        "vehicle_type": vtype.value, "license_plate": plate, "is_available": True,
    })
    return res.inserted_id


async def _driver(db, vehicle_id: ObjectId, lat: float, lng: float, phone: str) -> ObjectId:
    res = await db[DRIVERS].insert_one({
        "name": "Test", "phone": phone, "current_lat": lat, "current_lng": lng,
        "is_available": True, "vehicle_id": vehicle_id,
    })
    return res.inserted_id


async def _package(db, weight_kg: float, max_cm: float, **kwargs) -> dict:
    doc = {
        "customer_phone": "012345678",
        "receiver_phone": "098765432",
        "receiver_lat": PP_LAT,
        "receiver_lng": PP_LNG,
        "weight_kg": weight_kg,
        "max_dimension_cm": max_cm,
        "status": PackageStatus.created.value,
        **kwargs,
    }
    res = await db[PACKAGES].insert_one(doc)
    doc["_id"] = res.inserted_id
    return doc


# ── tests ──────────────────────────────────────────────────────────────────────

async def test_motorbike_assignment(db):
    """5 kg package → motorbike driver nearby → assigned."""
    v = await _vehicle(db, VehicleType.motorbike, "PP-MOTO-01")
    driver = await _driver(db, v, NEARBY_LAT, NEARBY_LNG, "010000001")
    pkg = await _package(db, weight_kg=5, max_cm=30)

    result = await assign_pickup(pkg, db, now=MORNING)

    assert result["type"] == "assigned"
    assert result["vehicle_type"] == VehicleType.motorbike
    assert result["driver_id"] == str(driver)
    assert result["distance_km"] < 15
    # Event written to audit log
    assert await db[PACKAGE_EVENTS].count_documents({"package_id": pkg["_id"]}) == 1


async def test_fallback_to_tuktuk(db):
    """5 kg fits motorbike but motorbike driver is >15 km away → falls back to tuktuk."""
    v_moto = await _vehicle(db, VehicleType.motorbike, "PP-MOTO-02")
    await _driver(db, v_moto, FAR_LAT, FAR_LNG, "010000002")  # too far

    v_tuk = await _vehicle(db, VehicleType.tuktuk, "PP-TUK-01")
    tuktuk_driver = await _driver(db, v_tuk, NEARBY_LAT, NEARBY_LNG, "010000003")

    pkg = await _package(db, weight_kg=5, max_cm=30)
    result = await assign_pickup(pkg, db, now=MORNING)

    assert result["type"] == "assigned"
    assert result["vehicle_type"] == VehicleType.tuktuk
    assert result["driver_id"] == str(tuktuk_driver)


async def test_container_scheduled_flow(db):
    """8 000 kg package — only container is eligible → scheduled response."""
    pkg = await _package(
        db,
        weight_kg=8_000,
        max_cm=500,
        pickup_window_start=datetime(2026, 1, 6, 8, 0, tzinfo=PP_TZ),
        pickup_window_end=datetime(2026, 1, 6, 12, 0, tzinfo=PP_TZ),
    )
    result = await assign_pickup(pkg, db)

    assert result["type"] == "scheduled"
    assert result["vehicle_type"] == VehicleType.container
    assert result["pickup_window_start"] is not None
    assert result["pickup_window_end"] is not None


async def test_no_driver_available_fallback(db):
    """All real-time drivers are out of radius → drop-at-branch response."""
    res = await db[BRANCHES].insert_one({"name": "Central PP Branch", "lat": PP_LAT, "lng": PP_LNG})
    branch_id = res.inserted_id

    # Motorbike driver exists but is too far away
    v = await _vehicle(db, VehicleType.motorbike, "PP-MOTO-03")
    await _driver(db, v, FAR_LAT, FAR_LNG, "010000004")

    pkg = await _package(db, weight_kg=5, max_cm=30)
    result = await assign_pickup(pkg, db, now=MORNING)

    assert result["type"] == "drop_at_branch"
    assert result["nearest_branch"]["id"] == str(branch_id)
    assert "15 km" in result["message"]
