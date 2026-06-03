"""
First-mile assignment tests.

Uses SQLite in-memory so no real DB is needed.
Run with: pytest tests/test_first_mile.py -v
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from models.first_mile import Base, Branch, Driver, Package, PackageEvent, PackageStatus, Vehicle, VehicleType
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


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# ── helpers ────────────────────────────────────────────────────────────────────

def _vehicle(db: Session, vtype: VehicleType, plate: str) -> Vehicle:
    v = Vehicle(vehicle_type=vtype, license_plate=plate, is_available=True)
    db.add(v)
    db.flush()
    return v


def _driver(db: Session, vehicle: Vehicle, lat: float, lng: float, phone: str) -> Driver:
    d = Driver(name="Test", phone=phone, current_lat=lat, current_lng=lng,
               is_available=True, vehicle_id=vehicle.id)
    db.add(d)
    db.flush()
    return d


def _package(db: Session, weight_kg: float, max_cm: float, **kwargs) -> Package:
    p = Package(
        customer_phone="012345678",
        receiver_phone="098765432",
        receiver_lat=PP_LAT,
        receiver_lng=PP_LNG,
        weight_kg=weight_kg,
        max_dimension_cm=max_cm,
        status=PackageStatus.created,
        **kwargs,
    )
    db.add(p)
    db.flush()
    return p


# ── tests ──────────────────────────────────────────────────────────────────────

def test_motorbike_assignment(db):
    """5 kg package → motorbike driver nearby → assigned."""
    v = _vehicle(db, VehicleType.motorbike, "PP-MOTO-01")
    driver = _driver(db, v, NEARBY_LAT, NEARBY_LNG, "010000001")
    pkg = _package(db, weight_kg=5, max_cm=30)

    result = assign_pickup(pkg, db, now=MORNING)

    assert result["type"] == "assigned"
    assert result["vehicle_type"] == VehicleType.motorbike
    assert result["driver_id"] == driver.id
    assert result["distance_km"] < 15
    # Event written to audit log
    assert db.query(PackageEvent).filter_by(package_id=pkg.id).count() == 1


def test_fallback_to_tuktuk(db):
    """5 kg fits motorbike but motorbike driver is >15 km away → falls back to tuktuk."""
    v_moto = _vehicle(db, VehicleType.motorbike, "PP-MOTO-02")
    _driver(db, v_moto, FAR_LAT, FAR_LNG, "010000002")  # too far

    v_tuk = _vehicle(db, VehicleType.tuktuk, "PP-TUK-01")
    tuktuk_driver = _driver(db, v_tuk, NEARBY_LAT, NEARBY_LNG, "010000003")

    pkg = _package(db, weight_kg=5, max_cm=30)
    result = assign_pickup(pkg, db, now=MORNING)

    assert result["type"] == "assigned"
    assert result["vehicle_type"] == VehicleType.tuktuk
    assert result["driver_id"] == tuktuk_driver.id


def test_container_scheduled_flow(db):
    """8 000 kg package — only container is eligible → scheduled response."""
    pkg = _package(
        db,
        weight_kg=8_000,
        max_cm=500,
        pickup_window_start=datetime(2026, 1, 6, 8, 0, tzinfo=PP_TZ),
        pickup_window_end=datetime(2026, 1, 6, 12, 0, tzinfo=PP_TZ),
    )
    result = assign_pickup(pkg, db)

    assert result["type"] == "scheduled"
    assert result["vehicle_type"] == VehicleType.container
    assert result["pickup_window_start"] is not None
    assert result["pickup_window_end"] is not None


def test_no_driver_available_fallback(db):
    """All real-time drivers are out of radius → drop-at-branch response."""
    branch = Branch(name="Central PP Branch", lat=PP_LAT, lng=PP_LNG)
    db.add(branch)
    db.flush()

    # Motorbike driver exists but is too far away
    v = _vehicle(db, VehicleType.motorbike, "PP-MOTO-03")
    _driver(db, v, FAR_LAT, FAR_LNG, "010000004")

    pkg = _package(db, weight_kg=5, max_cm=30)
    result = assign_pickup(pkg, db, now=MORNING)

    assert result["type"] == "drop_at_branch"
    assert result["nearest_branch"]["id"] == branch.id
    assert "15 km" in result["message"]
