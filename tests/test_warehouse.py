"""
Warehouse / mid-mile endpoint tests.

Uses SQLite in-memory DB + FastAPI TestClient. No network calls made.
Run with: pytest tests/test_warehouse.py -v
"""

from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database import get_db
from models.first_mile import (
    Base, Branch, Package, PackageStatus, WarehouseInventory,
)
from routes.warehouse import router as warehouse_router


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def client(db):
    app = FastAPI()
    app.include_router(warehouse_router)
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


# ── Data helpers ───────────────────────────────────────────────────────────────

def _branch(db: Session, name: str, lat: float = 11.5564, lng: float = 104.9282) -> Branch:
    b = Branch(name=name, lat=lat, lng=lng)
    db.add(b)
    db.flush()
    return b


def _package(
    db: Session,
    status: PackageStatus = PackageStatus.at_origin_branch,
    current_branch_id: int = None,
    destination_branch_id: int = None,
    customer_email: str = None,
) -> Package:
    p = Package(
        customer_phone="012345678",
        receiver_phone="098765432",
        receiver_lat=11.5564,
        receiver_lng=104.9282,
        weight_kg=5,
        max_dimension_cm=30,
        status=status,
        current_branch_id=current_branch_id,
        destination_branch_id=destination_branch_id,
        customer_email=customer_email,
    )
    db.add(p)
    db.flush()
    return p


def _inventory(db: Session, pkg: Package, branch: Branch, active: bool = True) -> WarehouseInventory:
    inv = WarehouseInventory(
        package_id=pkg.id,
        branch_id=branch.id,
        arrived_at=datetime.utcnow(),
        is_active=active,
        departed_at=None if active else datetime.utcnow(),
    )
    db.add(inv)
    db.flush()
    return inv


# ── POST /warehouse/{id}/arrive ────────────────────────────────────────────────

class TestArrive:
    def test_from_at_origin_branch(self, client, db):
        branch = _branch(db, "Warehouse A")
        pkg = _package(db, PackageStatus.at_origin_branch)

        r = client.post(f"/warehouse/{pkg.id}/arrive", json={"branch_id": branch.id})

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "at_warehouse"
        assert body["branch_id"] == branch.id

        db.expire_all()
        updated = db.get(Package, pkg.id)
        assert updated.status == PackageStatus.at_warehouse
        assert updated.current_branch_id == branch.id

        inv = db.query(WarehouseInventory).filter_by(package_id=pkg.id).one()
        assert inv.is_active is True
        assert inv.arrived_at is not None

    def test_from_in_transit_w2w(self, client, db):
        branch = _branch(db, "Warehouse B")
        pkg = _package(db, PackageStatus.in_transit_w2w)

        r = client.post(f"/warehouse/{pkg.id}/arrive", json={"branch_id": branch.id})

        assert r.status_code == 200
        assert r.json()["status"] == "at_warehouse"

    def test_wrong_status_returns_409(self, client, db):
        branch = _branch(db, "Warehouse C")
        pkg = _package(db, PackageStatus.picked_up)

        r = client.post(f"/warehouse/{pkg.id}/arrive", json={"branch_id": branch.id})

        assert r.status_code == 409
        assert "picked_up" in r.json()["detail"]

    def test_package_not_found_returns_404(self, client, db):
        branch = _branch(db, "Warehouse D")
        r = client.post("/warehouse/9999/arrive", json={"branch_id": branch.id})
        assert r.status_code == 404

    def test_branch_not_found_returns_404(self, client, db):
        pkg = _package(db, PackageStatus.at_origin_branch)
        r = client.post(f"/warehouse/{pkg.id}/arrive", json={"branch_id": 9999})
        assert r.status_code == 404

    def test_email_fires_at_destination_warehouse(self, client, db):
        branch = _branch(db, "Dest Warehouse")
        pkg = _package(
            db,
            PackageStatus.at_origin_branch,
            destination_branch_id=branch.id,
            customer_email="customer@example.com",
        )

        with patch("routes.warehouse.trigger_warehouse_arrival_email") as mock_email:
            r = client.post(f"/warehouse/{pkg.id}/arrive", json={"branch_id": branch.id})

        assert r.status_code == 200
        assert r.json()["is_destination_warehouse"] is True
        assert r.json()["confirmation_email_sent"] is True
        mock_email.assert_called_once()

    def test_no_email_at_transit_warehouse(self, client, db):
        transit = _branch(db, "Transit Warehouse")
        dest = _branch(db, "Dest Warehouse")
        pkg = _package(
            db,
            PackageStatus.at_origin_branch,
            destination_branch_id=dest.id,
            customer_email="customer@example.com",
        )

        with patch("routes.warehouse.trigger_warehouse_arrival_email") as mock_email:
            r = client.post(f"/warehouse/{pkg.id}/arrive", json={"branch_id": transit.id})

        assert r.status_code == 200
        assert r.json()["is_destination_warehouse"] is False
        assert r.json()["confirmation_email_sent"] is False
        mock_email.assert_not_called()

    def test_no_email_when_customer_email_missing(self, client, db):
        branch = _branch(db, "Warehouse E")
        pkg = _package(
            db,
            PackageStatus.at_origin_branch,
            destination_branch_id=branch.id,
            customer_email=None,
        )

        with patch("routes.warehouse.trigger_warehouse_arrival_email") as mock_email:
            r = client.post(f"/warehouse/{pkg.id}/arrive", json={"branch_id": branch.id})

        assert r.status_code == 200
        assert r.json()["confirmation_email_sent"] is False
        mock_email.assert_not_called()

    def test_email_failure_does_not_rollback_arrival(self, client, db):
        branch = _branch(db, "Warehouse F")
        pkg = _package(
            db,
            PackageStatus.at_origin_branch,
            destination_branch_id=branch.id,
            customer_email="customer@example.com",
        )

        with patch("routes.warehouse.trigger_warehouse_arrival_email", side_effect=Exception("SMTP down")):
            r = client.post(f"/warehouse/{pkg.id}/arrive", json={"branch_id": branch.id})

        assert r.status_code == 200
        assert r.json()["confirmation_email_sent"] is False

        db.expire_all()
        assert db.get(Package, pkg.id).status == PackageStatus.at_warehouse


# ── POST /warehouse/{id}/depart ────────────────────────────────────────────────

class TestDepart:
    def _at_warehouse(self, db: Session, branch: Branch) -> Package:
        pkg = _package(db, PackageStatus.at_warehouse, current_branch_id=branch.id)
        _inventory(db, pkg, branch, active=True)
        return pkg

    def test_final_leg_sets_out_for_delivery(self, client, db):
        branch = _branch(db, "Last Warehouse")
        pkg = self._at_warehouse(db, branch)

        r = client.post(f"/warehouse/{pkg.id}/depart", json={})

        assert r.status_code == 200
        assert r.json()["status"] == "out_for_delivery"

        db.expire_all()
        assert db.get(Package, pkg.id).status == PackageStatus.out_for_delivery

        inv = db.query(WarehouseInventory).filter_by(package_id=pkg.id).one()
        assert inv.is_active is False
        assert inv.departed_at is not None

    def test_w2w_leg_sets_in_transit(self, client, db):
        origin = _branch(db, "WH Origin", lat=11.55, lng=104.92)
        dest = _branch(db, "WH Dest", lat=12.10, lng=105.50)
        pkg = self._at_warehouse(db, origin)

        # 75 km road distance → (75 / 25) * 60 = 180 min
        with patch("routes.warehouse._osrm_route", return_value={"distance": 75000}):
            r = client.post(f"/warehouse/{pkg.id}/depart", json={"to_branch_id": dest.id})

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "in_transit_w2w"
        assert body["to_branch_id"] == dest.id
        assert body["to_branch_name"] == "WH Dest"
        assert body["estimated_travel_min"] == 180.0

        db.expire_all()
        assert db.get(Package, pkg.id).status == PackageStatus.in_transit_w2w

    def test_w2w_osrm_failure_falls_back_to_haversine(self, client, db):
        origin = _branch(db, "WH North", lat=11.55, lng=104.92)
        dest = _branch(db, "WH South", lat=11.60, lng=104.95)
        pkg = self._at_warehouse(db, origin)

        with patch("routes.warehouse._osrm_route", side_effect=Exception("OSRM down")):
            r = client.post(f"/warehouse/{pkg.id}/depart", json={"to_branch_id": dest.id})

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "in_transit_w2w"
        assert body["estimated_travel_min"] > 0  # haversine fallback produced a value

    def test_wrong_status_returns_409(self, client, db):
        branch = _branch(db, "Warehouse G")
        pkg = _package(db, PackageStatus.at_origin_branch, current_branch_id=branch.id)

        r = client.post(f"/warehouse/{pkg.id}/depart", json={})

        assert r.status_code == 409
        assert "at_origin_branch" in r.json()["detail"]

    def test_package_not_found_returns_404(self, client, db):
        r = client.post("/warehouse/9999/depart", json={})
        assert r.status_code == 404

    def test_to_branch_not_found_returns_404(self, client, db):
        branch = _branch(db, "Warehouse H")
        pkg = self._at_warehouse(db, branch)

        r = client.post(f"/warehouse/{pkg.id}/depart", json={"to_branch_id": 9999})
        assert r.status_code == 404


# ── GET /warehouse/{branch_id}/inventory ──────────────────────────────────────

class TestInventory:
    def test_returns_active_packages(self, client, db):
        branch = _branch(db, "Central WH")
        pkg1 = _package(db, PackageStatus.at_warehouse, current_branch_id=branch.id)
        pkg2 = _package(db, PackageStatus.at_warehouse, current_branch_id=branch.id)
        _inventory(db, pkg1, branch)
        _inventory(db, pkg2, branch)

        r = client.get(f"/warehouse/{branch.id}/inventory")

        assert r.status_code == 200
        body = r.json()
        assert body["branch_id"] == branch.id
        assert body["branch_name"] == "Central WH"
        assert body["count"] == 2
        ids = {p["package_id"] for p in body["packages"]}
        assert ids == {pkg1.id, pkg2.id}

    def test_excludes_departed_packages(self, client, db):
        branch = _branch(db, "Mixed WH")
        active_pkg = _package(db, PackageStatus.at_warehouse, current_branch_id=branch.id)
        gone_pkg = _package(db, PackageStatus.in_transit_w2w, current_branch_id=branch.id)
        _inventory(db, active_pkg, branch, active=True)
        _inventory(db, gone_pkg, branch, active=False)

        r = client.get(f"/warehouse/{branch.id}/inventory")

        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["packages"][0]["package_id"] == active_pkg.id

    def test_empty_warehouse_returns_zero(self, client, db):
        branch = _branch(db, "Empty WH")

        r = client.get(f"/warehouse/{branch.id}/inventory")

        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 0
        assert body["packages"] == []

    def test_branch_not_found_returns_404(self, client, db):
        r = client.get("/warehouse/9999/inventory")
        assert r.status_code == 404

    def test_response_includes_expected_fields(self, client, db):
        branch = _branch(db, "Field Check WH")
        dest = _branch(db, "Dest Branch")
        pkg = _package(
            db,
            PackageStatus.at_warehouse,
            current_branch_id=branch.id,
            destination_branch_id=dest.id,
        )
        _inventory(db, pkg, branch)

        r = client.get(f"/warehouse/{branch.id}/inventory")

        item = r.json()["packages"][0]
        assert "inventory_id" in item
        assert "package_id" in item
        assert "arrived_at" in item
        assert "status" in item
        assert "destination_branch_id" in item
        assert "receiver_lat" in item
        assert "receiver_lng" in item


# ── Full lifecycle chain ───────────────────────────────────────────────────────

class TestFullLifecycle:
    def test_origin_transit_warehouse_delivery(self, client, db):
        """
        at_origin_branch
          → arrive WH-A (no email, not destination)
          → depart to WH-B (in_transit_w2w, travel time returned)
          → arrive WH-B (email fires, is destination)
          → depart final leg (out_for_delivery)
          → both warehouse inventories empty
        """
        wh_a = _branch(db, "Warehouse A", lat=11.55, lng=104.92)
        wh_b = _branch(db, "Warehouse B", lat=12.10, lng=105.50)

        pkg = _package(
            db,
            PackageStatus.at_origin_branch,
            destination_branch_id=wh_b.id,
            customer_email="receiver@example.com",
        )

        # Step 1 — arrive at WH-A (transit, no email)
        with patch("routes.warehouse.trigger_warehouse_arrival_email") as mock_email:
            r = client.post(f"/warehouse/{pkg.id}/arrive", json={"branch_id": wh_a.id})
        assert r.status_code == 200
        assert r.json()["is_destination_warehouse"] is False
        mock_email.assert_not_called()

        # Step 2 — depart WH-A to WH-B, 100 km → 240 min
        with patch("routes.warehouse._osrm_route", return_value={"distance": 100_000}):
            r = client.post(f"/warehouse/{pkg.id}/depart", json={"to_branch_id": wh_b.id})
        assert r.status_code == 200
        assert r.json()["status"] == "in_transit_w2w"
        assert r.json()["estimated_travel_min"] == 240.0

        # WH-A inventory now inactive
        assert client.get(f"/warehouse/{wh_a.id}/inventory").json()["count"] == 0

        # Step 3 — arrive at WH-B (destination → email)
        with patch("routes.warehouse.trigger_warehouse_arrival_email") as mock_email:
            r = client.post(f"/warehouse/{pkg.id}/arrive", json={"branch_id": wh_b.id})
        assert r.status_code == 200
        assert r.json()["is_destination_warehouse"] is True
        assert r.json()["confirmation_email_sent"] is True
        mock_email.assert_called_once()

        # WH-B inventory now has 1 package
        assert client.get(f"/warehouse/{wh_b.id}/inventory").json()["count"] == 1

        # Step 4 — final dispatch
        r = client.post(f"/warehouse/{pkg.id}/depart", json={})
        assert r.status_code == 200
        assert r.json()["status"] == "out_for_delivery"

        # Both warehouses empty
        assert client.get(f"/warehouse/{wh_a.id}/inventory").json()["count"] == 0
        assert client.get(f"/warehouse/{wh_b.id}/inventory").json()["count"] == 0

        db.expire_all()
        assert db.get(Package, pkg.id).status == PackageStatus.out_for_delivery
