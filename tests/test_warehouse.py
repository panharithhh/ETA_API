"""
Warehouse / mid-mile endpoint tests.

Uses an in-memory mongomock_motor DB + httpx AsyncClient against the ASGI app.
No network calls made. Run with: pytest tests/test_warehouse.py -v
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest_asyncio
from bson import ObjectId
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from db import get_db
from models.first_mile import BRANCHES, PACKAGES, PackageStatus
from routes.warehouse import router as warehouse_router


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(db):
    app = FastAPI()
    app.include_router(warehouse_router)
    app.dependency_overrides[get_db] = lambda: db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Data helpers ───────────────────────────────────────────────────────────────

async def _branch(db, name: str, lat: float = 11.5564, lng: float = 104.9282) -> ObjectId:
    res = await db[BRANCHES].insert_one({"name": name, "lat": lat, "lng": lng})
    return res.inserted_id


async def _package(
    db,
    status: PackageStatus = PackageStatus.at_origin_branch,
    current_branch_id: ObjectId = None,
    destination_branch_id: ObjectId = None,
    customer_email: str = None,
    in_warehouse: bool = False,
) -> ObjectId:
    doc = {
        "customer_phone": "012345678",
        "receiver_phone": "098765432",
        "receiver_lat": 11.5564,
        "receiver_lng": 104.9282,
        "weight_kg": 5,
        "max_dimension_cm": 30,
        "status": status.value,
        "current_branch_id": current_branch_id,
        "destination_branch_id": destination_branch_id,
        "customer_email": customer_email,
    }
    if in_warehouse:
        doc["warehouse_arrived_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    res = await db[PACKAGES].insert_one(doc)
    return res.inserted_id


async def _get_package(db, pkg_id: ObjectId) -> dict:
    return await db[PACKAGES].find_one({"_id": pkg_id})


# ── POST /warehouse/{id}/arrive ────────────────────────────────────────────────

class TestArrive:
    async def test_from_at_origin_branch(self, client, db):
        branch = await _branch(db, "Warehouse A")
        pkg = await _package(db, PackageStatus.at_origin_branch)

        r = await client.post(f"/warehouse/{pkg}/arrive", json={"branch_id": str(branch)})

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "at_warehouse"
        assert body["branch_id"] == str(branch)

        updated = await _get_package(db, pkg)
        assert updated["status"] == PackageStatus.at_warehouse.value
        assert updated["current_branch_id"] == branch
        assert updated["warehouse_arrived_at"] is not None

    async def test_from_in_transit_w2w(self, client, db):
        branch = await _branch(db, "Warehouse B")
        pkg = await _package(db, PackageStatus.in_transit_w2w)

        r = await client.post(f"/warehouse/{pkg}/arrive", json={"branch_id": str(branch)})

        assert r.status_code == 200
        assert r.json()["status"] == "at_warehouse"

    async def test_wrong_status_returns_409(self, client, db):
        branch = await _branch(db, "Warehouse C")
        pkg = await _package(db, PackageStatus.picked_up)

        r = await client.post(f"/warehouse/{pkg}/arrive", json={"branch_id": str(branch)})

        assert r.status_code == 409
        assert "picked_up" in r.json()["detail"]

    async def test_package_not_found_returns_404(self, client, db):
        branch = await _branch(db, "Warehouse D")
        r = await client.post(f"/warehouse/{ObjectId()}/arrive", json={"branch_id": str(branch)})
        assert r.status_code == 404

    async def test_branch_not_found_returns_404(self, client, db):
        pkg = await _package(db, PackageStatus.at_origin_branch)
        r = await client.post(f"/warehouse/{pkg}/arrive", json={"branch_id": str(ObjectId())})
        assert r.status_code == 404

    async def test_email_fires_at_destination_warehouse(self, client, db):
        branch = await _branch(db, "Dest Warehouse")
        pkg = await _package(
            db,
            PackageStatus.at_origin_branch,
            destination_branch_id=branch,
            customer_email="customer@example.com",
        )

        with patch("routes.warehouse.trigger_warehouse_arrival_email", new_callable=AsyncMock) as mock_email:
            r = await client.post(f"/warehouse/{pkg}/arrive", json={"branch_id": str(branch)})

        assert r.status_code == 200
        assert r.json()["is_destination_warehouse"] is True
        assert r.json()["confirmation_email_sent"] is True
        mock_email.assert_awaited_once()

    async def test_no_email_at_transit_warehouse(self, client, db):
        transit = await _branch(db, "Transit Warehouse")
        dest = await _branch(db, "Dest Warehouse")
        pkg = await _package(
            db,
            PackageStatus.at_origin_branch,
            destination_branch_id=dest,
            customer_email="customer@example.com",
        )

        with patch("routes.warehouse.trigger_warehouse_arrival_email", new_callable=AsyncMock) as mock_email:
            r = await client.post(f"/warehouse/{pkg}/arrive", json={"branch_id": str(transit)})

        assert r.status_code == 200
        assert r.json()["is_destination_warehouse"] is False
        assert r.json()["confirmation_email_sent"] is False
        mock_email.assert_not_called()

    async def test_no_email_when_customer_email_missing(self, client, db):
        branch = await _branch(db, "Warehouse E")
        pkg = await _package(
            db,
            PackageStatus.at_origin_branch,
            destination_branch_id=branch,
            customer_email=None,
        )

        with patch("routes.warehouse.trigger_warehouse_arrival_email", new_callable=AsyncMock) as mock_email:
            r = await client.post(f"/warehouse/{pkg}/arrive", json={"branch_id": str(branch)})

        assert r.status_code == 200
        assert r.json()["confirmation_email_sent"] is False
        mock_email.assert_not_called()

    async def test_email_failure_does_not_rollback_arrival(self, client, db):
        branch = await _branch(db, "Warehouse F")
        pkg = await _package(
            db,
            PackageStatus.at_origin_branch,
            destination_branch_id=branch,
            customer_email="customer@example.com",
        )

        with patch(
            "routes.warehouse.trigger_warehouse_arrival_email",
            new_callable=AsyncMock,
            side_effect=Exception("SMTP down"),
        ):
            r = await client.post(f"/warehouse/{pkg}/arrive", json={"branch_id": str(branch)})

        assert r.status_code == 200
        assert r.json()["confirmation_email_sent"] is False

        assert (await _get_package(db, pkg))["status"] == PackageStatus.at_warehouse.value


# ── POST /warehouse/{id}/depart ────────────────────────────────────────────────

class TestDepart:
    async def _at_warehouse(self, db, branch: ObjectId) -> ObjectId:
        return await _package(
            db, PackageStatus.at_warehouse, current_branch_id=branch, in_warehouse=True
        )

    async def test_final_leg_sets_out_for_delivery(self, client, db):
        branch = await _branch(db, "Last Warehouse")
        pkg = await self._at_warehouse(db, branch)

        r = await client.post(f"/warehouse/{pkg}/depart", json={})

        assert r.status_code == 200
        assert r.json()["status"] == "out_for_delivery"

        updated = await _get_package(db, pkg)
        assert updated["status"] == PackageStatus.out_for_delivery.value
        assert "warehouse_arrived_at" not in updated

        # No longer shows in the branch inventory
        inv = await client.get(f"/warehouse/{branch}/inventory")
        assert inv.json()["count"] == 0

    async def test_w2w_leg_sets_in_transit(self, client, db):
        origin = await _branch(db, "WH Origin", lat=11.55, lng=104.92)
        dest = await _branch(db, "WH Dest", lat=12.10, lng=105.50)
        pkg = await self._at_warehouse(db, origin)

        # 75 km road distance → (75 / 25) * 60 = 180 min
        with patch("routes.warehouse._osrm_route", return_value={"distance": 75000}):
            r = await client.post(f"/warehouse/{pkg}/depart", json={"to_branch_id": str(dest)})

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "in_transit_w2w"
        assert body["to_branch_id"] == str(dest)
        assert body["to_branch_name"] == "WH Dest"
        assert body["estimated_travel_min"] == 180.0

        assert (await _get_package(db, pkg))["status"] == PackageStatus.in_transit_w2w.value

    async def test_w2w_osrm_failure_falls_back_to_haversine(self, client, db):
        origin = await _branch(db, "WH North", lat=11.55, lng=104.92)
        dest = await _branch(db, "WH South", lat=11.60, lng=104.95)
        pkg = await self._at_warehouse(db, origin)

        with patch("routes.warehouse._osrm_route", side_effect=Exception("OSRM down")):
            r = await client.post(f"/warehouse/{pkg}/depart", json={"to_branch_id": str(dest)})

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "in_transit_w2w"
        assert body["estimated_travel_min"] > 0  # haversine fallback produced a value

    async def test_wrong_status_returns_409(self, client, db):
        branch = await _branch(db, "Warehouse G")
        pkg = await _package(db, PackageStatus.at_origin_branch, current_branch_id=branch)

        r = await client.post(f"/warehouse/{pkg}/depart", json={})

        assert r.status_code == 409
        assert "at_origin_branch" in r.json()["detail"]

    async def test_package_not_found_returns_404(self, client, db):
        r = await client.post(f"/warehouse/{ObjectId()}/depart", json={})
        assert r.status_code == 404

    async def test_to_branch_not_found_returns_404(self, client, db):
        branch = await _branch(db, "Warehouse H")
        pkg = await self._at_warehouse(db, branch)

        r = await client.post(f"/warehouse/{pkg}/depart", json={"to_branch_id": str(ObjectId())})
        assert r.status_code == 404


# ── GET /warehouse/{branch_id}/inventory ──────────────────────────────────────

class TestInventory:
    async def test_returns_active_packages(self, client, db):
        branch = await _branch(db, "Central WH")
        pkg1 = await _package(db, PackageStatus.at_warehouse, current_branch_id=branch, in_warehouse=True)
        pkg2 = await _package(db, PackageStatus.at_warehouse, current_branch_id=branch, in_warehouse=True)

        r = await client.get(f"/warehouse/{branch}/inventory")

        assert r.status_code == 200
        body = r.json()
        assert body["branch_id"] == str(branch)
        assert body["branch_name"] == "Central WH"
        assert body["count"] == 2
        ids = {p["package_id"] for p in body["packages"]}
        assert ids == {str(pkg1), str(pkg2)}

    async def test_excludes_departed_packages(self, client, db):
        branch = await _branch(db, "Mixed WH")
        active_pkg = await _package(db, PackageStatus.at_warehouse, current_branch_id=branch, in_warehouse=True)
        # A departed package is no longer at_warehouse → excluded by the live query
        await _package(db, PackageStatus.in_transit_w2w, current_branch_id=branch)

        r = await client.get(f"/warehouse/{branch}/inventory")

        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["packages"][0]["package_id"] == str(active_pkg)

    async def test_empty_warehouse_returns_zero(self, client, db):
        branch = await _branch(db, "Empty WH")

        r = await client.get(f"/warehouse/{branch}/inventory")

        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 0
        assert body["packages"] == []

    async def test_branch_not_found_returns_404(self, client, db):
        r = await client.get(f"/warehouse/{ObjectId()}/inventory")
        assert r.status_code == 404

    async def test_response_includes_expected_fields(self, client, db):
        branch = await _branch(db, "Field Check WH")
        dest = await _branch(db, "Dest Branch")
        await _package(
            db,
            PackageStatus.at_warehouse,
            current_branch_id=branch,
            destination_branch_id=dest,
            in_warehouse=True,
        )

        r = await client.get(f"/warehouse/{branch}/inventory")

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
    async def test_origin_transit_warehouse_delivery(self, client, db):
        """
        at_origin_branch
          → arrive WH-A (no email, not destination)
          → depart to WH-B (in_transit_w2w, travel time returned)
          → arrive WH-B (email fires, is destination)
          → depart final leg (out_for_delivery)
          → both warehouse inventories empty
        """
        wh_a = await _branch(db, "Warehouse A", lat=11.55, lng=104.92)
        wh_b = await _branch(db, "Warehouse B", lat=12.10, lng=105.50)

        pkg = await _package(
            db,
            PackageStatus.at_origin_branch,
            destination_branch_id=wh_b,
            customer_email="receiver@example.com",
        )

        # Step 1 — arrive at WH-A (transit, no email)
        with patch("routes.warehouse.trigger_warehouse_arrival_email", new_callable=AsyncMock) as mock_email:
            r = await client.post(f"/warehouse/{pkg}/arrive", json={"branch_id": str(wh_a)})
        assert r.status_code == 200
        assert r.json()["is_destination_warehouse"] is False
        mock_email.assert_not_called()

        # Step 2 — depart WH-A to WH-B, 100 km → 240 min
        with patch("routes.warehouse._osrm_route", return_value={"distance": 100_000}):
            r = await client.post(f"/warehouse/{pkg}/depart", json={"to_branch_id": str(wh_b)})
        assert r.status_code == 200
        assert r.json()["status"] == "in_transit_w2w"
        assert r.json()["estimated_travel_min"] == 240.0

        # WH-A inventory now empty
        assert (await client.get(f"/warehouse/{wh_a}/inventory")).json()["count"] == 0

        # Step 3 — arrive at WH-B (destination → email)
        with patch("routes.warehouse.trigger_warehouse_arrival_email", new_callable=AsyncMock) as mock_email:
            r = await client.post(f"/warehouse/{pkg}/arrive", json={"branch_id": str(wh_b)})
        assert r.status_code == 200
        assert r.json()["is_destination_warehouse"] is True
        assert r.json()["confirmation_email_sent"] is True
        mock_email.assert_awaited_once()

        # WH-B inventory now has 1 package
        assert (await client.get(f"/warehouse/{wh_b}/inventory")).json()["count"] == 1

        # Step 4 — final dispatch
        r = await client.post(f"/warehouse/{pkg}/depart", json={})
        assert r.status_code == 200
        assert r.json()["status"] == "out_for_delivery"

        # Both warehouses empty
        assert (await client.get(f"/warehouse/{wh_a}/inventory")).json()["count"] == 0
        assert (await client.get(f"/warehouse/{wh_b}/inventory")).json()["count"] == 0

        assert (await _get_package(db, pkg))["status"] == PackageStatus.out_for_delivery.value
