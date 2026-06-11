"""
Admin endpoint tests — dashboard reads, seed writes, and the auto-routing
contract. OSRM + OR-Tools are mocked so the suite stays offline.

Run with: pytest tests/test_admin.py -v
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest_asyncio
from bson import ObjectId
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from db import get_db
from models.first_mile import BRANCHES, PACKAGE_EVENTS, PACKAGES, PackageStatus
from routes.admin import router as admin_router
from routing.vrp_solver import DriverRoute, RoutingSolution
from routing.warehouse_assignment import Branch, DeliveryAssignment


@pytest_asyncio.fixture
async def client(db):
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Seed + read ───────────────────────────────────────────────────────────────

async def test_create_and_list_branch(client):
    r = await client.post("/admin/branches", json={"name": "WH1", "lat": 11.5, "lng": 104.9})
    assert r.status_code == 201
    bid = r.json()["id"]

    r = await client.get("/admin/branches")
    assert r.json()["count"] == 1
    assert r.json()["branches"][0]["_id"] == bid


async def test_create_driver_with_vehicle_join(client):
    v = await client.post(
        "/admin/vehicles",
        json={"vehicle_type": "motorbike", "license_plate": "PP-1"},
    )
    vid = v.json()["id"]
    await client.post(
        "/admin/drivers",
        json={"name": "Sok", "phone": "0123", "vehicle_id": vid,
              "current_lat": 11.5, "current_lng": 104.9},
    )

    r = await client.get("/admin/drivers")
    body = r.json()
    assert body["count"] == 1
    assert body["drivers"][0]["vehicle"]["license_plate"] == "PP-1"


async def test_overview_counts(client, db):
    await db[BRANCHES].insert_one({"name": "B", "lat": 1, "lng": 2})
    await db[PACKAGES].insert_many([
        {"status": PackageStatus.at_warehouse.value},
        {"status": PackageStatus.at_warehouse.value},
        {"status": PackageStatus.created.value},
    ])

    body = (await client.get("/admin/overview")).json()
    assert body["branches"] == 1
    assert body["packages_total"] == 3
    assert body["packages_at_warehouse"] == 2
    assert body["packages_by_status"]["at_warehouse"] == 2


async def test_inventory_grouped_by_warehouse(client, db):
    b1 = (await db[BRANCHES].insert_one({"name": "WH-A", "lat": 1, "lng": 2})).inserted_id
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    await db[PACKAGES].insert_one({
        "status": PackageStatus.at_warehouse.value,
        "current_branch_id": b1,
        "warehouse_arrived_at": now,
    })

    body = (await client.get("/admin/inventory")).json()
    assert body["total_at_warehouse"] == 1
    assert body["warehouses"][0]["branch_name"] == "WH-A"
    assert body["warehouses"][0]["count"] == 1


async def test_package_timeline(client, db):
    pkg = (await db[PACKAGES].insert_one({"status": PackageStatus.created.value})).inserted_id
    await db[PACKAGE_EVENTS].insert_one({
        "package_id": pkg,
        "status": PackageStatus.created.value,
        "notes": "booked",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
    })

    body = (await client.get(f"/admin/packages/{pkg}/timeline")).json()
    assert len(body["events"]) == 1
    assert body["events"][0]["notes"] == "booked"


async def test_timeline_404_for_missing_package(client):
    r = await client.get(f"/admin/packages/{ObjectId()}/timeline")
    assert r.status_code == 404


# ── Auto routing ──────────────────────────────────────────────────────────────

async def test_routing_no_deliveries_returns_empty(client, db):
    await db[BRANCHES].insert_one({"name": "Depot", "lat": 11.5, "lng": 104.9})
    r = await client.post("/admin/routing/auto", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["routes"] == []
    assert "No deliveries" in body["message"]


async def test_routing_requires_a_branch(client):
    r = await client.post("/admin/routing/auto", json={})
    assert r.status_code == 409


async def test_routing_pipeline_response_shape(client, db):
    """The OSRM/OR-Tools steps are mocked; we assert the JSON the dashboard gets."""
    depot = (await db[BRANCHES].insert_one({"name": "Depot", "lat": 11.5, "lng": 104.9})).inserted_id
    # One delivery so load_pending_deliveries returns a row
    await db.delivery_log.insert_one({
        "order_id": 1, "courier_id": "c1",
        "accept_gps_lat": 11.50, "accept_gps_lng": 104.90,
        "delivery_gps_lat": 11.55, "delivery_gps_lng": 104.95,
        "accept_time": "2026-01-01 09:00:00",
    })

    depot_branch = Branch(id=str(depot), name="Depot", lat=11.5, lng=104.9)
    fake_assignment = DeliveryAssignment(
        order_id=1, pickup_lat=11.50, pickup_lng=104.90,
        dropoff_lat=11.55, dropoff_lng=104.95,
        pickup_branch=depot_branch, dropoff_branch=depot_branch, needs_transfer=False,
    )
    fake_locations = [(11.5, 104.9), (11.55, 104.95)]
    fake_solution = RoutingSolution(
        routes=[DriverRoute(driver_index=0, stop_nodes=[0, 1, 0], total_time_s=600.0)],
        is_feasible=True, total_time_s=600.0,
    )

    with patch("routes.admin.assign_warehouses", return_value=[fake_assignment]), \
         patch("routes.admin.build_vrp_locations", return_value=(fake_locations, [{"order_id": 1, "pickup_node": 0, "delivery_node": 1}])), \
         patch("routes.admin.build_distance_matrix", return_value=[[0, 1], [1, 0]]), \
         patch("routes.admin.solve_vrppd", return_value=fake_solution):
        r = await client.post("/admin/routing/auto", json={"num_drivers": 1})

    assert r.status_code == 200
    body = r.json()
    assert body["is_feasible"] is True
    assert body["total_time_min"] == 10.0
    assert len(body["routes"]) == 1
    assert body["routes"][0]["stops"][1] == {"node": 1, "lat": 11.55, "lng": 104.95}
    assert body["assignments"][0]["order_id"] == "1"
    assert body["assignments"][0]["needs_transfer"] is False
