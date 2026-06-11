"""
Admin API — everything the Next.js admin dashboard drives.

Two groups of endpoints:

  Read / dashboard   GET  /admin/overview          aggregate counts
                     GET  /admin/branches          list branches
                     GET  /admin/drivers           list drivers (+ vehicle)
                     GET  /admin/vehicles          list vehicles
                     GET  /admin/packages          list packages (filter by status)
                     GET  /admin/inventory         live inventory across all warehouses
                     GET  /admin/packages/{id}/timeline   audit log for one package

  Seed / write       POST /admin/branches          create a branch
                     POST /admin/drivers           create a driver
                     POST /admin/vehicles          create a vehicle

  Auto routing       POST /admin/routing/auto      run the OSRM + OR-Tools VRP pipeline
                                                    and return an optimized route per driver

The routing endpoint is the HTTP front door for the routing/ package
(warehouse assignment → distance matrix → VRPPD solver).
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field

from db import get_db
from models.first_mile import (
    BRANCHES,
    DRIVERS,
    PACKAGE_EVENTS,
    PACKAGES,
    VEHICLES,
    PackageStatus,
    VehicleType,
    serialize,
    to_object_id,
)
from routing import (
    PickupDeliveryPair,
    assign_warehouses,
    build_distance_matrix,
    build_vrp_locations,
    load_branches,
    load_pending_deliveries,
    solve_vrppd,
)

router = APIRouter(prefix="/admin", tags=["Admin — Dashboard & Routing"])


# ── Dashboard / read ──────────────────────────────────────────────────────────

@router.get("/overview")
async def overview(db: AsyncIOMotorDatabase = Depends(get_db)):
    """Top-line counts for the dashboard landing page."""
    status_rows = await db[PACKAGES].aggregate(
        [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
    ).to_list(length=None)
    packages_by_status = {row["_id"]: row["count"] for row in status_rows}

    return {
        "branches": await db[BRANCHES].count_documents({}),
        "drivers": await db[DRIVERS].count_documents({}),
        "drivers_available": await db[DRIVERS].count_documents({"is_available": True}),
        "vehicles": await db[VEHICLES].count_documents({}),
        "packages_total": await db[PACKAGES].count_documents({}),
        "packages_by_status": packages_by_status,
        "packages_at_warehouse": await db[PACKAGES].count_documents(
            {"status": PackageStatus.at_warehouse.value}
        ),
    }


@router.get("/branches")
async def list_branches(db: AsyncIOMotorDatabase = Depends(get_db)):
    rows = await db[BRANCHES].find().to_list(length=None)
    return {"count": len(rows), "branches": [serialize(r) for r in rows]}


@router.get("/drivers")
async def list_drivers(db: AsyncIOMotorDatabase = Depends(get_db)):
    rows = await db[DRIVERS].find().to_list(length=None)
    # Join each driver's vehicle so the dashboard can show type/plate
    vehicles = {v["_id"]: v async for v in db[VEHICLES].find()}
    out = []
    for r in rows:
        d = serialize(r)
        veh = vehicles.get(r.get("vehicle_id"))
        d["vehicle"] = serialize(veh) if veh else None
        out.append(d)
    return {"count": len(out), "drivers": out}


@router.get("/vehicles")
async def list_vehicles(db: AsyncIOMotorDatabase = Depends(get_db)):
    rows = await db[VEHICLES].find().to_list(length=None)
    return {"count": len(rows), "vehicles": [serialize(r) for r in rows]}


@router.get("/packages")
async def list_packages(
    status: Optional[PackageStatus] = Query(None, description="Filter by package status"),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    query = {"status": status.value} if status else {}
    rows = await db[PACKAGES].find(query).sort("created_at", -1).to_list(length=limit)
    return {"count": len(rows), "packages": [serialize(r) for r in rows]}


@router.get("/inventory")
async def all_inventory(db: AsyncIOMotorDatabase = Depends(get_db)):
    """Live inventory grouped by warehouse — every package sitting at a branch now."""
    branches = {b["_id"]: b async for b in db[BRANCHES].find()}
    rows = await db[PACKAGES].find({"status": PackageStatus.at_warehouse.value}).to_list(length=None)

    grouped: dict = {}
    for p in rows:
        bid = p.get("current_branch_id")
        bucket = grouped.setdefault(str(bid), {
            "branch_id": str(bid) if bid else None,
            "branch_name": branches.get(bid, {}).get("name", "Unknown"),
            "count": 0,
            "packages": [],
        })
        bucket["count"] += 1
        bucket["packages"].append({
            "package_id": str(p["_id"]),
            "arrived_at": p["warehouse_arrived_at"].isoformat() if p.get("warehouse_arrived_at") else None,
            "destination_branch_id": str(p["destination_branch_id"]) if p.get("destination_branch_id") else None,
        })

    return {"warehouses": list(grouped.values()), "total_at_warehouse": len(rows)}


@router.get("/packages/{package_id}/timeline")
async def package_timeline(package_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    pkg_oid = to_object_id(package_id)
    package = await db[PACKAGES].find_one({"_id": pkg_oid}) if pkg_oid else None
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")

    events = await db[PACKAGE_EVENTS].find({"package_id": pkg_oid}).sort("created_at", 1).to_list(length=None)
    return {
        "package": serialize(package),
        "events": [
            {
                "status": e.get("status"),
                "branch_id": str(e["branch_id"]) if e.get("branch_id") else None,
                "driver_id": str(e["driver_id"]) if e.get("driver_id") else None,
                "notes": e.get("notes"),
                "created_at": e["created_at"].isoformat() if e.get("created_at") else None,
            }
            for e in events
        ],
    }


# ── Seed / write ──────────────────────────────────────────────────────────────

class BranchCreate(BaseModel):
    name: str
    lat: float
    lng: float


class VehicleCreate(BaseModel):
    vehicle_type: VehicleType
    license_plate: str
    is_available: bool = True


class DriverCreate(BaseModel):
    name: str
    phone: str
    current_lat: Optional[float] = None
    current_lng: Optional[float] = None
    is_available: bool = True
    vehicle_id: Optional[str] = None


def _created(doc: dict) -> dict:
    """Shape an inserted doc for the response: _id (ObjectId) → id (str)."""
    out = serialize(doc)
    out["id"] = out.pop("_id")
    return out


@router.post("/branches", status_code=201)
async def create_branch(data: BranchCreate, db: AsyncIOMotorDatabase = Depends(get_db)):
    doc = data.model_dump()
    await db[BRANCHES].insert_one(doc)  # insert_one injects _id into doc
    return _created(doc)


@router.post("/vehicles", status_code=201)
async def create_vehicle(data: VehicleCreate, db: AsyncIOMotorDatabase = Depends(get_db)):
    doc = data.model_dump()
    doc["vehicle_type"] = data.vehicle_type.value
    await db[VEHICLES].insert_one(doc)
    return _created(doc)


@router.post("/drivers", status_code=201)
async def create_driver(data: DriverCreate, db: AsyncIOMotorDatabase = Depends(get_db)):
    doc = data.model_dump()
    doc["vehicle_id"] = to_object_id(data.vehicle_id) if data.vehicle_id else None
    await db[DRIVERS].insert_one(doc)
    return _created(doc)


# ── Auto routing ──────────────────────────────────────────────────────────────

class AutoRouteRequest(BaseModel):
    depot_branch_id: Optional[str] = Field(
        None, description="Branch the drivers start/end at. Defaults to the first branch."
    )
    num_drivers: int = Field(3, ge=1, le=50, description="Number of available vehicles to route")
    include_completed: bool = Field(
        False, description="Route every delivery (dev/demo) instead of only pending ones"
    )
    time_limit_s: int = Field(30, ge=1, le=120, description="OR-Tools solver budget")


@router.post("/routing/auto")
async def auto_routing(req: AutoRouteRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    """
    Run the full mid-mile routing pipeline and return an optimized route per driver:

      1. load branches + pending deliveries from Mongo
      2. assign each delivery's pickup/dropoff to its nearest branch (OSRM)
      3. build an OSRM travel-time matrix over all stops
      4. solve the pickup-and-delivery VRP with OR-Tools

    Requires at least one branch and one pending delivery. The OSRM steps hit
    the configured OSRM server (OSRM_BASE_URL), so a 502 here means OSRM was
    unreachable, not a bug in the pipeline.
    """
    try:
        branches = await load_branches(db)
    except RuntimeError:
        raise HTTPException(status_code=409, detail="No branches configured — create one first")

    depot_id = req.depot_branch_id or branches[0].id
    if depot_id not in {b.id for b in branches}:
        raise HTTPException(status_code=404, detail=f"Depot branch {depot_id} not found")

    deliveries = await load_pending_deliveries(db, include_completed=req.include_completed)
    if not deliveries:
        return {
            "is_feasible": True,
            "routes": [],
            "unassigned_order_ids": [],
            "message": "No deliveries to route",
        }

    # OSRM + OR-Tools are blocking/CPU-bound — keep them off the event loop.
    def _solve():
        assignments = assign_warehouses(deliveries, branches)
        locations, pair_specs = build_vrp_locations(assignments, branches, depot_branch_id=depot_id)
        matrix = build_distance_matrix(locations)
        pairs = [PickupDeliveryPair(**p) for p in pair_specs]
        solution = solve_vrppd(
            matrix, num_drivers=req.num_drivers, depot_node=0,
            pairs=pairs, time_limit_s=req.time_limit_s,
        )
        return assignments, locations, solution

    try:
        assignments, locations, solution = await run_in_threadpool(_solve)
    except Exception as exc:  # OSRM down, bad coords, etc.
        raise HTTPException(status_code=502, detail=f"Routing failed: {exc}")

    return {
        "is_feasible": solution.is_feasible,
        "depot_branch_id": depot_id,
        "num_drivers": req.num_drivers,
        "total_time_min": round(solution.total_time_s / 60, 1),
        "unassigned_order_ids": [str(o) for o in solution.unassigned_order_ids],
        "routes": [
            {
                "driver_index": r.driver_index,
                "total_time_min": round(r.total_time_s / 60, 1),
                "stops": [
                    {"node": node, "lat": locations[node][0], "lng": locations[node][1]}
                    for node in r.stop_nodes
                ],
            }
            for r in solution.routes
        ],
        "assignments": [
            {
                "order_id": str(a.order_id),
                "pickup_branch": a.pickup_branch.name,
                "dropoff_branch": a.dropoff_branch.name,
                "needs_transfer": a.needs_transfer,
            }
            for a in assignments
        ],
    }
