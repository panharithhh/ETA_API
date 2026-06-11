"""
Database loader for the routing module (MongoDB / Motor).

Reads branch coordinates and pending deliveries from the live DB.
Works with the collections that actually exist:
  - branches      (first-mile collection — { _id, name, lat, lng })
  - delivery_log  (legacy ETA collection — pickup=accept_gps, dropoff=delivery_gps)
  - c2c_log       (C2C collection — pickup=pickup_lat/lon, dropoff=delivery_lat/lon)
  - packages      (first-mile collection — origin branch → receiver)

All loaders are async and take the Motor database handle (`from db import get_db`).
"""

from routing.warehouse_assignment import Branch

# Phnom Penh seed branches — used only by setup_routing_tables when `branches`
# is empty. Replace with your real warehouse locations.
_SEED_BRANCHES = [
    {"name": "Central Warehouse", "lat": 11.5625, "lng": 104.9160, "address": "Phnom Penh Central"},
    {"name": "South Branch", "lat": 11.5220, "lng": 104.8850, "address": "South Phnom Penh"},
    {"name": "North Branch", "lat": 11.5900, "lng": 104.9100, "address": "North Phnom Penh"},
]


# ── Schema setup ──────────────────────────────────────────────────────────────

async def setup_routing_tables(db) -> None:
    """Seed example branches if the collection is empty. Idempotent."""
    if await db.branches.count_documents({}) == 0:
        await db.branches.insert_many([dict(b) for b in _SEED_BRANCHES])


# ── Branch loader ─────────────────────────────────────────────────────────────

async def load_branches(db) -> list[Branch]:
    """Load warehouse coordinates from the `branches` collection."""
    rows = await db.branches.find().sort("_id", 1).to_list(length=None)
    if not rows:
        raise RuntimeError("Branch collection is empty — run setup_routing_tables(db) first.")
    
    branches = []
    for r in rows:
        lat = r.get("lat")
        if lat is None:
            lat = r.get("latitude")
        
        lng = r.get("lng")
        if lng is None:
            lng = r.get("longitude")
        if lng is None:
            lng = r.get("lon")
            
        if lat is not None and lng is not None:
            branches.append(Branch(
                id=str(r["_id"]),
                name=r.get("name", "Unknown"),
                lat=float(lat),
                lng=float(lng)
            ))
            
    if not branches:
        raise RuntimeError("No branches with valid coordinates found.")
        
    return branches


# ── Delivery loader ───────────────────────────────────────────────────────────

async def load_pending_deliveries(db, include_completed: bool = False) -> list[dict]:
    """
    Load deliveries that need routing.

    Returns a unified list of dicts with keys:
      order_id, courier_id, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
      accept_time, source ('delivery_log', 'c2c_log' or 'packages')

    By default only rows with delivery_time missing (not yet completed).
    """
    deliveries: list[dict] = []
    incomplete = {"$or": [{"delivery_time": None}, {"delivery_time": {"$exists": False}}]}

    # delivery_log
    query = {} if include_completed else incomplete
    async for row in db.delivery_log.find(query).sort("accept_time", 1):
        rec = {
            "order_id": row.get("order_id"),
            "courier_id": row.get("courier_id"),
            "pickup_lat": row.get("accept_gps_lat"),
            "pickup_lng": row.get("accept_gps_lng"),
            "dropoff_lat": row.get("delivery_gps_lat"),
            "dropoff_lng": row.get("delivery_gps_lng"),
            "accept_time": row.get("accept_time"),
            "source": "delivery_log",
        }
        if _valid_coords(rec):
            deliveries.append(rec)

    # c2c_log
    async for row in db.c2c_log.find(query).sort("accept_time", 1):
        rec = {
            "order_id": row.get("order_id"),
            "courier_id": None,
            "pickup_lat": row.get("pickup_lat"),
            "pickup_lng": row.get("pickup_lon"),
            "dropoff_lat": row.get("delivery_lat"),
            "dropoff_lng": row.get("delivery_lon"),
            "accept_time": row.get("accept_time"),
            "source": "c2c_log",
        }
        if _valid_coords(rec):
            deliveries.append(rec)

    # packages — join origin branch coords for the pickup point
    branches = {b["_id"]: b async for b in db.branches.find()}
    pkg_query = (
        {} if include_completed
        else {"status": {"$in": ["at_origin_branch", "arrived_at_warehouse"]}}
    )
    async for p in db.packages.find(pkg_query).sort("created_at", 1):
        origin = branches.get(p.get("origin_branch_id"))
        
        pickup_lat, pickup_lng = None, None
        if origin:
            pickup_lat = origin.get("lat")
            if pickup_lat is None:
                pickup_lat = origin.get("latitude")
                
            pickup_lng = origin.get("lng")
            if pickup_lng is None:
                pickup_lng = origin.get("longitude")
            if pickup_lng is None:
                pickup_lng = origin.get("lon")

        rec = {
            "order_id": str(p["_id"]),
            "courier_id": p.get("assigned_driver_id"),
            "pickup_lat": pickup_lat,
            "pickup_lng": pickup_lng,
            "dropoff_lat": p.get("receiver_lat"),
            "dropoff_lng": p.get("receiver_lng"),
            "accept_time": p.get("created_at"),
            "source": "packages",
        }
        if _valid_coords(rec):
            deliveries.append(rec)

    return deliveries


def _valid_coords(row: dict) -> bool:
    """Drop rows where any coordinate is None or clearly invalid."""
    for key in ("pickup_lat", "pickup_lng", "dropoff_lat", "dropoff_lng"):
        v = row.get(key)
        if v is None:
            return False
        if not (-90 <= float(v) <= 90 if "lat" in key else -180 <= float(v) <= 180):
            return False
    return True
