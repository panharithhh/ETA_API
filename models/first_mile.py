"""
MongoDB data layer for the first-mile / mid-mile domain.

There is no ORM — documents are plain dicts. This module holds the shared
enums, the collection names, and small helpers for ObjectId handling and
document serialization.

Collections
-----------
branches            { _id, name, lat, lng }
vehicles            { _id, vehicle_type, license_plate, is_available }
drivers             { _id, name, phone, current_lat, current_lng,
                      is_available, vehicle_id }
packages            { _id, customer_phone, customer_email, receiver_phone,
                      receiver_lat, receiver_lng, weight_kg, max_dimension_cm,
                      status, origin_branch_id, destination_branch_id,
                      current_branch_id, assigned_driver_id, assigned_vehicle_id,
                      pickup_window_start, pickup_window_end, created_at,
                      warehouse_arrived_at }
package_events      { _id, package_id, status, driver_id, branch_id, notes,
                      created_at }                       # append-only audit log
confirmation_tokens { _id, package_id, token, action, expires_at, used,
                      created_at }

Warehouse inventory is NOT a collection — "packages at a warehouse" is a live
query over packages where status == at_warehouse and current_branch_id == branch.
"""

import enum

from bson import ObjectId
from bson.errors import InvalidId


# ── Collection names ──────────────────────────────────────────────────────────

BRANCHES = "branches"
VEHICLES = "vehicles"
DRIVERS = "drivers"
PACKAGES = "packages"
PACKAGE_EVENTS = "package_events"
CONFIRMATION_TOKENS = "confirmation_tokens"


# ── Enums (stored as plain strings) ───────────────────────────────────────────

class VehicleType(str, enum.Enum):
    motorbike = "motorbike"
    tuktuk = "tuktuk"
    van = "van"
    container = "container"


class PackageStatus(str, enum.Enum):
    created = "created"
    pending_pickup = "pending_pickup"
    picked_up = "picked_up"
    at_origin_branch = "at_origin_branch"
    arrived_at_warehouse = "arrived_at_warehouse"
    confirmed = "confirmed"
    rejected = "rejected"
    # Mid-mile statuses
    at_warehouse = "at_warehouse"
    in_transit_w2w = "in_transit_w2w"
    out_for_delivery = "out_for_delivery"


class ConfirmationAction(str, enum.Enum):
    confirm = "confirm"
    reject = "reject"


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_object_id(value) -> ObjectId | None:
    """Parse a value into an ObjectId, returning None when it is not a valid id.

    Lets routes turn a bad path param straight into a 404 instead of a 500.
    """
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def serialize(doc: dict | None) -> dict | None:
    """Convert a Mongo document for JSON output: ObjectId fields → str."""
    if doc is None:
        return None
    out = dict(doc)
    for key, value in out.items():
        if isinstance(value, ObjectId):
            out[key] = str(value)
    return out
