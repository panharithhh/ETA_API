import os

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConfigurationError

# ── Connection ────────────────────────────────────────────────────────────────
# MONGODB_URI (preferred) or legacy MONGO_URI. The URI carries the default db
# name (…/chonhchoun); fall back to a local mongod + "chonchoun" db otherwise.
_MONGO_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI", "mongodb://localhost:27017")
_DEFAULT_DB_NAME = "chonchoun"

_client = AsyncIOMotorClient(_MONGO_URI)

try:
    _db: AsyncIOMotorDatabase = _client.get_default_database()
except ConfigurationError:  # URI carried no default db name
    _db = None
if _db is None:
    _db = _client[_DEFAULT_DB_NAME]


def get_db() -> AsyncIOMotorDatabase:
    """FastAPI dependency — returns the shared Motor database handle.

    Tests override this via app.dependency_overrides[get_db].
    """
    return _db


async def init_db() -> None:
    """Create indexes. Idempotent — safe to run on every startup."""
    await _db.predictions.create_index("order_id", unique=True)
    await _db.delivery_log.create_index("order_id")
    await _db.c2c_log.create_index("order_id", unique=True)
    await _db.confirmation_tokens.create_index("token", unique=True)
    await _db.packages.create_index("status")
    await _db.packages.create_index([("status", 1), ("current_branch_id", 1)])
    await _db.package_events.create_index("package_id")
    await _db.drivers.create_index("vehicle_id")
