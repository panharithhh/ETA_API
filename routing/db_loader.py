"""
Database loader for the routing module.

Reads branch coordinates and pending deliveries from the live DB.
Works with the tables that actually exist:
  - routing_branches  (created here if absent; standalone, no FK deps)
  - branches          (first-mile SQLAlchemy table, used if it exists)
  - delivery_log      (legacy ETA table — pickup=accept_gps, dropoff=delivery_gps)
  - c2c_log           (C2C table — pickup=pickup_lat/lon, dropoff=delivery_lat/lon)

Call setup_routing_tables(conn) once at app startup (safe to call repeatedly).
"""

import psycopg2
import psycopg2.extras

from routing.warehouse_assignment import Branch


# ── Schema setup ──────────────────────────────────────────────────────────────

def setup_routing_tables(conn) -> None:
    """
    Create routing_branches if it doesn't exist yet.
    Safe to call on every startup (IF NOT EXISTS).

    Seed example branches for Phnom Penh — delete/replace these rows with
    your real warehouse locations.
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS routing_branches (
                id       SERIAL PRIMARY KEY,
                name     VARCHAR(100) NOT NULL,
                lat      DOUBLE PRECISION NOT NULL,
                lng      DOUBLE PRECISION NOT NULL,
                address  VARCHAR(255)
            )
        """)
        # Only seed if empty so reruns don't duplicate
        cur.execute("SELECT COUNT(*) FROM routing_branches")
        if cur.fetchone()[0] == 0:
            # ← Replace with your actual warehouse coordinates
            cur.executemany(
                "INSERT INTO routing_branches (name, lat, lng, address) VALUES (%s, %s, %s, %s)",
                [
                    ("Central Warehouse",   11.5625, 104.9160, "Phnom Penh Central"),
                    ("South Branch",        11.5220, 104.8850, "South Phnom Penh"),
                    ("North Branch",        11.5900, 104.9100, "North Phnom Penh"),
                ],
            )
    conn.commit()


# ── Branch loader ─────────────────────────────────────────────────────────────

def _table_exists(cur, name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=%s",
        (name,),
    )
    return cur.fetchone() is not None


def load_branches(conn) -> list[Branch]:
    """
    Load warehouse coordinates.

    Prefers the first-mile 'branches' table if it exists (migration 001 applied).
    Falls back to 'routing_branches'.
    Raises RuntimeError if neither table is present.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if _table_exists(cur, "branches"):
            cur.execute("SELECT id, name, lat, lng FROM branches ORDER BY id")
        elif _table_exists(cur, "routing_branches"):
            cur.execute("SELECT id, name, lat, lng FROM routing_branches ORDER BY id")
        else:
            raise RuntimeError(
                "No branch table found. Run setup_routing_tables(conn) first."
            )
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError("Branch table is empty — add warehouse coordinates first.")

    return [Branch(id=r["id"], name=r["name"], lat=r["lat"], lng=r["lng"]) for r in rows]


# ── Delivery loader ───────────────────────────────────────────────────────────

def load_pending_deliveries(conn, include_completed: bool = False) -> list[dict]:
    """
    Load deliveries that need routing.

    Returns a unified list of dicts with keys:
      order_id, courier_id, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
      accept_time, source ('delivery_log' or 'c2c_log')

    By default only rows with delivery_time IS NULL (not yet completed).
    Pass include_completed=True to load all rows (useful in dev/test when
    no pending rows exist yet).
    """
    deliveries: list[dict] = []
    where = "" if include_completed else "WHERE delivery_time IS NULL"

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if _table_exists(cur, "delivery_log"):
            cur.execute(f"""
                SELECT
                    order_id,
                    courier_id,
                    accept_gps_lat  AS pickup_lat,
                    accept_gps_lng  AS pickup_lng,
                    delivery_gps_lat AS dropoff_lat,
                    delivery_gps_lng AS dropoff_lng,
                    accept_time,
                    'delivery_log'  AS source
                FROM delivery_log
                {where}
                ORDER BY accept_time
            """)
            for row in cur.fetchall():
                if _valid_coords(row):
                    deliveries.append(dict(row))

        if _table_exists(cur, "c2c_log"):
            cur.execute(f"""
                SELECT
                    order_id,
                    NULL            AS courier_id,
                    pickup_lat,
                    pickup_lon      AS pickup_lng,
                    delivery_lat    AS dropoff_lat,
                    delivery_lon    AS dropoff_lng,
                    accept_time,
                    'c2c_log'       AS source
                FROM c2c_log
                {where}
                ORDER BY accept_time
            """)
            for row in cur.fetchall():
                if _valid_coords(row):
                    deliveries.append(dict(row))

        # Also check first-mile packages table if migrations have been applied
        if _table_exists(cur, "packages"):
            status_filter = (
                "" if include_completed
                else "WHERE status IN ('at_origin_branch', 'arrived_at_warehouse')"
            )
            cur.execute(f"""
                SELECT
                    p.id            AS order_id,
                    p.assigned_driver_id AS courier_id,
                    b_orig.lat      AS pickup_lat,
                    b_orig.lng      AS pickup_lng,
                    p.receiver_lat  AS dropoff_lat,
                    p.receiver_lng  AS dropoff_lng,
                    p.created_at    AS accept_time,
                    'packages'      AS source
                FROM packages p
                LEFT JOIN branches b_orig ON b_orig.id = p.origin_branch_id
                {status_filter}
                ORDER BY p.created_at
            """)
            for row in cur.fetchall():
                if _valid_coords(row):
                    deliveries.append(dict(row))

    return deliveries


def _valid_coords(row: dict) -> bool:
    """Drop rows where any coordinate is NULL or clearly invalid."""
    for key in ("pickup_lat", "pickup_lng", "dropoff_lat", "dropoff_lng"):
        v = row.get(key)
        if v is None:
            return False
        if not (-90 <= float(v) <= 90 if "lat" in key else -180 <= float(v) <= 180):
            return False
    return True
