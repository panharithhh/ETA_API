# Chonchoun Courier API — Architecture

FastAPI backend for a courier/logistics platform. Data lives in **MongoDB**
(async **Motor** driver). The Next.js admin app talks to this service over HTTP.

The system covers four domains:

| Domain | What it does | Router |
| --- | --- | --- |
| **Delivery ETA** | ML model predicts last-mile ETAs, greedy driver↔stop auto-mapping | `routes/delivery.py` |
| **C2C** | Customer-to-customer ETA prediction + logging | `routes/c2c.py` |
| **First mile** | Book a pickup, auto-assign a driver/vehicle, scan, drop at branch | `routes/pickups.py` |
| **Mid mile** | Warehouse arrive/depart, live inventory, customer confirmation emails | `routes/warehouse.py`, `routes/confirmation.py` |
| **Admin** | Dashboard reads, seed data, **auto-routing (VRP)** | `routes/admin.py` |

---

## Repo map — what each file is

```
api.py                     App entry. Builds FastAPI, runs init_db() on startup, mounts all routers.
db.py                      MongoDB connection (Motor). get_db() dependency + init_db() index creation.
api_auth.py                X-API-Key header check for protected (train/retrain/auto-mapping) endpoints.
schema.py                  Pydantic request models for delivery + c2c endpoints.

models/
  first_mile.py            NOT an ORM. Enums (PackageStatus, VehicleType, ConfirmationAction),
                           collection-name constants, and helpers: to_object_id(), serialize().

schemas/
  pickups.py               Pydantic request models for the pickups endpoints.

routes/
  delivery.py              /delivery/*  — ETA predict, accept, train, retrain, auto-mapping.
  c2c.py                   /c2c/*       — predict, active, confirm, retrain.
  pickups.py               /pickups/*   — book, scan, dropoff (writes packages + package_events).
  warehouse.py             /warehouse/* — arrive, depart, inventory (live query).
  confirmation.py          /confirmation/* — customer confirm/reject links (HTML pages).
  admin.py                 /admin/*     — dashboard, seed CRUD, auto-routing entry point.

services/
  first_mile.py            assign_pickup(): driver/vehicle selection by tier, radius, cost, rush hour.
  warehouse_confirmation.py  Issues single-use confirm/reject tokens + sends the email.
  email_service.py         SMTP send (warehouse arrival email).

routing/                   Pure mid-mile routing engine (no HTTP layer of its own).
  db_loader.py             Loads branches + pending deliveries from Mongo.
  warehouse_assignment.py  Assigns each delivery's pickup/dropoff to nearest branch (OSRM).
  distance_matrix.py       Builds N×N travel-time matrix via OSRM (batched).
  osrm_client.py           OSRM HTTP client (/table and /route).
  vrp_solver.py            OR-Tools pickup-and-delivery VRP solver.

model.py, c2c.py, retrain.py   ML models (delivery + c2c) and retraining. DB-free except retrain reads logs.

utils/geo.py               haversine() distance helper.

tests/                     pytest suite — runs fully offline against mongomock_motor.
```

> Removed in the Postgres→Mongo migration: `database.py`, `alembic/`, `alembic.ini`
> (Mongo is schemaless — no migrations).

---

## Data model (MongoDB collections)

IDs are `ObjectId`. In the API they appear as 24-char hex **strings**.
`order_id` (delivery/c2c) stays an **int** business key — it is not the Mongo `_id`.

```
branches            { _id, name, lat, lng }
vehicles            { _id, vehicle_type, license_plate, is_available }
drivers             { _id, name, phone, current_lat, current_lng, is_available, vehicle_id }
packages            { _id, customer_phone, customer_email, receiver_phone,
                      receiver_lat, receiver_lng, weight_kg, max_dimension_cm,
                      status, origin_branch_id, destination_branch_id, current_branch_id,
                      assigned_driver_id, assigned_vehicle_id,
                      pickup_window_start, pickup_window_end, created_at, warehouse_arrived_at }
package_events      { _id, package_id, status, driver_id, branch_id, notes, created_at }   # append-only audit log
confirmation_tokens { _id, package_id, token, action, expires_at, used, created_at }
predictions         { order_id, ... , delivery_time }        # delivery ETA log
delivery_log        { order_id, courier_id, ... }            # completed deliveries
c2c_log             { order_id, ... }                        # completed c2c trips
```

**Package status flow:** `created → pending_pickup → picked_up → at_origin_branch
→ at_warehouse ⇄ in_transit_w2w → at_warehouse(destination) → out_for_delivery`,
plus `confirmed` / `rejected` from the customer email link.

**Warehouse inventory is not a collection** — "packages at a warehouse" is a live
query: `packages where status == at_warehouse and current_branch_id == <branch>`.

---

## Admin API (what the Next.js dashboard calls)

Base prefix: `/admin`. All return JSON.

### Dashboard reads
| Method | Path | Returns |
| --- | --- | --- |
| GET | `/admin/overview` | counts: branches, drivers (+available), vehicles, packages by status, at-warehouse |
| GET | `/admin/branches` | all branches |
| GET | `/admin/drivers` | all drivers, each with its joined `vehicle` |
| GET | `/admin/vehicles` | all vehicles |
| GET | `/admin/packages?status=&limit=` | packages, newest first, optional status filter |
| GET | `/admin/inventory` | live inventory grouped per warehouse |
| GET | `/admin/packages/{id}/timeline` | the package + its ordered `package_events` |

### Seed / write (needed before routing can run)
| Method | Path | Body |
| --- | --- | --- |
| POST | `/admin/branches` | `{ name, lat, lng }` |
| POST | `/admin/vehicles` | `{ vehicle_type, license_plate, is_available? }` |
| POST | `/admin/drivers` | `{ name, phone, current_lat?, current_lng?, vehicle_id?, is_available? }` |

`vehicle_type` ∈ `motorbike | tuktuk | van | container`.

### Auto routing
```
POST /admin/routing/auto
{
  "depot_branch_id": "<branch id>",   // optional, defaults to first branch
  "num_drivers": 3,                    // vehicles available
  "include_completed": false,          // true = route every delivery (demo)
  "time_limit_s": 30                   // OR-Tools budget
}
```
Response:
```jsonc
{
  "is_feasible": true,
  "depot_branch_id": "...",
  "num_drivers": 3,
  "total_time_min": 124.5,
  "unassigned_order_ids": [],
  "routes": [
    { "driver_index": 0, "total_time_min": 62.1,
      "stops": [ { "node": 0, "lat": 11.56, "lng": 104.92 }, ... ] }
  ],
  "assignments": [
    { "order_id": "1", "pickup_branch": "Central WH",
      "dropoff_branch": "South Branch", "needs_transfer": true }
  ]
}
```

**How auto-routing works** (`routes/admin.py` → `routing/` package):
1. `load_branches(db)` + `load_pending_deliveries(db)` — pull data from Mongo.
2. `assign_warehouses()` — nearest branch per pickup/dropoff via **OSRM** travel time.
   Cross-branch orders are flagged `needs_transfer` and get an extra warehouse-to-warehouse leg.
3. `build_distance_matrix()` — N×N OSRM travel-time matrix over all stops.
4. `solve_vrppd()` — **OR-Tools** pickup-and-delivery VRP, returns an ordered stop list per driver.

Steps 2–4 are blocking (network + CPU), so the endpoint runs them in a threadpool.
A `502` means **OSRM was unreachable**, not a logic bug — set `OSRM_BASE_URL` to a
reachable server (default is the public demo, ~100-location limit).

---

## Other routers (used by drivers/customers, not the admin)

- `POST /pickups` book → returns the auto-assigned driver (or scheduled/drop-at-branch).
- `POST /pickups/{id}/scan`, `POST /pickups/{id}/dropoff` — pickup lifecycle.
- `POST /warehouse/{id}/arrive`, `/depart`, `GET /warehouse/{branch}/inventory`.
- `GET /confirmation/confirm/{token}`, `/reject/{token}` — HTML pages from the email link.
- `POST /delivery/predict`-style + `/accept`, `/auto-mapping`; `POST /c2c/predict` + `/confirm`.
  Train/retrain/auto-mapping require the `X-API-Key` header.

---

## Running it

```bash
pip install -r requirements.txt
# .env needs at least: MONGODB_URI, API_KEY  (see .env for the rest)
uvicorn api:app --reload          # http://localhost:8000/docs for live Swagger UI
pytest -q                         # 34 tests, fully offline (mongomock_motor)
```

Key env vars: `MONGODB_URI` (Mongo Atlas / local), `API_KEY`, `OSRM_BASE_URL`,
`OSRM_BATCH_SIZE`, `SMTP_*`, `BASE_URL` (for confirmation email links).

Interactive API docs are always at **`/docs`** (Swagger) — the fastest way for the
Next.js dev to see every endpoint, its request body, and try it live.
