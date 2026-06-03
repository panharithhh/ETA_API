# Warehouse / Mid-Mile Implementation

## What was added

Three new endpoints mounted at `/warehouse` that cover the gap between
`at_origin_branch` and final delivery.

---

## Endpoints

### POST `/warehouse/{package_id}/arrive`
Package reaches a warehouse.

**Body:** `{ "branch_id": int }`

- Validates current status is `at_origin_branch` or `in_transit_w2w`.
- Sets status → `at_warehouse`, updates `current_branch_id`.
- Inserts a `warehouse_inventory` row (`arrived_at` timestamp, `is_active=true`).
- If `branch_id` matches `destination_branch_id` **and** the package has a
  `customer_email`, calls `trigger_warehouse_arrival_email()` — sends the
  confirm/reject email to the customer.
- Email failure is caught and logged as a `PackageEvent`; it does **not**
  roll back the arrival.

---

### POST `/warehouse/{package_id}/depart`
Package leaves a warehouse for the next leg.

**Body:** `{ "to_branch_id": int | null }`

- Validates current status is `at_warehouse`.
- Marks the active `warehouse_inventory` row as shipped
  (`is_active=false`, `departed_at` timestamp).
- **`to_branch_id` omitted / null → final leg**
  - Status → `out_for_delivery`
- **`to_branch_id` provided → warehouse-to-warehouse leg**
  - Status → `in_transit_w2w`
  - Computes travel time: fetches road distance from OSRM, applies
    `minutes = (distance_km / 25) * 60`
  - Falls back to haversine if OSRM is unreachable
  - Estimated minutes returned in the response and recorded in `PackageEvent.notes`

---

### GET `/warehouse/{branch_id}/inventory`
Lists all packages currently sitting at a warehouse.

Returns only `is_active=true` inventory rows for that branch.
Each item includes `package_id`, `arrived_at`, current `status`,
`destination_branch_id`, and receiver coordinates.

---

## New database objects

| Object | Description |
|--------|-------------|
| `warehouse_inventory` table | Tracks which package is at which branch; has `arrived_at`, `departed_at`, `is_active` |
| `at_warehouse` status | Package is physically at a warehouse |
| `in_transit_w2w` status | Package is moving between two warehouses |
| `out_for_delivery` status | Package is on its way to the customer |

Migration file: `alembic/versions/003_warehouse_midmile.py`
Run with: `alembic upgrade head`

---

## Files changed / added

| File | Change |
|------|--------|
| `routes/warehouse.py` | New — all 3 endpoints |
| `models/first_mile.py` | Added `WarehouseInventory` model and 3 new `PackageStatus` values |
| `alembic/versions/003_warehouse_midmile.py` | New — DB migration |
| `api.py` | Imports and mounts `warehouse_router` |
| `services/warehouse_confirmation.py` | Existing hook called by `arrive` |
| `routing/osrm_client.py` | Existing client used for W2W travel time |

---

## Status flow (mid-mile portion)

```
at_origin_branch
      |
      v  POST /warehouse/{id}/arrive
  at_warehouse  ──────────────────────────────────> (email sent if destination branch)
      |
      |  POST /warehouse/{id}/depart
      |
      +-- to_branch_id provided --> in_transit_w2w --> (arrive at next warehouse)
      |
      +-- to_branch_id omitted  --> out_for_delivery
```

---

## Configuration

No new env vars required. Existing variables used:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL connection |
| `SMTP_*` | Email sending (already set) |
| `BASE_URL` | Builds confirm/reject links in emails |
| `OSRM_BASE_URL` | OSRM server (defaults to public demo) |
| `OSRM_TIMEOUT_S` | Request timeout in seconds (default 30) |
