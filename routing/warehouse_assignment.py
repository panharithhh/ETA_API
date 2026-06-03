"""
Warehouse assignment: for each delivery, find the nearest branch (by OSRM
travel time) for its pickup point and its dropoff point.

If the two branches differ, the route needs an inter-warehouse transfer leg.
Those orders are flagged and their branch-to-branch arc is injected as an
extra PickupDeliveryPair in the VRP input.
"""

from dataclasses import dataclass

from routing.osrm_client import get_table


@dataclass
class Branch:
    id: int
    name: str
    lat: float
    lng: float


@dataclass
class DeliveryAssignment:
    order_id: int
    pickup_lat: float
    pickup_lng: float
    dropoff_lat: float
    dropoff_lng: float
    pickup_branch: Branch
    dropoff_branch: Branch
    needs_transfer: bool       # True when pickup and dropoff are under different branches


def assign_warehouses(
    deliveries: list[dict],
    branches: list[Branch],
) -> list[DeliveryAssignment]:
    """
    Assign each delivery to its nearest pickup branch and dropoff branch
    using real OSRM road-network travel times.

    deliveries : list of dicts with keys:
                   order_id, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng
    branches   : list of Branch objects loaded from routing_branches (or branches) table

    Returns a DeliveryAssignment per delivery.
    OSRM is called once in batch (all delivery points + all branch points together)
    to avoid N×M individual HTTP requests.
    """
    if not deliveries or not branches:
        return []

    branch_coords = [(b.lat, b.lng) for b in branches]

    # One OSRM call per delivery point (pickup + dropoff) to all branches
    # Batch: [delivery_pickup_0, delivery_pickup_1, ..., delivery_dropoff_0, ..., branch_0, ...]
    n_deliveries = len(deliveries)
    pickup_coords  = [(d["pickup_lat"],  d["pickup_lng"])  for d in deliveries]
    dropoff_coords = [(d["dropoff_lat"], d["dropoff_lng"]) for d in deliveries]

    all_locs = pickup_coords + dropoff_coords + branch_coords
    n_branch = len(branches)
    branch_dest_indices = list(range(2 * n_deliveries, 2 * n_deliveries + n_branch))

    # Source rows: each delivery's pickup and dropoff point
    source_indices = list(range(2 * n_deliveries))

    # Single OSRM table call: rows = delivery points, cols = all locations
    # We only care about the distances to branch columns
    durations = get_table(all_locs, sources=source_indices)
    # durations[i][j] = travel time from all_locs[i] to all_locs[j]

    assignments: list[DeliveryAssignment] = []
    for i, d in enumerate(deliveries):
        pickup_row  = durations[i]                  # row for pickup point i
        dropoff_row = durations[n_deliveries + i]   # row for dropoff point i

        # Nearest branch by travel time from pickup
        pickup_branch = min(
            branches,
            key=lambda b, row=pickup_row, bi=branch_dest_indices: row[bi[branches.index(b)]],
        )
        dropoff_branch = min(
            branches,
            key=lambda b, row=dropoff_row, bi=branch_dest_indices: row[bi[branches.index(b)]],
        )

        assignments.append(DeliveryAssignment(
            order_id=d["order_id"],
            pickup_lat=d["pickup_lat"],
            pickup_lng=d["pickup_lng"],
            dropoff_lat=d["dropoff_lat"],
            dropoff_lng=d["dropoff_lng"],
            pickup_branch=pickup_branch,
            dropoff_branch=dropoff_branch,
            needs_transfer=(pickup_branch.id != dropoff_branch.id),
        ))

    return assignments


def build_vrp_locations(
    assignments: list[DeliveryAssignment],
    branches: list[Branch],
    depot_branch_id: int,
) -> tuple[list[tuple[float, float]], list[dict]]:
    """
    Translate DeliveryAssignments into a flat locations list and VRP pair specs
    that vrp_solver.solve_vrppd() can consume.

    Returns:
      locations : flat list of (lat, lng); index 0 is the depot branch
      pairs     : list of dicts with order_id, pickup_node, delivery_node
                  (nodes are indices into locations)

    Inter-warehouse orders get an EXTRA pair for the branch-to-branch transfer
    leg, chained with their last-mile pair. Both pairs share the same order_id
    so results can be joined back.
    """
    branch_map = {b.id: b for b in branches}
    depot = branch_map[depot_branch_id]

    # Slot 0: depot
    locations: list[tuple[float, float]] = [(depot.lat, depot.lng)]
    branch_node: dict[int, int] = {depot_branch_id: 0}

    def _branch_node(b: Branch) -> int:
        if b.id not in branch_node:
            branch_node[b.id] = len(locations)
            locations.append((b.lat, b.lng))
        return branch_node[b.id]

    pairs: list[dict] = []

    for a in assignments:
        pb = _branch_node(a.pickup_branch)
        db = _branch_node(a.dropoff_branch)

        # Customer delivery point
        delivery_node = len(locations)
        locations.append((a.dropoff_lat, a.dropoff_lng))

        if a.needs_transfer:
            # Transfer leg: pickup_branch → dropoff_branch
            pairs.append({"order_id": a.order_id, "pickup_node": pb, "delivery_node": db})
            # Last-mile leg: dropoff_branch → customer
            pairs.append({"order_id": a.order_id, "pickup_node": db, "delivery_node": delivery_node})
        else:
            # Direct: pickup_branch → customer
            pairs.append({"order_id": a.order_id, "pickup_node": pb, "delivery_node": delivery_node})

    return locations, pairs
