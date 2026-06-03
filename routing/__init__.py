"""
Warehouse routing module.

Typical call sequence:

    from db import get_conn
    from routing import (
        setup_routing_tables,
        load_branches,
        load_pending_deliveries,
        build_distance_matrix,
        assign_warehouses,
        build_vrp_locations,
        solve_vrppd,
        PickupDeliveryPair,
    )

    conn = get_conn()
    setup_routing_tables(conn)          # idempotent, creates routing_branches if absent

    branches   = load_branches(conn)
    deliveries = load_pending_deliveries(conn)

    assignments = assign_warehouses(deliveries, branches)
    locations, pair_specs = build_vrp_locations(assignments, branches, depot_branch_id=1)

    matrix = build_distance_matrix(locations)

    pairs = [PickupDeliveryPair(**p) for p in pair_specs]
    solution = solve_vrppd(matrix, num_drivers=3, depot_node=0, pairs=pairs)

    for route in solution.routes:
        named_stops = [locations[n] for n in route.stop_nodes]
        print(f"Driver {route.driver_index}: {named_stops}  ({route.total_time_s/60:.1f} min)")
"""

from routing.db_loader import (
    load_branches,
    load_pending_deliveries,
    setup_routing_tables,
)
from routing.distance_matrix import build_distance_matrix
from routing.vrp_solver import PickupDeliveryPair, RoutingSolution, solve_vrppd
from routing.warehouse_assignment import (
    assign_warehouses,
    build_vrp_locations,
    DeliveryAssignment,
)

__all__ = [
    "setup_routing_tables",
    "load_branches",
    "load_pending_deliveries",
    "build_distance_matrix",
    "assign_warehouses",
    "build_vrp_locations",
    "solve_vrppd",
    "PickupDeliveryPair",
    "RoutingSolution",
    "DeliveryAssignment",
]
