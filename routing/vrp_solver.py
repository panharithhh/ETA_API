"""
OR-Tools VRPPD (Vehicle Routing Problem with Pickup and Delivery) solver.

Uses:
  First solution  : PARALLEL_CHEAPEST_INSERTION
  Metaheuristic   : GUIDED_LOCAL_SEARCH
  Constraint type : AddPickupAndDelivery with same-vehicle + precedence constraints

This module is pure logic — it takes a pre-built integer distance matrix
and returns ordered stop indices. It has no DB or OSRM calls.
"""

from dataclasses import dataclass, field

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# OR-Tools requires integer costs; multiply float seconds by this scale factor
_SCALE = 100  # 0.01-second resolution


@dataclass
class PickupDeliveryPair:
    """One order's pickup and delivery, referenced by index into the locations list."""
    order_id: int
    pickup_node: int    # index in locations list
    delivery_node: int  # index in locations list
    demand: int = 1     # package count, used if you add capacity constraints later


@dataclass
class DriverRoute:
    driver_index: int
    stop_nodes: list[int]           # ordered location indices (depot excluded from middle)
    total_time_s: float


@dataclass
class RoutingSolution:
    routes: list[DriverRoute] = field(default_factory=list)
    unassigned_order_ids: list[int] = field(default_factory=list)
    is_feasible: bool = False
    total_time_s: float = 0.0


def solve_vrppd(
    distance_matrix: list[list[float]],
    num_drivers: int,
    depot_node: int,
    pairs: list[PickupDeliveryPair],
    time_limit_s: int = 30,
    max_route_time_s: int = 86_400,   # 24 h per vehicle
) -> RoutingSolution:
    """
    Solve VRPPD and return an ordered route per driver.

    distance_matrix : N × N float matrix (seconds), built by distance_matrix.py
    num_drivers     : number of available vehicles
    depot_node      : index in distance_matrix that is the start/end depot
    pairs           : pickup-delivery pairs (indices into distance_matrix)
    time_limit_s    : solver wall-clock budget
    max_route_time_s: hard ceiling on any single driver's total route time

    Returns RoutingSolution.  If OR-Tools finds no feasible solution,
    is_feasible=False and all order_ids are listed in unassigned_order_ids.
    """
    n = len(distance_matrix)
    if not pairs:
        return RoutingSolution(is_feasible=True)

    manager = pywrapcp.RoutingIndexManager(n, num_drivers, depot_node)
    routing = pywrapcp.RoutingModel(manager)

    # ── Transit cost (integer-scaled seconds) ────────────────────────────────
    int_matrix = [
        [int(v * _SCALE) for v in row]
        for row in distance_matrix
    ]

    def _transit(from_idx: int, to_idx: int) -> int:
        return int_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_id = routing.RegisterTransitCallback(_transit)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_id)

    # ── Time dimension (needed for pickup-before-delivery precedence) ─────────
    routing.AddDimension(
        transit_id,
        0,                                  # no waiting slack
        int(max_route_time_s * _SCALE),     # max cumulative time per vehicle
        True,                               # fix start cumul to zero
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # ── Pickup & delivery pairs ───────────────────────────────────────────────
    solver = routing.solver()
    # High penalty so dropping an order is a last resort (e.g. if unreachable)
    penalty = int(100_000 * _SCALE)
    
    for pd in pairs:
        pickup_idx = manager.NodeToIndex(pd.pickup_node)
        delivery_idx = manager.NodeToIndex(pd.delivery_node)

        routing.AddPickupAndDelivery(pickup_idx, delivery_idx)

        # Allow dropping the order instead of failing the entire route
        routing.AddDisjunction([pickup_idx], penalty)
        routing.AddDisjunction([delivery_idx], penalty)

        # Same vehicle must handle both ends
        solver.Add(routing.VehicleVar(pickup_idx) == routing.VehicleVar(delivery_idx))

        # Pickup must be visited before delivery
        solver.Add(
            time_dim.CumulVar(pickup_idx) <= time_dim.CumulVar(delivery_idx)
        )

    # ── Search parameters ─────────────────────────────────────────────────────
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = time_limit_s

    solution = routing.SolveWithParameters(params)

    if not solution:
        return RoutingSolution(
            is_feasible=False,
            unassigned_order_ids=[p.order_id for p in pairs],
        )

    # ── Extract routes ────────────────────────────────────────────────────────
    result = RoutingSolution(is_feasible=True)
    for vehicle in range(num_drivers):
        index = routing.Start(vehicle)
        stops: list[int] = []
        route_time = 0.0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            stops.append(node)
            next_index = solution.Value(routing.NextVar(index))
            route_time += _transit(index, next_index) / _SCALE
            index = next_index

        stops.append(manager.IndexToNode(index))  # end depot

        if len(stops) > 2:  # more than just depot→depot
            result.routes.append(DriverRoute(
                driver_index=vehicle,
                stop_nodes=stops,
                total_time_s=route_time,
            ))
            result.total_time_s += route_time

    # Collect truly unassigned orders (both nodes still at depot)
    for pd in pairs:
        p_idx = manager.NodeToIndex(pd.pickup_node)
        if solution.Value(routing.NextVar(p_idx)) == routing.NextVar(p_idx).Max():
            result.unassigned_order_ids.append(pd.order_id)

    return result
