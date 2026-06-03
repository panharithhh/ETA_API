"""
Distance matrix builder using OSRM road-network travel times.

Handles batching for when N exceeds the OSRM server's per-request limit.
Set OSRM_BATCH_SIZE in .env (default 100 for public demo, raise when self-hosted).
"""

import os

from routing.osrm_client import get_table

_BATCH = int(os.getenv("OSRM_BATCH_SIZE", "100"))


def build_distance_matrix(
    locations: list[tuple[float, float]],
) -> list[list[float]]:
    """
    Build a full N × N travel-time matrix (seconds) using OSRM road distances.

    locations : list of (lat, lng) tuples, in the order you want indices.
                Index 0 is typically the depot/warehouse.

    For N ≤ OSRM_BATCH_SIZE  → single OSRM request.
    For N > OSRM_BATCH_SIZE  → row-batched requests using OSRM's `sources` param.
    Each batch sends all N coordinates but limits which source rows are returned,
    keeping the URL size O(N) rather than O(N²).

    Returns a list[list[float]] where matrix[i][j] is travel time in seconds
    from locations[i] to locations[j]. Diagonal is 0.
    """
    n = len(locations)
    if n == 0:
        return []
    if n == 1:
        return [[0.0]]

    if n <= _BATCH:
        return _get_full(locations)

    # Build empty N × N matrix then fill row-by-row in batches
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for start in range(0, n, _BATCH):
        end = min(start + _BATCH, n)
        sources = list(range(start, end))
        partial = get_table(locations, sources=sources)
        for offset, row in enumerate(partial):
            matrix[start + offset] = [float(v) if v is not None else 1e9 for v in row]

    return matrix


def _get_full(locations: list[tuple[float, float]]) -> list[list[float]]:
    """Single-request full matrix for N ≤ batch size."""
    raw = get_table(locations)
    # Replace OSRM nulls (unreachable) with a large penalty
    return [
        [float(v) if v is not None else 1e9 for v in row]
        for row in raw
    ]
