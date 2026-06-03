"""
OSRM HTTP client.

Swap the server by setting OSRM_BASE_URL in .env:
  OSRM_BASE_URL=http://router.project-osrm.org   ← public demo (default)
  OSRM_BASE_URL=http://your-osrm-server:5000      ← self-hosted

Public demo limits: ~100 total locations per /table request.
Set OSRM_BATCH_SIZE=100 (default) or higher when self-hosted.
"""

import os
from typing import Optional

import requests

BASE_URL = os.getenv("OSRM_BASE_URL", "http://router.project-osrm.org")
_TIMEOUT = int(os.getenv("OSRM_TIMEOUT_S", "30"))


def _coords_str(locations: list[tuple[float, float]]) -> str:
    """OSRM expects longitude,latitude order."""
    return ";".join(f"{lng},{lat}" for lat, lng in locations)


def get_table(
    locations: list[tuple[float, float]],
    sources: Optional[list[int]] = None,
) -> list[list[float]]:
    """
    Call OSRM /table/v1/driving to get travel-time durations (seconds).

    locations : list of (lat, lng) tuples
    sources   : if given, only return rows for these indices (0-based).
                Destinations are always all locations.
                Returns a (len(sources) × len(locations)) matrix.
                If None, returns a full (N × N) matrix.

    Raises RuntimeError on OSRM error or non-200 HTTP response.
    """
    url = f"{BASE_URL}/table/v1/driving/{_coords_str(locations)}"
    params: dict = {"annotations": "duration"}
    if sources is not None:
        params["sources"] = ";".join(map(str, sources))

    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM /table error: {data.get('message', data.get('code'))}")

    return data["durations"]


def get_route(
    locations: list[tuple[float, float]],
) -> dict:
    """
    Call OSRM /route/v1/driving for a single ordered route.

    Returns the raw OSRM route dict (distance_m, duration_s, geometry, etc.).
    """
    url = f"{BASE_URL}/route/v1/driving/{_coords_str(locations)}"
    params = {"overview": "full", "steps": "false", "geometries": "geojson"}

    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM /route error: {data.get('message', data.get('code'))}")

    return data["routes"][0]
