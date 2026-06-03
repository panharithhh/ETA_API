import joblib
from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Security

from db import get_conn
from model import train_model as tmd, predict as predict_fn, haversine, MODEL_PATH
from retrain import retrain_delivery as retrain_delivery_fn
from schema import (
    AcceptDeliveryInput,
    AutoMapingInput,
    BatchPredictionInput,
    DeliveryItem,
    DriverInput,
    StopInput,
)
from api_auth import verify_api_key

router = APIRouter(prefix="/delivery", tags=["Delivery — ETA & Routing"])

model_1 = joblib.load(MODEL_PATH)
prediction_store: dict = {}  # order_id → record, held until driver confirms


def _run_predictions(data: BatchPredictionInput):
    trip_package_count = len(data.deliveries)
    stop_counts = {}
    for item in data.deliveries:
        key = (item.delivery_gps_lat, item.delivery_gps_lng)
        stop_counts[key] = stop_counts.get(key, 0) + 1
    trip_stop_count = len(stop_counts)

    now = datetime.now(ZoneInfo("Asia/Phnom_Penh"))
    first_accept = data.deliveries[0].accept_time or data.accept_time or now
    trip_start = first_accept if first_accept.tzinfo else first_accept.replace(tzinfo=now.tzinfo)
    last_eta = trip_start
    prev_lat = data.deliveries[0].accept_gps_lat
    prev_lng = data.deliveries[0].accept_gps_lng
    rows = []
    stop_idx = -1
    remaining_stops = trip_stop_count
    stop_progress = 0.0
    is_first_stop = 1

    for i, item in enumerate(data.deliveries):
        stop_package_count = stop_counts[(item.delivery_gps_lat, item.delivery_gps_lng)]
        item_time = last_eta

        same_stop = (
            i > 0
            and item.delivery_gps_lat == data.deliveries[i - 1].delivery_gps_lat
            and item.delivery_gps_lng == data.deliveries[i - 1].delivery_gps_lng
        )

        if same_stop:
            eta = last_eta
            segment_dist = 0.0
            segment_minutes = 0.0
        else:
            stop_idx += 1
            remaining_stops = trip_stop_count - stop_idx - 1
            stop_progress = stop_idx / max(trip_stop_count - 1, 1)
            is_first_stop = 1 if stop_idx == 0 else 0
            segment_dist = haversine(prev_lat, prev_lng, item.delivery_gps_lat, item.delivery_gps_lng)
            segment_minutes = predict_fn(
                segment_dist,
                trip_package_count,
                stop_package_count,
                item_time,
                stop_idx,
                remaining_stops,
                trip_stop_count,
                stop_progress,
                is_first_stop,
                courier_id=data.courier_id,
            )
            eta = last_eta + timedelta(minutes=segment_minutes)

        eta_early = eta - timedelta(minutes=15)
        eta_late = eta + timedelta(minutes=15)
        last_eta = eta
        prev_lat, prev_lng = item.delivery_gps_lat, item.delivery_gps_lng

        prediction_store[item.order_id] = {
            "order_id": item.order_id,
            "courier_id": data.courier_id,
            "accept_time": item_time.strftime("%Y-%m-%d %H:%M:%S"),
            "accept_gps_lng": item.accept_gps_lng,
            "accept_gps_lat": item.accept_gps_lat,
            "delivery_gps_lng": item.delivery_gps_lng,
            "delivery_gps_lat": item.delivery_gps_lat,
            "stop_package_count": stop_package_count,
            "stop_index": stop_idx,
            "remaining_stops": remaining_stops,
            "segment_distance": round(segment_dist, 4),
            "predicted_minutes": round(segment_minutes, 2),
            "eta": eta.isoformat(),
        }

        cumulative_minutes = (eta - trip_start).total_seconds() / 60
        rows.append({
            "order_id": item.order_id,
            "predicted_minutes": round(cumulative_minutes, 2),
            "eta_early": eta_early.isoformat(),
            "eta": eta.isoformat(),
            "eta_late": eta_late.isoformat(),
            "display": {
                "early": eta_early.strftime("%-I:%M %p"),
                "eta": eta.strftime("%-I:%M %p"),
                "late": eta_late.strftime("%-I:%M %p"),
            },
        })

    return trip_package_count, rows


@router.post("/train", summary="Retrain the delivery ETA model from scratch", dependencies=[Security(verify_api_key)])
def train_model():
    tmd()
    global model_1
    model_1 = joblib.load(MODEL_PATH)
    return {"status": "training_complete"}


@router.post("/accept", summary="Confirm a delivery — logs actual delivery time and triggers DB write")
def accept_delivery(data: AcceptDeliveryInput):
    record = prediction_store.pop(data.order_id, None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No pending prediction for order {data.order_id}")

    delivery_time = datetime.now(ZoneInfo("Asia/Phnom_Penh")).strftime("%Y-%m-%d %H:%M:%S")
    record["delivery_time"] = delivery_time

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO delivery_log
                    (order_id, courier_id, accept_time, accept_gps_lng, accept_gps_lat,
                     delivery_gps_lng, delivery_gps_lat, stop_package_count, predicted_minutes, eta, delivery_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    int(record["order_id"]), str(record["courier_id"]), record["accept_time"],
                    float(record["accept_gps_lng"]), float(record["accept_gps_lat"]),
                    float(record["delivery_gps_lng"]), float(record["delivery_gps_lat"]),
                    int(record["stop_package_count"]), float(record["predicted_minutes"]),
                    record["eta"], delivery_time,
                ),
            )
            cur.execute(
                "UPDATE predictions SET delivery_time = %s WHERE order_id = %s",
                (delivery_time, data.order_id),
            )
        conn.commit()

    return {"status": "recorded", "order_id": data.order_id}


@router.post("/retrain", summary="Incremental retrain using confirmed deliveries from DB", dependencies=[Security(verify_api_key)])
def retrain():
    ok = retrain_delivery_fn()
    if ok:
        return {"status": "successfully retrained"}
    return {"status": "retrain failed"}


@router.post("/auto-mapping", summary="Assign stops to drivers and generate optimized routes with ETAs", dependencies=[Security(verify_api_key)])
def auto_mapping(data: AutoMapingInput):
    if not data.stops:
        raise HTTPException(status_code=400, detail="No stops provided")
    if not data.drivers:
        raise HTTPException(status_code=400, detail="No drivers provided")

    if data.orders:
        data.stops = [
            StopInput(
                order_id=o.order_id,
                accept_time=o.accept_time,
                accept_gps_lat=o.pickup_lat,
                accept_gps_lng=o.pickup_lng,
                delivery_gps_lat=o.dropoff_lat,
                delivery_gps_lng=o.dropoff_lng,
            )
            for o in data.orders
        ]

    remaining = list(data.stops)
    assigned = {}
    cur_pos = {}

    for d in data.drivers:
        assigned[d.driver_id] = []
        cur_pos[d.driver_id] = (d.current_lat, d.current_lng)

    while remaining:
        for driver in data.drivers:
            if not remaining:
                break
            nearest = min(
                remaining,
                key=lambda s, d=driver: haversine(
                    cur_pos[d.driver_id][0], cur_pos[d.driver_id][1],
                    s.delivery_gps_lat, s.delivery_gps_lng,
                ),
            )
            assigned[driver.driver_id].append(nearest)
            cur_pos[driver.driver_id] = (nearest.delivery_gps_lat, nearest.delivery_gps_lng)
            remaining.remove(nearest)

    results = []
    for driver in data.drivers:
        stops = assigned[driver.driver_id]
        if not stops:
            continue

        unvisited = list(stops)
        ordered = []
        cur_lat, cur_lng = driver.current_lat, driver.current_lng
        while unvisited:
            nearest = min(unvisited, key=lambda s: haversine(cur_lat, cur_lng, s.delivery_gps_lat, s.delivery_gps_lng))
            ordered.append(nearest)
            cur_lat, cur_lng = nearest.delivery_gps_lat, nearest.delivery_gps_lng
            unvisited.remove(nearest)

        batch = BatchPredictionInput(
            courier_id=driver.driver_id,
            accept_time=data.accept_time,
            deliveries=[
                DeliveryItem(
                    order_id=s.order_id,
                    accept_time=s.accept_time,
                    accept_gps_lat=s.accept_gps_lat,
                    accept_gps_lng=s.accept_gps_lng,
                    delivery_gps_lat=s.delivery_gps_lat,
                    delivery_gps_lng=s.delivery_gps_lng,
                )
                for s in ordered
            ],
        )

        trip_package_count, deliveries = _run_predictions(batch)

        with get_conn() as conn:
            with conn.cursor() as cur:
                for d in deliveries:
                    rec = prediction_store[d["order_id"]]
                    cur.execute(
                        """
                        INSERT INTO predictions
                            (order_id, accept_time, accept_gps_lng, accept_gps_lat,
                             delivery_gps_lng, delivery_gps_lat, stop_package_count,
                             stop_index, remaining_stops, segment_distance, predicted_min, eta)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (order_id) DO NOTHING
                        """,
                        (
                            int(rec["order_id"]), rec["accept_time"], float(rec["accept_gps_lng"]),
                            float(rec["accept_gps_lat"]), float(rec["delivery_gps_lng"]),
                            float(rec["delivery_gps_lat"]), int(rec["stop_package_count"]),
                            int(rec["stop_index"]), int(rec["remaining_stops"]),
                            float(rec["segment_distance"]), float(rec["predicted_minutes"]),
                            d["display"]["eta"],
                        ),
                    )
            conn.commit()

        results.append({
            "driver_id": driver.driver_id,
            "total_packages": trip_package_count,
            "optimized_route": [
                {"stop": i + 1, "order_id": s.order_id, "lat": s.delivery_gps_lat, "lng": s.delivery_gps_lng}
                for i, s in enumerate(ordered)
            ],
            "deliveries": deliveries,
        })

    return {"drivers": results}
