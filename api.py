import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import joblib
from model import train_model as tmd, predict as predict_fn, haversine
from schema import BatchPredictionInput, PredictionInput, AcceptDeliveryInput, AutoMapingInput, DeliveryItem, DriverInput
from typing import List
from retrain import retrain_delivery as retrain_delivery_fn

load_dotenv()

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

app = FastAPI()

model_1 = joblib.load("model_1.pkl")

LOG_PATH = os.path.join(os.path.dirname(__file__), "delivery_log.csv")
LOG_COLS  = ["order_id", "courier_id", "accept_time", "accept_gps_lng", "accept_gps_lat",
             "delivery_gps_lng", "delivery_gps_lat", "stop_package_count",
             "predicted_minutes", "eta", "delivery_time"]

prediction_store: dict = {}  # order_id -> record, held until driver confirms

@app.post("/train_model_delivery", dependencies=[Security(verify_api_key)])
def initial_training():
    tmd()
    global model_1
    model_1 = joblib.load("model_1.pkl")
    return {"status": "training_complete"}

# @app.post("/train_model_pickup", dependencies=[Security(verify_api_key)])
# def train_pickup():
#     tmp()
#     return {"status": "training_complete"}

# @app.post("/retrain_delivery", dependencies=[Security(verify_api_key)])
# def retrain_delivery(records: List[RetrainItem]):
#     retrain_fn([r.model_dump() for r in records])
#     global model_1
#     model_1 = joblib.load("model_1.pkl")
#     return {"status": "retrain_complete", "new_samples": len(records)}

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
            i > 0 and
            item.delivery_gps_lat == data.deliveries[i - 1].delivery_gps_lat and
            item.delivery_gps_lng == data.deliveries[i - 1].delivery_gps_lng
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
                segment_dist, trip_package_count, stop_package_count,
                item_time, stop_idx, remaining_stops,
                trip_stop_count, stop_progress, is_first_stop,
            )
            eta = last_eta + timedelta(minutes=segment_minutes)

        eta_early = eta - timedelta(minutes=15)
        eta_late  = eta + timedelta(minutes=15)
        last_eta  = eta
        prev_lat, prev_lng = item.delivery_gps_lat, item.delivery_gps_lng

        prediction_store[item.order_id] = {
            "order_id":           item.order_id,
            "courier_id":         data.courier_id,
            "accept_time":        item_time.strftime("%Y-%m-%d %H:%M:%S"),
            "accept_gps_lng":     item.accept_gps_lng,
            "accept_gps_lat":     item.accept_gps_lat,
            "delivery_gps_lng":   item.delivery_gps_lng,
            "delivery_gps_lat":   item.delivery_gps_lat,
            "stop_package_count": stop_package_count,
            "stop_index":         stop_idx,
            "remaining_stops":    remaining_stops,
            "segment_distance":   round(segment_dist, 4),
            "predicted_minutes":  round(segment_minutes, 2),
            "eta":                eta.isoformat(),
        }

        cumulative_minutes = (eta - trip_start).total_seconds() / 60
        rows.append({
            "order_id":          item.order_id,
            "predicted_minutes": round(cumulative_minutes, 2),
            "eta_early":         eta_early.isoformat(),
            "eta":               eta.isoformat(),
            "eta_late":          eta_late.isoformat(),
            "display": {
                "early": eta_early.strftime("%-I:%M %p"),
                "eta":   eta.strftime("%-I:%M %p"),
                "late":  eta_late.strftime("%-I:%M %p"),
            },
        })

    return trip_package_count, rows

PREDICTIONS_PATH = os.path.join(os.path.dirname(__file__), "predictions.csv")
PREDICTIONS_COLS = ["Order ID", "Accept Time", "Accept GPS Lng", "Accept GPS Lat",
                    "Delivery GPS Lng", "Delivery GPS Lat",
                    "Stop Package Count", "Stop Index", "Remaining Stops",
                    "Segment Distance", "Predicted (min)", "ETA", "Delivery Time"]

if os.path.exists(PREDICTIONS_PATH):
    _existing = pd.read_csv(PREDICTIONS_PATH, nrows=0)
    if list(_existing.columns) != PREDICTIONS_COLS:
        os.remove(PREDICTIONS_PATH)


@app.post("/accept_delivery")
def accept_delivery(data: AcceptDeliveryInput): 
    record = prediction_store.pop(data.order_id, None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No pending prediction for order {data.order_id}")
    delivery_time = datetime.now(ZoneInfo("Asia/Phnom_Penh")).strftime("%Y-%m-%d %H:%M:%S")
    record["delivery_time"] = delivery_time

    file_exists = os.path.exists(LOG_PATH)
    pd.DataFrame([record], columns=LOG_COLS).to_csv(
        LOG_PATH, mode="a", header=not file_exists, index=False
    )

    if os.path.exists(PREDICTIONS_PATH):
        df = pd.read_csv(PREDICTIONS_PATH)
        df["Order ID"] = df["Order ID"].astype(int)
        df.loc[df["Order ID"] == data.order_id, "Delivery Time"] = delivery_time
        df.to_csv(PREDICTIONS_PATH, index=False)

    return {"status": "recorded", "order_id": data.order_id}

@app.post("/retrain", dependencies=[Security(verify_api_key)])
def retrain():
    ok = retrain_delivery_fn()
    if ok:
        return {"status": "successfully retrained"}
    return {"status": "retrain failed"}
    
 

@app.post("/autoMaping", dependencies=[Security(verify_api_key)])
def autoMaping(data: AutoMapingInput):
    if not data.stops:
        raise HTTPException(status_code=400, detail="No stops provided")
    if not data.drivers:
        raise HTTPException(status_code=400, detail="No drivers provided")

    remaining = list(data.stops)
    assigned = {}
    cur_pos = {}
    
    for d in data.drivers:
        assigned[d.driver_id] = [] #initialize total driver
        cur_pos[d.driver_id] = (d.current_lat, d.current_lng) #track intial 

    while remaining:
        for driver in data.drivers:
            if not remaining:
                break
            nearest = min(remaining, key=lambda s, d=driver: haversine(
                cur_pos[d.driver_id][0],
                cur_pos[d.driver_id][1],
                s.delivery_gps_lat, 
                s.delivery_gps_lng,
            ))
            assigned[driver.driver_id].append(nearest)
            cur_pos[driver.driver_id] = (nearest.delivery_gps_lat, nearest.delivery_gps_lng)
            remaining.remove(nearest)

    file_exists = os.path.exists(PREDICTIONS_PATH)
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

        new_rows = pd.DataFrame([{
            "Order ID":           d["order_id"],
            "Accept Time":        prediction_store[d["order_id"]]["accept_time"],
            "Accept GPS Lng":     prediction_store[d["order_id"]]["accept_gps_lng"],
            "Accept GPS Lat":     prediction_store[d["order_id"]]["accept_gps_lat"],
            "Delivery GPS Lng":   prediction_store[d["order_id"]]["delivery_gps_lng"],
            "Delivery GPS Lat":   prediction_store[d["order_id"]]["delivery_gps_lat"],
            "Stop Package Count": prediction_store[d["order_id"]]["stop_package_count"],
            "Stop Index":         prediction_store[d["order_id"]]["stop_index"],
            "Remaining Stops":    prediction_store[d["order_id"]]["remaining_stops"],
            "Segment Distance":   prediction_store[d["order_id"]]["segment_distance"],
            "Predicted (min)":    prediction_store[d["order_id"]]["predicted_minutes"],
            "ETA":                d["display"]["eta"],
            "Delivery Time":      "",
        } for d in deliveries])

        new_rows.to_csv(PREDICTIONS_PATH, mode="a", header=not file_exists, index=False)
        file_exists = True

        results.append({
            "driver_id":       driver.driver_id,
            "total_packages":  trip_package_count,
            "optimized_route": [{"stop": i + 1, "order_id": s.order_id, "lat": s.delivery_gps_lat, "lng": s.delivery_gps_lng} for i, s in enumerate(ordered)],
            "deliveries":      deliveries,
        })


    return {"drivers": results}
    