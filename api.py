import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import joblib
from sklearn.ensemble import RandomForestRegressor
from model import train_model as tmd, predict as predict_fn, haversine, MODEL_PATH
from schema import BatchPredictionInput, PredictionInput, AcceptDeliveryInput, AutoMapingInput, DeliveryItem, DriverInput, StopInput, C2CPredictInput, C2CConfirmInput
from typing import List
from retrain import retrain_delivery as retrain_delivery_fn
from db import get_conn, init_db
import c2c as c2c_model

load_dotenv()

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

app = FastAPI()

model_1 = joblib.load(MODEL_PATH)

init_db()

prediction_store: dict = {}  # order_id -> record, held until driver confirms

@app.post("/train_model_delivery", dependencies=[Security(verify_api_key)])
def initial_training():
    tmd()
    global model_1
    model_1 = joblib.load(MODEL_PATH)
    return {"status": "training_complete"}

# @app.post("/train_model_pickup", dependencies=[Security(verify_api_key)])
# def train_pickup():
#     tmp()
#     return {"status": "training_complete"}

# @app.post("/retrain_delivery", dependencies=[Security(verify_api_key)])
# def retrain_delivery(records: List[RetrainItem]):
#     retrain_fn([r.model_dump() for r in records])
#     global model_1
#     model_1 = joblib.load(MODEL_PATH)
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
                courier_id=data.courier_id,
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

@app.post("/accept_delivery")
def accept_delivery(data: AcceptDeliveryInput): 
    record = prediction_store.pop(data.order_id, None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No pending prediction for order {data.order_id}")
    delivery_time = datetime.now(ZoneInfo("Asia/Phnom_Penh")).strftime("%Y-%m-%d %H:%M:%S")
    record["delivery_time"] = delivery_time

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO delivery_log
                    (order_id, courier_id, accept_time, accept_gps_lng, accept_gps_lat,
                     delivery_gps_lng, delivery_gps_lat, stop_package_count, predicted_minutes, eta, delivery_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (int(record["order_id"]), str(record["courier_id"]), record["accept_time"],
                  float(record["accept_gps_lng"]), float(record["accept_gps_lat"]),
                  float(record["delivery_gps_lng"]), float(record["delivery_gps_lat"]),
                  int(record["stop_package_count"]), float(record["predicted_minutes"]),
                  record["eta"], delivery_time))
            cur.execute(
                "UPDATE predictions SET delivery_time = %s WHERE order_id = %s",
                (delivery_time, data.order_id)
            )
        conn.commit()

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
                    cur.execute("""
                        INSERT INTO predictions
                            (order_id, accept_time, accept_gps_lng, accept_gps_lat,
                             delivery_gps_lng, delivery_gps_lat, stop_package_count,
                             stop_index, remaining_stops, segment_distance, predicted_min, eta)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (order_id) DO NOTHING
                    """, (int(rec["order_id"]), rec["accept_time"], float(rec["accept_gps_lng"]),
                          float(rec["accept_gps_lat"]), float(rec["delivery_gps_lng"]), float(rec["delivery_gps_lat"]),
                          int(rec["stop_package_count"]), int(rec["stop_index"]), int(rec["remaining_stops"]),
                          float(rec["segment_distance"]), float(rec["predicted_minutes"]), d["display"]["eta"]))
            conn.commit()

        results.append({
            "driver_id":       driver.driver_id,
            "total_packages":  trip_package_count,
            "optimized_route": [{"stop": i + 1, "order_id": s.order_id, "lat": s.delivery_gps_lat, "lng": s.delivery_gps_lng} for i, s in enumerate(ordered)],
            "deliveries":      deliveries,
        })


    return {"drivers": results}


c2c_store: dict = {}


@app.post("/c2c/predict")
def c2c_predict(data: C2CPredictInput):
    now = datetime.now(ZoneInfo("Asia/Phnom_Penh"))
    predicted_min = c2c_model.predict(
        data.pickup_lat, data.pickup_lon,
        data.delivery_lat, data.delivery_lon,
        data.courier_rating,
        at=now,
    )
    eta = now + timedelta(minutes=predicted_min)
    eta_early = eta - timedelta(minutes=15)
    eta_late  = eta + timedelta(minutes=15)

    c2c_store[data.order_id] = {
        "order_id":       data.order_id,
        "courier_rating": data.courier_rating,
        "pickup_lat":     data.pickup_lat,
        "pickup_lon":     data.pickup_lon,
        "delivery_lat":   data.delivery_lat,
        "delivery_lon":   data.delivery_lon,
        "distance_km":    round(c2c_model.haversine(data.pickup_lat, data.pickup_lon, data.delivery_lat, data.delivery_lon), 4),
        "accept_time":    now.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_min":  round(predicted_min, 2),
    }

    return {
        "order_id":        data.order_id,
        "predicted_min":   round(predicted_min, 2),
        "eta_early":       eta_early.isoformat(),
        "eta":             eta.isoformat(),
        "eta_late":        eta_late.isoformat(),
        "display": {
            "early": eta_early.strftime("%-I:%M %p"),
            "eta":   eta.strftime("%-I:%M %p"),
            "late":  eta_late.strftime("%-I:%M %p"),
        },
    }


@app.post("/c2c/confirm")
def c2c_confirm(data: C2CConfirmInput):
    record = c2c_store.pop(data.order_id, None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No pending prediction for order {data.order_id}")

    delivery_time = datetime.now(ZoneInfo("Asia/Phnom_Penh")).strftime("%Y-%m-%d %H:%M:%S")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO c2c_log
                    (order_id, courier_rating, pickup_lat, pickup_lon,
                     delivery_lat, delivery_lon, distance_km, accept_time, predicted_min, delivery_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (order_id) DO NOTHING
            """, (
                record["order_id"], record["courier_rating"],
                record["pickup_lat"], record["pickup_lon"],
                record["delivery_lat"], record["delivery_lon"],
                record["distance_km"], record["accept_time"],
                record["predicted_min"], delivery_time,
            ))
        conn.commit()

    return {"status": "recorded", "order_id": data.order_id}


@app.post("/c2c/retrain", dependencies=[Security(verify_api_key)])
def c2c_retrain():
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT courier_rating, pickup_lat, pickup_lon,
                   delivery_lat, delivery_lon, distance_km,
                   accept_time, delivery_time
            FROM c2c_log
            WHERE delivery_time IS NOT NULL
        """, conn)

    if df.empty:
        return {"status": "no data to retrain on"}

    df["accept_time"]   = pd.to_datetime(df["accept_time"]).dt.tz_localize(None)
    df["delivery_time"] = pd.to_datetime(df["delivery_time"]).dt.tz_localize(None)
    df["time_taken"]    = (df["delivery_time"] - df["accept_time"]).dt.total_seconds() / 60
    df = df[(df["time_taken"] > 0) & (df["time_taken"] <= 180)]

    if df.empty:
        return {"status": "no valid rows after filtering"}

    hour           = df["accept_time"].dt.hour
    dow            = df["accept_time"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"]  = np.sin(2 * np.pi * dow / 7)
    df["day_cos"]  = np.cos(2 * np.pi * dow / 7)

    new_X = df[c2c_model.FEATURE_COLS].dropna()
    new_y = df.loc[new_X.index, "time_taken"]

    if c2c_model.X is not None and c2c_model.y is not None:
        X_retrain = pd.concat([c2c_model.X, new_X], ignore_index=True)
        y_retrain = pd.concat([c2c_model.y, new_y], ignore_index=True)
        weights   = np.concatenate([np.ones(len(c2c_model.X)), np.full(len(new_X), 10000.0)])
    else:
        X_retrain, y_retrain, weights = new_X, new_y, None

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_retrain, y_retrain, sample_weight=weights)

    joblib.dump(rf, c2c_model.MODEL_PATH)
    joblib.dump(X_retrain, c2c_model.X_PATH)
    joblib.dump(y_retrain, c2c_model.Y_PATH)

    c2c_model.model = rf
    c2c_model.X     = X_retrain
    c2c_model.y     = y_retrain

    return {"status": "retrain complete", "new_samples": len(new_X)}
