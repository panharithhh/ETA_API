import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import joblib
from fastapi import APIRouter, Depends, HTTPException, Security
from motor.motor_asyncio import AsyncIOMotorDatabase
from sklearn.ensemble import RandomForestRegressor

import c2c as c2c_model
from db import get_db
from schema import C2CConfirmInput, C2CPredictInput
from api_auth import verify_api_key

router = APIRouter(prefix="/c2c", tags=["C2C — Customer to Customer"])

c2c_store: dict = {}  # order_id → record, held until confirmed


@router.post("/predict")
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
    eta_late = eta + timedelta(minutes=15)

    c2c_store[data.order_id] = {
        "order_id": data.order_id,
        "courier_rating": data.courier_rating,
        "pickup_lat": data.pickup_lat,
        "pickup_lon": data.pickup_lon,
        "delivery_lat": data.delivery_lat,
        "delivery_lon": data.delivery_lon,
        "distance_km": round(c2c_model.haversine(data.pickup_lat, data.pickup_lon, data.delivery_lat, data.delivery_lon), 4),
        "accept_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_min": round(predicted_min, 2),
        "eta_early": eta_early.isoformat(),
        "eta": eta.isoformat(),
        "eta_late": eta_late.isoformat(),
        "display": {
            "early": eta_early.strftime("%-I:%M %p"),
            "eta": eta.strftime("%-I:%M %p"),
            "late": eta_late.strftime("%-I:%M %p"),
        },
    }

    return {
        "order_id": data.order_id,
        "predicted_min": round(predicted_min, 2),
        "pickup_lat": data.pickup_lat,
        "pickup_lon": data.pickup_lon,
        "delivery_lat": data.delivery_lat,
        "delivery_lon": data.delivery_lon,
        "eta_early": eta_early.isoformat(),
        "eta": eta.isoformat(),
        "eta_late": eta_late.isoformat(),
        "display": {
            "early": eta_early.strftime("%-I:%M %p"),
            "eta": eta.strftime("%-I:%M %p"),
            "late": eta_late.strftime("%-I:%M %p"),
        },
    }


@router.get("/active")
def c2c_active():
    return {"orders": list(c2c_store.values())}


@router.post("/confirm")
async def c2c_confirm(data: C2CConfirmInput, db: AsyncIOMotorDatabase = Depends(get_db)):
    record = c2c_store.pop(data.order_id, None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No pending prediction for order {data.order_id}")

    delivery_time = datetime.now(ZoneInfo("Asia/Phnom_Penh")).strftime("%Y-%m-%d %H:%M:%S")

    # $setOnInsert + upsert == INSERT … ON CONFLICT (order_id) DO NOTHING
    await db.c2c_log.update_one(
        {"order_id": record["order_id"]},
        {"$setOnInsert": {
            "order_id": record["order_id"],
            "courier_rating": record["courier_rating"],
            "pickup_lat": record["pickup_lat"],
            "pickup_lon": record["pickup_lon"],
            "delivery_lat": record["delivery_lat"],
            "delivery_lon": record["delivery_lon"],
            "distance_km": record["distance_km"],
            "accept_time": record["accept_time"],
            "predicted_min": record["predicted_min"],
            "delivery_time": delivery_time,
        }},
        upsert=True,
    )

    return {"status": "recorded", "order_id": data.order_id}


@router.post("/retrain", dependencies=[Security(verify_api_key)])
async def c2c_retrain(db: AsyncIOMotorDatabase = Depends(get_db)):
    rows = await db.c2c_log.find({"delivery_time": {"$ne": None}}).to_list(length=None)
    if not rows:
        return {"status": "no data to retrain on"}

    df = pd.DataFrame([
        {
            "courier_rating": r.get("courier_rating"),
            "pickup_lat": r.get("pickup_lat"),
            "pickup_lon": r.get("pickup_lon"),
            "delivery_lat": r.get("delivery_lat"),
            "delivery_lon": r.get("delivery_lon"),
            "distance_km": r.get("distance_km"),
            "accept_time": r.get("accept_time"),
            "delivery_time": r.get("delivery_time"),
        }
        for r in rows
    ])

    if df.empty:
        return {"status": "no data to retrain on"}

    df["accept_time"] = pd.to_datetime(df["accept_time"]).dt.tz_localize(None)
    df["delivery_time"] = pd.to_datetime(df["delivery_time"]).dt.tz_localize(None)
    df["time_taken"] = (df["delivery_time"] - df["accept_time"]).dt.total_seconds() / 60
    df = df[(df["time_taken"] > 0) & (df["time_taken"] <= 180)]

    if df.empty:
        return {"status": "no valid rows after filtering"}

    hour = df["accept_time"].dt.hour
    dow = df["accept_time"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * dow / 7)
    df["day_cos"] = np.cos(2 * np.pi * dow / 7)

    new_X = df[c2c_model.FEATURE_COLS].dropna()
    new_y = df.loc[new_X.index, "time_taken"]

    if c2c_model.X is not None and c2c_model.y is not None:
        X_retrain = pd.concat([c2c_model.X, new_X], ignore_index=True)
        y_retrain = pd.concat([c2c_model.y, new_y], ignore_index=True)
        weights = np.concatenate([np.ones(len(c2c_model.X)), np.full(len(new_X), 10000.0)])
    else:
        X_retrain, y_retrain, weights = new_X, new_y, None

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_split=5,
        min_samples_leaf=4, max_features=0.7, random_state=42, n_jobs=-1,
    )
    rf.fit(X_retrain, y_retrain, sample_weight=weights)

    joblib.dump(rf, c2c_model.MODEL_PATH)
    joblib.dump(X_retrain, c2c_model.X_PATH)
    joblib.dump(y_retrain, c2c_model.Y_PATH)

    c2c_model.model = rf
    c2c_model.X = X_retrain
    c2c_model.y = y_retrain

    return {"status": "retrain complete", "new_samples": len(new_X)}
