import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import model as m

PREDICTIONS_PATH = os.path.join(os.path.dirname(__file__), "predictions.csv")


def retrain_delivery() -> bool:
    if m.X is None or m.y is None:#train the model
        m.train_model()

    if m.X is None or m.y is None:#either one is empty
        return False

    if not os.path.exists(PREDICTIONS_PATH): #wrong path
        return False

    df = pd.read_csv(PREDICTIONS_PATH)
    df = df[df["Delivery Time"].notna() & (df["Delivery Time"] != "")]

    if df.empty:
        return False

    df["Accept Time"] = pd.to_datetime(df["Accept Time"]).dt.tz_localize(None)

    def parse_dt(s):
        dt = pd.to_datetime(s)
        if dt.tzinfo is not None:
            return dt.tz_convert("Asia/Phnom_Penh").replace(tzinfo=None)
        return dt

    df["Delivery Time"] = df["Delivery Time"].apply(parse_dt)

    df["time_taken"] = (df["Delivery Time"] - df["Accept Time"]).dt.total_seconds() / 60
    df = df[(df["time_taken"] > 0) & (df["time_taken"] <= 600)]

    if df.empty:
        return False

    df["distance"] = df.apply(lambda r: m.haversine(
        r["Accept GPS Lat"], r["Accept GPS Lng"],
        r["Delivery GPS Lat"], r["Delivery GPS Lng"],
    ), axis=1)
    df = df[df["distance"] <= 50]

    if df.empty:
        return False

    df["trip_package_count"] = df.groupby("Accept Time")["Order ID"].transform("count")
    df["hour"]        = df["Accept Time"].dt.hour
    df["day_of_week"] = df["Accept Time"].dt.dayofweek

    new_X = df[["distance", "hour", "day_of_week", "trip_package_count", "Stop Package Count", "Vehicle Type"]].copy()
    new_X.columns = ["distance", "hour", "day_of_week", "trip_package_count", "stop_package_count", "vehicle_type"]
    new_y = df["time_taken"].reset_index(drop=True)

    X_retrain = pd.concat([m.X, new_X], ignore_index=True)
    y_retrain = pd.concat([m.y, new_y], ignore_index=True)
    weights   = np.concatenate([np.ones(len(m.X)), np.full(len(new_X), 10000.0)])

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.5,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_retrain, y_retrain, sample_weight=weights)

    joblib.dump(rf, m.MODEL_PATH)
    joblib.dump(X_retrain, m.X_PATH)
    joblib.dump(y_retrain, m.Y_PATH)

    m.model = rf
    m.X     = X_retrain
    m.y     = y_retrain

    return True
