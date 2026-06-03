import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from numpy import radians, sin, cos, sqrt, arcsin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "c2c_model.pkl")
X_PATH     = os.path.join(BASE_DIR, "models", "c2c_X.pkl")
Y_PATH     = os.path.join(BASE_DIR, "models", "c2c_y.pkl")

FEATURE_COLS = ["distance_km", "courier_rating", "hour_sin", "hour_cos", "day_sin", "day_cos"]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * arcsin(sqrt(a))


def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    df.replace("NaN", np.nan, inplace=True)

    df["time_taken"] = (
        df["Time_taken(min)"]
        .str.replace("(min)", "", regex=False)
        .str.strip()
        .astype(float)
    )
    df["distance_km"] = haversine(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    )
    df["courier_rating"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")

    df["pickup_dt"] = pd.to_datetime(
        df["Order_Date"] + " " + df["Time_Order_picked"],
        format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )
    hour           = df["pickup_dt"].dt.hour
    dow            = df["pickup_dt"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"]  = np.sin(2 * np.pi * dow / 7)
    df["day_cos"]  = np.cos(2 * np.pi * dow / 7)

    return df


def separate_outliers(df):
    dist_q1, dist_q3 = df["distance_km"].quantile([0.25, 0.75])
    dist_hi = min(dist_q3 + 1.5 * (dist_q3 - dist_q1), 30.0)

    t_q1, t_q3 = df["time_taken"].quantile([0.25, 0.75])
    t_iqr = t_q3 - t_q1
    t_lo, t_hi = t_q1 - 1.5 * t_iqr, t_q3 + 1.5 * t_iqr

    mask = (
        (df["distance_km"] > dist_hi) |
        (df["time_taken"]  < t_lo)    |
        (df["time_taken"]  > t_hi)
    )
    return df[~mask].copy(), df[mask].copy()


def train():
    global model, X, y

    df = load_and_clean()
    clean_df, _ = separate_outliers(df)

    needed = FEATURE_COLS + ["time_taken"]
    sub = clean_df[needed].dropna()
    X_all, y_all = sub[FEATURE_COLS], sub["time_taken"]

    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    
    
    for name, Xs, ys in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds = rf.predict(Xs)
        rmse  = np.sqrt(np.mean((preds - ys.values) ** 2))
        mae   = np.mean(np.abs(preds - ys.values))
        print(f"C2C {name} RMSE: {rmse:.2f} min   MAE: {mae:.2f} min")

    print(rmse/y_test.std())
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(X_train, X_PATH)
    joblib.dump(y_train, Y_PATH)

    model, X, y = rf, X_train, y_train
    return rf


def predict(
    pickup_lat: float,
    pickup_lon: float,
    delivery_lat: float,
    delivery_lon: float,
    courier_rating: float,
    at: datetime = None,
) -> float:
    if at is None:
        at = datetime.now()

    hour = at.hour
    dow  = at.weekday()
    row  = {
        "distance_km":   haversine(pickup_lat, pickup_lon, delivery_lat, delivery_lon),
        "courier_rating": courier_rating,
        "hour_sin":      np.sin(2 * np.pi * hour / 24),
        "hour_cos":      np.cos(2 * np.pi * hour / 24),
        "day_sin":       np.sin(2 * np.pi * dow / 7),
        "day_cos":       np.cos(2 * np.pi * dow / 7),
    }
    return float(model.predict(pd.DataFrame([row])[FEATURE_COLS])[0])


if not os.path.exists(MODEL_PATH):
    model = train()
else:
    model = joblib.load(MODEL_PATH)

X = joblib.load(X_PATH) if os.path.exists(X_PATH) else None
y = joblib.load(Y_PATH) if os.path.exists(Y_PATH) else None

if not os.path.exists(MODEL_PATH):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"C2C model not found at {MODEL_PATH} and training data not found at {DATA_PATH}. "
            "Run train() locally and commit models/c2c_model.pkl."
        )
    train()