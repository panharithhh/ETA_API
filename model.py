import pandas as pd
import numpy as np
from numpy import radians, sin, cos, sqrt, arcsin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.compose import TransformedTargetRegressor

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH         = os.path.join(MODELS_DIR, "model_1.pkl")
X_PATH             = os.path.join(MODELS_DIR, "X_train.pkl")
Y_PATH             = os.path.join(MODELS_DIR, "y_train.pkl")
COURIER_SPEED_PATH = os.path.join(MODELS_DIR, "courier_speed.pkl")

# BUG 7 FIX: was 300s — collapsed genuine separate stops into one
STOP_GROUPING_SECONDS = 120

FEATURE_COLS = [
    "segment_distance",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "trip_package_count", "stop_package_count", "trip_stop_count",
    "stop_index", "remaining_stops", "stop_progress",
    "is_first_stop", "is_last_stop",
    "packages_per_stop", "wait_time",
    "courier_avg_speed",
]


def assign_trips(df):
    df = df.sort_values(["courier_id", "accept_time"]).copy()
    ids = []
    for courier_id, group in df.groupby("courier_id", sort=False):
        trip_id = 0
        trip_start = None
        for t in group["accept_time"]:
            if (trip_start is None or
                    t.date() != trip_start.date() or
                    (t - trip_start).total_seconds() > 3600):
                trip_id += 1
                trip_start = t
            ids.append(trip_id)
    df["trip_id"] = ids
    return df


def assign_stops(df):
    df = df.sort_values(["courier_id", "delivery_time"]).copy()
    ids = []
    for courier_id, group in df.groupby("courier_id", sort=False):
        stop_id = 0
        stop_start = None
        for t in group["delivery_time"]:
            if (stop_start is None or
                    t.date() != stop_start.date() or
                    (t - stop_start).total_seconds() > STOP_GROUPING_SECONDS):
                stop_id += 1
                stop_start = t
            ids.append(stop_id)
    df["stop_id"] = ids
    return df


def train_model():
    global model, X, y, courier_speed
    df = load_data()
    clean_df = clean_data(df)

    clean_df["accept_time"]       = pd.to_datetime(clean_df["accept_time"],       format="%m-%d %H:%M:%S")
    clean_df["delivery_time"]     = pd.to_datetime(clean_df["delivery_time"],     format="%m-%d %H:%M:%S")
    clean_df["accept_gps_time"]   = pd.to_datetime(clean_df["accept_gps_time"],   format="%m-%d %H:%M:%S")
    clean_df["delivery_gps_time"] = pd.to_datetime(clean_df["delivery_gps_time"], format="%m-%d %H:%M:%S")

    clean_df["time_taken"] = (
        clean_df["delivery_gps_time"] - clean_df["accept_gps_time"]
    ).dt.total_seconds() / 60
    clean_df = clean_df[clean_df["time_taken"] > 0]

    clean_df["distance"] = haversine(
        clean_df["accept_gps_lat"], clean_df["accept_gps_lng"],
        clean_df["delivery_gps_lat"], clean_df["delivery_gps_lng"],
    )
    # BUG 6 FIX: was 50 km — too loose for urban last-mile
    clean_df = clean_df[clean_df["distance"] <= 15]
    print("distance.describe():\n", clean_df["distance"].describe().to_string(), "\n")

    clean_df["hour"]        = clean_df["accept_gps_time"].dt.hour
    clean_df["day_of_week"] = clean_df["accept_gps_time"].dt.day_of_week

    clean_df = assign_trips(clean_df)
    clean_df = assign_stops(clean_df)
    clean_df["trip_package_count"] = (
        clean_df.groupby(["courier_id", "trip_id"])["order_id"].transform("count")
    )
    clean_df["stop_package_count"] = (
        clean_df.groupby(["courier_id", "stop_id"])["order_id"].transform("count")
    )

    # BUG 2 FIX: previous code deduplicated before ranking, which caused NaN for
    # any row whose index wasn't in the deduplicated frame (multi-package stops).
    # rank() on the full grouped frame handles duplicates correctly via method="dense".
    clean_df["stop_index"] = (
        clean_df.groupby(["courier_id", "trip_id"])["stop_id"]
        .rank(method="dense").astype(int) - 1
    )

    trip_stop_counts = (
        clean_df.groupby(["courier_id", "trip_id"])["stop_id"].transform("nunique")
    )
    clean_df["remaining_stops"] = trip_stop_counts - clean_df["stop_index"] - 1
    clean_df["trip_stop_count"] = trip_stop_counts
    clean_df["stop_progress"]   = clean_df["stop_index"] / (trip_stop_counts - 1).clip(lower=1)
    clean_df["is_first_stop"]   = (clean_df["stop_index"] == 0).astype(int)

    clean_df["hour_sin"] = np.sin(2 * np.pi * clean_df["hour"] / 24)
    clean_df["hour_cos"] = np.cos(2 * np.pi * clean_df["hour"] / 24)
    clean_df["day_sin"]  = np.sin(2 * np.pi * clean_df["day_of_week"] / 7)
    clean_df["day_cos"]  = np.cos(2 * np.pi * clean_df["day_of_week"] / 7)

    clean_df["wait_time"]         = (
        clean_df["accept_gps_time"] - clean_df["accept_time"]
    ).dt.total_seconds() / 60
    clean_df["is_last_stop"]      = (clean_df["remaining_stops"] == 0).astype(int)
    clean_df["packages_per_stop"] = (
        clean_df["trip_package_count"] / clean_df["trip_stop_count"].clip(lower=1)
    )

    prev_stop_info = (
        clean_df.drop_duplicates(["courier_id", "trip_id", "stop_index"])
        [["courier_id", "trip_id", "stop_index",
          "delivery_gps_lat", "delivery_gps_lng", "delivery_gps_time"]]
        .copy()
        .rename(columns={
            "stop_index":        "prev_stop_index",
            "delivery_gps_lat":  "prev_delivery_lat",
            "delivery_gps_lng":  "prev_delivery_lng",
            "delivery_gps_time": "prev_delivery_gps_time",
        })
    )

    clean_df["prev_stop_index"] = clean_df["stop_index"] - 1
    clean_df = clean_df.merge(
        prev_stop_info, on=["courier_id", "trip_id", "prev_stop_index"], how="left"
    )

    clean_df["segment_distance"] = haversine(
        clean_df["prev_delivery_lat"].fillna(clean_df["accept_gps_lat"]),
        clean_df["prev_delivery_lng"].fillna(clean_df["accept_gps_lng"]),
        clean_df["delivery_gps_lat"],
        clean_df["delivery_gps_lng"],
    )
    print("segment_distance.describe():\n", clean_df["segment_distance"].describe().to_string(), "\n")

    clean_df["segment_time"] = np.where(
        clean_df["is_first_stop"] == 1,
        clean_df["time_taken"],
        (clean_df["delivery_gps_time"] - clean_df["prev_delivery_gps_time"]).dt.total_seconds() / 60,
    )

    print("segment_time by is_first_stop:")
    print(clean_df.groupby("is_first_stop")["segment_time"].describe().to_string(), "\n")

    # First stops are kept: their longer segment_time (load + travel) is real behaviour
    # in the Cambodia app where couriers pick up all packages before the first delivery.
    clean_df = clean_df[clean_df["segment_time"] > 0]
    print(f"Training rows: {len(clean_df)}\n")

    X_all        = clean_df[FEATURE_COLS[:-1]]   # courier_avg_speed added after split
    y_all        = clean_df["segment_time"]
    courier_ids  = clean_df["courier_id"]

    X_train, X_temp, y_train, y_temp, cid_train, cid_temp = train_test_split(
        X_all, y_all, courier_ids, random_state=42, test_size=0.4
    )
    X_cv, X_test, y_cv, y_test, cid_cv, cid_test = train_test_split(
        X_temp, y_temp, cid_temp, random_state=42, test_size=0.5
    )

    # BUG 5 FIX: cutoff computed on y_train only — was computed on full dataset before split
    seg_p99 = y_train.quantile(0.85)
    print(f"segment_time p85 (train only): {seg_p99:.1f} min")
    mask = y_train <= seg_p99
    X_train, y_train, cid_train = X_train[mask], y_train[mask], cid_train[mask]

    # BUG 1 FIX: courier_avg_speed was computed on full dataset before split,
    # leaking the target (segment_speed = distance/time uses segment_time = target).
    # Now fitted only on training rows.
    train_rows = clean_df.loc[X_train.index].copy()
    train_rows["segment_speed"] = (
        train_rows["segment_distance"] / train_rows["segment_time"].clip(lower=0.1)
    )
    courier_speed_map = train_rows.groupby("courier_id")["segment_speed"].mean()
    global_avg_speed  = float(courier_speed_map.mean())

    print(f"\nTop 5 courier_avg_speed (km/min):")
    print(courier_speed_map.sort_values(ascending=False).head().to_string())
    print(f"Global avg: {global_avg_speed:.3f} km/min  ({global_avg_speed * 60:.1f} km/h)\n")

    def _add_speed(X_split, cid_split):
        spd = cid_split.map(courier_speed_map).fillna(global_avg_speed)
        return X_split.copy().assign(courier_avg_speed=spd.values)

    X_train = _add_speed(X_train, cid_train)
    X_cv    = _add_speed(X_cv,    cid_cv)
    X_test  = _add_speed(X_test,  cid_test)

    joblib.dump({"map": courier_speed_map.to_dict(), "global": global_avg_speed}, COURIER_SPEED_PATH)
    courier_speed = {"map": courier_speed_map.to_dict(), "global": global_avg_speed}

    randomForestModel = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )
    randomForestModel.fit(X_train, y_train)
    joblib.dump(randomForestModel, MODEL_PATH)
    joblib.dump(X_train, X_PATH)
    joblib.dump(y_train, Y_PATH)
    model = randomForestModel
    X = X_train
    y = y_train

    cv_preds   = randomForestModel.predict(X_cv)
    test_preds = randomForestModel.predict(X_test)
    cv_rmse    = np.sqrt(np.mean((cv_preds   - y_cv.values)   ** 2))
    test_rmse  = np.sqrt(np.mean((test_preds - y_test.values) ** 2))
    std = y_train.std()
    print(f"CV   RMSE: {cv_rmse:.2f} min  ({cv_rmse / std:.4f} normalized)")
    print(f"Test RMSE: {test_rmse:.2f} min  ({test_rmse / std:.4f} normalized)")
    print(f"X_train: {X_train.shape}, X_cv: {X_cv.shape}, X_test: {X_test.shape}\n")

    importances = pd.Series(randomForestModel.feature_importances_, index=X_train.columns)
    print("Feature importances:")
    print(importances.sort_values(ascending=False).to_string(), "\n")

    # Sanity check: 1 km segment, 9 AM Monday, stop 3 of 6, mid-trip
    sanity = predict(1.0, 6, 1, pd.Timestamp("2026-01-05 09:00:00"),
                     stop_index=3, remaining_stops=2, trip_stop_count=6,
                     stop_progress=0.5, is_first_stop=0, courier_id=None)
    print(f"Sanity check — 1 km, 9 AM Mon, mid-trip: {sanity:.1f} min  (expected 5–15)")


DATA_PATH = os.path.join(BASE_DIR, "data", "delivery_yt.csv")


def load_data():
    drop_cols = ["region_id", "city", "lng", "lat", "aoi_id", "aoi_type"]
    return pd.read_csv(DATA_PATH).drop(columns=drop_cols)


def clean_data(df: pd.DataFrame):
    df = df.dropna()
    df = df.drop_duplicates(subset="order_id", keep="first")
    return df


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * arcsin(sqrt(a))


def predict(segment_distance: float, trip_package_count: int, stop_package_count: int,
            accept_gps_time, stop_index: int, remaining_stops: int,
            trip_stop_count: int, stop_progress: float, is_first_stop: int,
            wait_time: float = 0.0, courier_id: int = None) -> float:
    hour = accept_gps_time.hour
    dow  = accept_gps_time.weekday()
    avg_speed = (
        courier_speed["map"].get(courier_id, courier_speed["global"])
        if courier_id is not None else courier_speed["global"]
    )
    features = pd.DataFrame([{
        "segment_distance":   segment_distance,
        "hour_sin":           np.sin(2 * np.pi * hour / 24),
        "hour_cos":           np.cos(2 * np.pi * hour / 24),
        "day_sin":            np.sin(2 * np.pi * dow / 7),
        "day_cos":            np.cos(2 * np.pi * dow / 7),
        "trip_package_count": trip_package_count,
        "stop_package_count": stop_package_count,
        "trip_stop_count":    trip_stop_count,
        "stop_index":         stop_index,
        "remaining_stops":    remaining_stops,
        "stop_progress":      stop_progress,
        "is_first_stop":      is_first_stop,
        "is_last_stop":       int(remaining_stops == 0),
        "packages_per_stop":  trip_package_count / max(trip_stop_count, 1),
        "wait_time":          wait_time,
        "courier_avg_speed":  avg_speed,
    }])
    return float(model.predict(features)[0])

# BUG 4 FIX: bare train_model() call at the bottom caused retraining on every import
if not os.path.exists(MODEL_PATH):
    train_model()

model         = joblib.load(MODEL_PATH)
courier_speed = (
    joblib.load(COURIER_SPEED_PATH)
    if os.path.exists(COURIER_SPEED_PATH)
    else {"map": {}, "global": 0.5}
)
X = joblib.load(X_PATH) if os.path.exists(X_PATH) else None
y = joblib.load(Y_PATH) if os.path.exists(Y_PATH) else None
