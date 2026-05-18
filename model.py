import pandas as pd
import numpy as np
from numpy import radians, sin, cos, sqrt, arcsin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.compose import TransformedTargetRegressor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model_1.pkl")
X_PATH    = os.path.join(MODELS_DIR, "X_train.pkl")
Y_PATH    = os.path.join(MODELS_DIR, "y_train.pkl")



#order_id,region_id,city,courier_id,lng,lat,aoi_id,aoi_type,
#accept_time,accept_gps_time,accept_gps_lng,accept_gps_lat,
#delivery_time,delivery_gps_time,delivery_gps_lng,delivery_gps_lat,ds

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
                    (t - stop_start).total_seconds() > 300):
                stop_id += 1
                stop_start = t
            ids.append(stop_id)
    df["stop_id"] = ids
    return df

def train_model():
    global model, X, y
    df = load_data()
    clean_df = clean_data(df)

    clean_df["accept_time"]        = pd.to_datetime(clean_df["accept_time"],        format="%m-%d %H:%M:%S")
    clean_df["delivery_time"]      = pd.to_datetime(clean_df["delivery_time"],      format="%m-%d %H:%M:%S")
    clean_df["accept_gps_time"]    = pd.to_datetime(clean_df["accept_gps_time"],    format="%m-%d %H:%M:%S")
    clean_df["delivery_gps_time"]  = pd.to_datetime(clean_df["delivery_gps_time"],  format="%m-%d %H:%M:%S")

    clean_df["time_taken"] = (clean_df["delivery_gps_time"] - clean_df["accept_gps_time"]).dt.total_seconds() / 60
    print(clean_df["time_taken"].describe())
    p99 = clean_df["time_taken"].quantile(0.99)
    print(f"99th percentile cutoff: {p99:.1f} min")
    clean_df = clean_df[clean_df["time_taken"] > 0]
    clean_df = clean_df[clean_df["time_taken"] <= p99]

    accept_gps_lat  = clean_df["accept_gps_lat"]
    accept_gps_lng  = clean_df["accept_gps_lng"]
    delivery_gps_lat = clean_df["delivery_gps_lat"]
    delivery_gps_lng = clean_df["delivery_gps_lng"]

    clean_df["distance"] = haversine(accept_gps_lat, accept_gps_lng, delivery_gps_lat, delivery_gps_lng)
    clean_df = clean_df[clean_df["distance"] <= 50]
    clean_df["hour"]        = clean_df["accept_gps_time"].dt.hour
    clean_df["day_of_week"] = clean_df["accept_gps_time"].dt.day_of_week

    clean_df = assign_trips(clean_df)
    clean_df = assign_stops(clean_df)
    clean_df["trip_package_count"] = clean_df.groupby(["courier_id", "trip_id"])["order_id"].transform("count")
    clean_df["stop_package_count"] = clean_df.groupby(["courier_id", "stop_id"])["order_id"].transform("count")
    
    stop_rank = (
        clean_df.drop_duplicates(["courier_id", "trip_id", "stop_id"])
        .groupby(["courier_id", "trip_id"])["stop_id"]
        .rank(method="dense")
        .astype(int)
    )
    clean_df["stop_index"] = clean_df.index.map(stop_rank) - 1  

    trip_stop_counts = clean_df.groupby(["courier_id", "trip_id"])["stop_id"].transform("nunique")
    clean_df["remaining_stops"] = trip_stop_counts - clean_df["stop_index"] - 1
    clean_df["trip_stop_count"] = trip_stop_counts
    clean_df["stop_progress"]   = clean_df["stop_index"] / (trip_stop_counts - 1).clip(lower=1)
    clean_df["is_first_stop"]   = (clean_df["stop_index"] == 0).astype(int)

    clean_df["hour_sin"] = np.sin(2 * np.pi * clean_df["hour"] / 24)
    clean_df["hour_cos"] = np.cos(2 * np.pi * clean_df["hour"] / 24)
    clean_df["day_sin"]  = np.sin(2 * np.pi * clean_df["day_of_week"] / 7)
    clean_df["day_cos"]  = np.cos(2 * np.pi * clean_df["day_of_week"] / 7)

    clean_df["wait_time"]         = (clean_df["accept_gps_time"] - clean_df["accept_time"]).dt.total_seconds() / 60
    clean_df["is_last_stop"]      = (clean_df["remaining_stops"] == 0).astype(int)
    clean_df["packages_per_stop"] = clean_df["trip_package_count"] / clean_df["trip_stop_count"].clip(lower=1)

    prev_stop_info = (
        clean_df.drop_duplicates(["courier_id", "trip_id", "stop_index"])
        [["courier_id", "trip_id", "stop_index", "delivery_gps_lat", "delivery_gps_lng", "delivery_gps_time"]]
        .copy()
        .rename(columns={"stop_index":          "prev_stop_index",
                         "delivery_gps_lat":    "prev_delivery_lat",
                         "delivery_gps_lng":    "prev_delivery_lng",
                         "delivery_gps_time":   "prev_delivery_gps_time"})
    )

    clean_df["prev_stop_index"] = clean_df["stop_index"] - 1
    clean_df = clean_df.merge(prev_stop_info, on=["courier_id", "trip_id", "prev_stop_index"], how="left")

    clean_df["segment_distance"] = haversine(
        clean_df["prev_delivery_lat"].fillna(clean_df["accept_gps_lat"]),
        clean_df["prev_delivery_lng"].fillna(clean_df["accept_gps_lng"]),
        clean_df["delivery_gps_lat"],
        clean_df["delivery_gps_lng"],
    )

    clean_df["segment_time"] = np.where(
        clean_df["is_first_stop"] == 1,
        clean_df["time_taken"],
        (clean_df["delivery_gps_time"] - clean_df["prev_delivery_gps_time"]).dt.total_seconds() / 60,
    )
    seg_p99 = clean_df["segment_time"].quantile(0.99)
    clean_df = clean_df[(clean_df["segment_time"] > 0) & (clean_df["segment_time"] <= seg_p99)]

    X = clean_df[["segment_distance",
                  "hour_sin", "hour_cos", "day_sin", "day_cos",
                  "trip_package_count", "stop_package_count", "trip_stop_count",
                  "stop_index", "remaining_stops", "stop_progress",
                  "is_first_stop", "is_last_stop",
                  "packages_per_stop", "wait_time"]]

    y = clean_df["segment_time"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, test_size=0.4)
    X_cv, X_test, y_cv, y_test       = train_test_split(X_temp, y_temp, random_state=42, test_size=0.5)
 

    randomForestModel = RandomForestRegressor(
        # 
        n_estimators=100, # right is 500
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )
    randomForestModel.fit(X_train, y_train)
    joblib.dump(randomForestModel, MODEL_PATH)
    joblib.dump(X, X_PATH)
    joblib.dump(y, Y_PATH)
    model = randomForestModel

    cv_preds   = randomForestModel.predict(X_cv)
    test_preds = randomForestModel.predict(X_test)
    cv_rmse    = np.sqrt(np.mean((cv_preds   - y_cv.values)   ** 2))
    test_rmse  = np.sqrt(np.mean((test_preds - y_test.values) ** 2))
    std = y_train.std()
    print(f"CV   error rate: {cv_rmse   / std:.4f}")
    print(f"Test error rate: {test_rmse / std:.4f}")
    print(f"X_train: {X_train.shape}, X_cv: {X_cv.shape}, X_test: {X_test.shape}")
    

    

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
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * arcsin(sqrt(a))

def predict(segment_distance: float, trip_package_count: int, stop_package_count: int,
            accept_gps_time, stop_index: int, remaining_stops: int,
            trip_stop_count: int, stop_progress: float, is_first_stop: int,
            wait_time: float = 0.0) -> float:
    hour = accept_gps_time.hour
    dow  = accept_gps_time.weekday()
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
    }])
    return float(model.predict(features)[0])

if not os.path.exists(MODEL_PATH):
    train_model()

model = joblib.load(MODEL_PATH)
X = joblib.load(X_PATH) if os.path.exists(X_PATH) else None
y = joblib.load(Y_PATH) if os.path.exists(Y_PATH) else None