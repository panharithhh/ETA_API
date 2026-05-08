import pandas as pd
import numpy as np
from numpy import radians, sin, cos, sqrt, arcsin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
BASE_DIR = "/Users/cheapanharith/AI/chonchoun"
MODEL_PATH = os.path.join(BASE_DIR, "model_1.pkl")
X_PATH    = os.path.join(BASE_DIR, "X_train.pkl")
Y_PATH    = os.path.join(BASE_DIR, "y_train.pkl")

#order_id,region_id,city,courier_id,lng,lat,aoi_id,aoi_type,
#accept_time,accept_gps_time,accept_gps_lng,accept_gps_lat,
#delivery_time,delivery_gps_time,delivery_gps_lng,delivery_gps_lat,ds

def assign_trips(df):
    def label(group):
        trip_id = 0
        trip_start = None
        ids = []
        for t in group["accept_time"]:
            if (trip_start is None or
                    t.date() != trip_start.date() or
                    (t - trip_start).total_seconds() > 3600):
                trip_id += 1
                trip_start = t
            ids.append(trip_id)
        group = group.copy()
        group["trip_id"] = ids
        return group
    return df.sort_values(["courier_id", "accept_time"]).groupby("courier_id", group_keys=False).apply(label)

def assign_stops(df):
    def label(group):
        stop_id = 0
        stop_start = None
        
        ids = []
        for t in group["delivery_time"]:
            if (stop_start is None or
                    t.date() != stop_start.date() or
                    (t - stop_start).total_seconds() > 300):
                stop_id += 1
                stop_start = t
            ids.append(stop_id)
        group = group.copy()
        group["stop_id"] = ids
        return group
    return df.sort_values(["courier_id", "delivery_time"]).groupby("courier_id", group_keys=False).apply(label)

def train_model():
    global model, X, y
    df = load_data()
    clean_df = clean_data(df)

    clean_df["accept_time"]   = pd.to_datetime(clean_df["accept_time"],   format="%m-%d %H:%M:%S")
    clean_df["delivery_time"] = pd.to_datetime(clean_df["delivery_time"], format="%m-%d %H:%M:%S")

    clean_df["time_taken"] = (clean_df["delivery_time"] - clean_df["accept_time"]).dt.total_seconds() / 60
    clean_df = clean_df[clean_df["time_taken"] <= 300]

    accept_gps_lat  = clean_df["accept_gps_lat"]
    accept_gps_lng  = clean_df["accept_gps_lng"]
    delivery_gps_lat = clean_df["delivery_gps_lat"]
    delivery_gps_lng = clean_df["delivery_gps_lng"]

    clean_df["distance"] = haversine(accept_gps_lat, accept_gps_lng, delivery_gps_lat, delivery_gps_lng)
    clean_df = clean_df[clean_df["distance"] <= 50]
    clean_df["hour"]        = clean_df["accept_time"].dt.hour
    clean_df["day_of_week"] = clean_df["accept_time"].dt.day_of_week

    clean_df = assign_trips(clean_df)
    clean_df = assign_stops(clean_df)
    clean_df["trip_package_count"] = clean_df.groupby(["courier_id", "trip_id"])["order_id"].transform("count")
    clean_df["stop_package_count"] = clean_df.groupby(["courier_id", "stop_id"])["order_id"].transform("count")
    clean_df["vehicle_type"]       = 0  
    
    
    X = clean_df[["distance", "hour", "day_of_week", "trip_package_count", "stop_package_count", "vehicle_type"]]
    
    
    y = clean_df["time_taken"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    randomForestModel = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    )

    randomForestModel.fit(X_train, y_train)
    joblib.dump(randomForestModel, MODEL_PATH)
    joblib.dump(X, X_PATH)
    joblib.dump(y, Y_PATH)
    model = randomForestModel

    preds = randomForestModel.predict(X_test)
    test_rmse = np.sqrt(np.mean((preds - y_test.values) ** 2))
    print(f"Error rate: {test_rmse / y_train.std():.4f}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")


def load_data():
    drop_cols = ["region_id", "city", "lng", "lat", "aoi_id", "aoi_type", "accept_gps_time", "delivery_gps_time"]
    data_yt = pd.read_csv("LaDe/delivery/delivery_yt.csv").drop(columns=drop_cols)
    return pd.concat([data_yt], ignore_index=True)

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

def predict(data, trip_package_count: int, stop_package_count: int, vehicle_type: int = 0, accept_time=None):
    distance = haversine(
        data.accept_gps_lat, data.accept_gps_lng,
        data.delivery_gps_lat, data.delivery_gps_lng
    )
    t = accept_time or data.accept_time
    hour = t.hour
    day_of_week = t.weekday()

    features = pd.DataFrame([{
        "distance":           distance,
        "hour":               hour,
        "day_of_week":        day_of_week,
        "trip_package_count": trip_package_count,
        "stop_package_count": stop_package_count,
        "vehicle_type":       vehicle_type,
    }])

    return float(model.predict(features)[0])

if not os.path.exists(MODEL_PATH):
    train_model()

model = joblib.load(MODEL_PATH)
X = joblib.load(X_PATH) if os.path.exists(X_PATH) else None
y = joblib.load(Y_PATH) if os.path.exists(Y_PATH) else None
    