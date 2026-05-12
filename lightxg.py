import pandas as pd
import numpy as np
from numpy import radians, sin, cos, sqrt, arcsin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

BASE_DIR = "/Users/cheapanharith/AI/chonchoun"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * arcsin(sqrt(a))

def assign_trips(df):
    def label(group):
        trip_id, trip_start, ids = 0, None, []
        for t in group["accept_time"]:
            if (trip_start is None or t.date() != trip_start.date() or
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
        stop_id, stop_start, ids = 0, None, []
        for t in group["delivery_time"]:
            if (stop_start is None or t.date() != stop_start.date() or
                    (t - stop_start).total_seconds() > 300):
                stop_id += 1
                stop_start = t
            ids.append(stop_id)
        group = group.copy()
        group["stop_id"] = ids
        return group
    return df.sort_values(["courier_id", "delivery_time"]).groupby("courier_id", group_keys=False).apply(label)

def build_features():
    drop_cols = ["region_id", "city", "lng", "lat", "aoi_id", "aoi_type"]
    df = pd.read_csv(f"{BASE_DIR}/LaDe/delivery/delivery_yt.csv").drop(columns=drop_cols)
    df = df.dropna().drop_duplicates(subset="order_id", keep="first")

    df["accept_time"]       = pd.to_datetime(df["accept_time"],       format="%m-%d %H:%M:%S")
    df["delivery_time"]     = pd.to_datetime(df["delivery_time"],     format="%m-%d %H:%M:%S")
    df["accept_gps_time"]   = pd.to_datetime(df["accept_gps_time"],   format="%m-%d %H:%M:%S")
    df["delivery_gps_time"] = pd.to_datetime(df["delivery_gps_time"], format="%m-%d %H:%M:%S")

    df["time_taken"] = (df["delivery_gps_time"] - df["accept_gps_time"]).dt.total_seconds() / 60
    p99 = df["time_taken"].quantile(0.99)
    df = df[(df["time_taken"] > 0) & (df["time_taken"] <= p99)]

    df["distance"] = haversine(df["accept_gps_lat"], df["accept_gps_lng"],
                                df["delivery_gps_lat"], df["delivery_gps_lng"])
    df = df[df["distance"] <= 50]

    df["hour"]        = df["accept_gps_time"].dt.hour
    df["day_of_week"] = df["accept_gps_time"].dt.day_of_week

    df = assign_trips(df)
    df = assign_stops(df)
    
    df["trip_package_count"] = df.groupby(["courier_id", "trip_id"])["order_id"].transform("count")
    df["stop_package_count"] = df.groupby(["courier_id", "stop_id"])["order_id"].transform("count")

    stop_rank = (
        df.drop_duplicates(["courier_id", "trip_id", "stop_id"])
        .groupby(["courier_id", "trip_id"])["stop_id"]
        .rank(method="dense").astype(int)
    )
    df["stop_index"] = df.index.map(stop_rank) - 1

    trip_stop_counts = df.groupby(["courier_id", "trip_id"])["stop_id"].transform("nunique")
    df["remaining_stops"] = trip_stop_counts - df["stop_index"] - 1
    df["trip_stop_count"] = trip_stop_counts
    df["stop_progress"]   = df["stop_index"] / (trip_stop_counts - 1).clip(lower=1)
    df["is_first_stop"]   = (df["stop_index"] == 0).astype(int)
    df["is_last_stop"]    = (df["remaining_stops"] == 0).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["wait_time"]         = (df["accept_gps_time"] - df["accept_time"]).dt.total_seconds() / 60
    df["packages_per_stop"] = df["trip_package_count"] / df["trip_stop_count"].clip(lower=1)

    prev_stop_info = (
        df.drop_duplicates(["courier_id", "trip_id", "stop_index"])
        [["courier_id", "trip_id", "stop_index", "delivery_gps_lat", "delivery_gps_lng", "delivery_gps_time"]]
        .copy()
        .rename(columns={"stop_index":        "prev_stop_index",
                         "delivery_gps_lat":  "prev_delivery_lat",
                         "delivery_gps_lng":  "prev_delivery_lng",
                         "delivery_gps_time": "prev_delivery_gps_time"})
    )
    df["prev_stop_index"] = df["stop_index"] - 1
    df = df.merge(prev_stop_info, on=["courier_id", "trip_id", "prev_stop_index"], how="left")

    df["segment_distance"] = haversine(
        df["prev_delivery_lat"].fillna(df["accept_gps_lat"]),
        df["prev_delivery_lng"].fillna(df["accept_gps_lng"]),
        df["delivery_gps_lat"], df["delivery_gps_lng"],
    )
    df["segment_time"] = np.where(
        df["is_first_stop"] == 1,
        df["time_taken"],
        (df["delivery_gps_time"] - df["prev_delivery_gps_time"]).dt.total_seconds() / 60,
    )
    seg_p99 = df["segment_time"].quantile(0.99)
    df = df[(df["segment_time"] > 0) & (df["segment_time"] <= seg_p99)]

    X = df[["segment_distance",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "trip_package_count", "stop_package_count", "trip_stop_count",
            "stop_index", "remaining_stops", "stop_progress",
            "is_first_stop", "is_last_stop",
            "packages_per_stop", "wait_time"]]
    y = df["segment_time"]
    return X, y


def rmse_rate(preds, y_true, std):
    return np.sqrt(np.mean((preds - y_true.values) ** 2)) / std


if __name__ == "__main__":
    print("Building features...")
    X, y = build_features()

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, test_size=0.4)
    X_cv, X_test, y_cv, y_test       = train_test_split(X_temp, y_temp, random_state=42, test_size=0.5)
    std = y_train.std()
    print(f"Train: {X_train.shape[0]}  CV: {X_cv.shape[0]}  Test: {X_test.shape[0]}\n")

    # Random Forest — already tuned, use best known params
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=10, min_samples_split=5,
        min_samples_leaf=4, max_features=0.7, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # XGBoost — tune via RandomizedSearchCV
    xgb_params = {
        "n_estimators":    [200, 500, 800],
        "max_depth":       [4, 6, 8, 10],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "min_child_weight":[1, 3, 5],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        param_distributions=xgb_params,
        n_iter=30, cv=3, scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=1,
    )
    xgb_search.fit(X_train, y_train)
    print(f"XGBoost best:  {xgb_search.best_params_}")
    xgb = xgb_search.best_estimator_

    # LightGBM — tune via RandomizedSearchCV
    lgbm_params = {
        "n_estimators":   [200, 500, 800],
        "max_depth":      [4, 6, 8, 10],
        "learning_rate":  [0.01, 0.05, 0.1, 0.2],
        "subsample":      [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "num_leaves":     [31, 63, 127],
        "min_child_samples":[10, 20, 50],
    }
    lgbm_search = RandomizedSearchCV(
        LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        param_distributions=lgbm_params,
        n_iter=30, cv=3, scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=1,
    )
    lgbm_search.fit(X_train, y_train)
    print(f"LightGBM best: {lgbm_search.best_params_}")
    lgbm = lgbm_search.best_estimator_

    print(f"\n{'Model':<16} {'CV':>8} {'Test':>8}")
    print("-" * 34)
    for name, m in [("Random Forest", rf), ("XGBoost", xgb), ("LightGBM", lgbm)]:
        cv_rate   = rmse_rate(m.predict(X_cv),   y_cv,   std)
        test_rate = rmse_rate(m.predict(X_test), y_test, std)
        print(f"{name:<16} {cv_rate:>8.4f} {test_rate:>8.4f}")
