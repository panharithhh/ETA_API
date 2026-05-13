"""
Experiment: does adding region_id + lng/lat improve segment-time predictions?
Trains baseline (current features) vs enhanced (+ geo features), prints metrics,
and plots a 4-panel comparison.
"""
import pandas as pd
import numpy as np
from numpy import radians, sin, cos, sqrt, arcsin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── helpers ───────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * arcsin(sqrt(a))

def assign_trips(df):
    def label(g):
        tid, ts, ids = 0, None, []
        for t in g["accept_time"]:
            if ts is None or t.date() != ts.date() or (t - ts).total_seconds() > 3600:
                tid += 1; ts = t
            ids.append(tid)
        g = g.copy(); g["trip_id"] = ids; return g
    return df.sort_values(["courier_id","accept_time"]).groupby("courier_id", group_keys=False).apply(label)

def assign_stops(df):
    def label(g):
        sid, ss, ids = 0, None, []
        for t in g["delivery_time"]:
            if ss is None or t.date() != ss.date() or (t - ss).total_seconds() > 300:
                sid += 1; ss = t
            ids.append(sid)
        g = g.copy(); g["stop_id"] = ids; return g
    return df.sort_values(["courier_id","delivery_time"]).groupby("courier_id", group_keys=False).apply(label)

# ── load & clean ──────────────────────────────────────────────────────────────
print("Loading data…")
drop_cols = ["city", "aoi_id", "aoi_type"]   # keep region_id, lng, lat this time
df = pd.read_csv("LaDe/delivery/delivery_yt.csv").drop(columns=drop_cols)
df = df.dropna().drop_duplicates(subset="order_id", keep="first")

for col, fmt in [("accept_time","%m-%d %H:%M:%S"), ("delivery_time","%m-%d %H:%M:%S"),
                 ("accept_gps_time","%m-%d %H:%M:%S"), ("delivery_gps_time","%m-%d %H:%M:%S")]:
    df[col] = pd.to_datetime(df[col], format=fmt)

df["time_taken"] = (df["delivery_gps_time"] - df["accept_gps_time"]).dt.total_seconds() / 60
p99 = df["time_taken"].quantile(0.99)
df  = df[(df["time_taken"] > 0) & (df["time_taken"] <= p99)]

df["distance"] = haversine(df["accept_gps_lat"], df["accept_gps_lng"],
                            df["delivery_gps_lat"], df["delivery_gps_lng"])
df = df[df["distance"] <= 50]

df["hour"]        = df["accept_gps_time"].dt.hour
df["day_of_week"] = df["accept_gps_time"].dt.day_of_week
df["hour_sin"]    = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"]    = np.cos(2*np.pi*df["hour"]/24)
df["day_sin"]     = np.sin(2*np.pi*df["day_of_week"]/7)
df["day_cos"]     = np.cos(2*np.pi*df["day_of_week"]/7)
df["wait_time"]   = (df["accept_gps_time"] - df["accept_time"]).dt.total_seconds() / 60

df = assign_trips(df); df = assign_stops(df)
df["trip_package_count"] = df.groupby(["courier_id","trip_id"])["order_id"].transform("count")
df["stop_package_count"] = df.groupby(["courier_id","stop_id"])["order_id"].transform("count")

stop_rank = (df.drop_duplicates(["courier_id","trip_id","stop_id"])
               .groupby(["courier_id","trip_id"])["stop_id"]
               .rank(method="dense").astype(int))
df["stop_index"] = df.index.map(stop_rank) - 1

tsc = df.groupby(["courier_id","trip_id"])["stop_id"].transform("nunique")
df["remaining_stops"]  = tsc - df["stop_index"] - 1
df["trip_stop_count"]  = tsc
df["stop_progress"]    = df["stop_index"] / (tsc - 1).clip(lower=1)
df["is_first_stop"]    = (df["stop_index"] == 0).astype(int)
df["is_last_stop"]     = (df["remaining_stops"] == 0).astype(int)
df["packages_per_stop"]= df["trip_package_count"] / df["trip_stop_count"].clip(lower=1)

prev = (df.drop_duplicates(["courier_id","trip_id","stop_index"])
          [["courier_id","trip_id","stop_index","delivery_gps_lat","delivery_gps_lng","delivery_gps_time"]]
          .copy()
          .rename(columns={"stop_index":"prev_stop_index","delivery_gps_lat":"prev_delivery_lat",
                           "delivery_gps_lng":"prev_delivery_lng","delivery_gps_time":"prev_delivery_gps_time"}))
df["prev_stop_index"] = df["stop_index"] - 1
df = df.merge(prev, on=["courier_id","trip_id","prev_stop_index"], how="left")

df["segment_distance"] = haversine(
    df["prev_delivery_lat"].fillna(df["accept_gps_lat"]),
    df["prev_delivery_lng"].fillna(df["accept_gps_lng"]),
    df["delivery_gps_lat"], df["delivery_gps_lng"])

df["segment_time"] = np.where(
    df["is_first_stop"] == 1, df["time_taken"],
    (df["delivery_gps_time"] - df["prev_delivery_gps_time"]).dt.total_seconds() / 60)
seg_p99 = df["segment_time"].quantile(0.99)
df = df[(df["segment_time"] > 0) & (df["segment_time"] <= seg_p99)]

# ── encode region_id ──────────────────────────────────────────────────────────
le = LabelEncoder()
df["region_enc"] = le.fit_transform(df["region_id"])
n_regions = df["region_enc"].nunique()
print(f"Regions: {n_regions} unique after encoding")

# normalise lng/lat to zero-mean unit-std so they're on the same scale as other features
df["lng_norm"] = (df["lng"] - df["lng"].mean()) / df["lng"].std()
df["lat_norm"] = (df["lat"] - df["lat"].mean()) / df["lat"].std()
# also add delivery GPS position as region-level context
df["dlng_norm"] = (df["delivery_gps_lng"] - df["delivery_gps_lng"].mean()) / df["delivery_gps_lng"].std()
df["dlat_norm"] = (df["delivery_gps_lat"] - df["delivery_gps_lat"].mean()) / df["delivery_gps_lat"].std()

# ── feature sets ──────────────────────────────────────────────────────────────
BASE_FEATURES = ["segment_distance",
                 "hour_sin","hour_cos","day_sin","day_cos",
                 "trip_package_count","stop_package_count","trip_stop_count",
                 "stop_index","remaining_stops","stop_progress",
                 "is_first_stop","is_last_stop","packages_per_stop","wait_time"]

GEO_FEATURES  = BASE_FEATURES + ["region_enc", "lng_norm", "lat_norm",
                                  "dlng_norm", "dlat_norm"]

y = df["segment_time"]

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_split=5,
                 min_samples_leaf=4, max_features=0.7, random_state=42, n_jobs=-1)

results = {}
models  = {}
for name, feats in [("Baseline", BASE_FEATURES), ("+ Region/Geo", GEO_FEATURES)]:
    X = df[feats]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_cv, X_te, y_cv, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    print(f"\nTraining {name}  ({len(feats)} features, {len(X_tr):,} rows)…")
    m = RandomForestRegressor(**RF_PARAMS)
    m.fit(X_tr, y_tr)
    cv_rmse   = np.sqrt(np.mean((m.predict(X_cv) - y_cv.values)**2))
    te_rmse   = np.sqrt(np.mean((m.predict(X_te) - y_te.values)**2))
    std        = y_tr.std()
    cv_r2     = 1 - np.sum((m.predict(X_cv)-y_cv.values)**2)/np.sum((y_cv.values-y_cv.mean())**2)
    te_r2     = 1 - np.sum((m.predict(X_te)-y_te.values)**2)/np.sum((y_te.values-y_te.mean())**2)
    print(f"  CV   RMSE {cv_rmse:.2f} min | error rate {cv_rmse/std:.4f} | R² {cv_r2:.4f}")
    print(f"  Test RMSE {te_rmse:.2f} min | error rate {te_rmse/std:.4f} | R² {te_r2:.4f}")
    results[name] = dict(cv_rmse=cv_rmse, te_rmse=te_rmse,
                         cv_r2=cv_r2, te_r2=te_r2, std=std,
                         feats=feats, X_cv=X_cv, y_cv=y_cv,
                         X_te=X_te, y_te=y_te, preds_te=m.predict(X_te))
    models[name]  = m

# ── plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Baseline vs + Region/Geo Features", fontsize=15, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

palette = {"Baseline": "#1f77b4", "+ Region/Geo": "#d62728"}

# 1. RMSE bar comparison
ax1 = fig.add_subplot(gs[0, 0])
names  = list(results.keys())
cv_r   = [results[n]["cv_rmse"]  for n in names]
te_r   = [results[n]["te_rmse"]  for n in names]
x      = np.arange(len(names))
w      = 0.35
b1 = ax1.bar(x - w/2, cv_r, w, label="CV",   color=[palette[n] for n in names], alpha=0.75)
b2 = ax1.bar(x + w/2, te_r, w, label="Test", color=[palette[n] for n in names], alpha=1.0, hatch="//")
for b, v in zip(list(b1)+list(b2), cv_r+te_r):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.15, f"{v:.2f}", ha="center", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel("RMSE (minutes)"); ax1.set_title("RMSE Comparison (lower = better)")
ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

# 2. R² bar comparison
ax2 = fig.add_subplot(gs[0, 1])
cv_r2_vals = [results[n]["cv_r2"] for n in names]
te_r2_vals = [results[n]["te_r2"] for n in names]
b1 = ax2.bar(x - w/2, cv_r2_vals, w, label="CV",   color=[palette[n] for n in names], alpha=0.75)
b2 = ax2.bar(x + w/2, te_r2_vals, w, label="Test", color=[palette[n] for n in names], alpha=1.0, hatch="//")
for b, v in zip(list(b1)+list(b2), cv_r2_vals+te_r2_vals):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f"{v:.3f}", ha="center", fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9)
ax2.set_ylabel("R²"); ax2.set_title("R² Comparison (higher = better)")
ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

# 3. Feature importance — geo model only, top 15
ax3 = fig.add_subplot(gs[0, 2])
geo_model = models["+ Region/Geo"]
geo_feats = results["+ Region/Geo"]["feats"]
imp  = geo_model.feature_importances_
idx  = np.argsort(imp)[-15:]
cols = [geo_feats[i] for i in idx]
vals = imp[idx]
bar_colors = ["#d62728" if c in ["region_enc","lng_norm","lat_norm","dlng_norm","dlat_norm"]
              else "#1f77b4" for c in cols]
ax3.barh(cols, vals, color=bar_colors)
ax3.set_xlabel("Importance"); ax3.set_title("Feature Importance (red = new geo features)")
ax3.grid(axis="x", alpha=0.3)

# 4 & 5. Predicted vs actual scatter for each model
for col, name in enumerate(names):
    ax = fig.add_subplot(gs[1, col])
    y_true = results[name]["y_te"].values
    y_pred = results[name]["preds_te"]
    lim    = np.percentile(y_true, 98)
    mask   = (y_true <= lim) & (y_pred <= lim)
    ax.scatter(y_true[mask], y_pred[mask], alpha=0.15, s=4,
               color=palette[name], rasterized=True)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Actual time (min)"); ax.set_ylabel("Predicted time (min)")
    ax.set_title(f"{name} — Predicted vs Actual\nTest RMSE={results[name]['te_rmse']:.2f} min, R²={results[name]['te_r2']:.3f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

# 6. Error distribution overlay
ax6 = fig.add_subplot(gs[1, 2])
for name in names:
    errs = results[name]["preds_te"] - results[name]["y_te"].values
    ax6.hist(errs, bins=80, density=True, alpha=0.5, label=name, color=palette[name])
ax6.axvline(0, color="black", lw=1, ls="--")
ax6.set_xlabel("Prediction Error (min)"); ax6.set_ylabel("Density")
ax6.set_title("Error Distribution Overlay")
ax6.legend(fontsize=9); ax6.grid(alpha=0.25)
ax6.set_xlim(-60, 60)

plt.savefig("geo_experiment.png", dpi=150, bbox_inches="tight")
print("\nSaved geo_experiment.png")
