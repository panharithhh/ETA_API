import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

X = joblib.load("X_train.pkl")
y = joblib.load("y_train.pkl")

dist = X["segment_distance"].values   # km
time = y.values                        # minutes
speed = (dist / time) * 60            # km/h

# cap outliers for display
p99_speed = np.percentile(speed, 99)
p99_time  = np.percentile(time, 99)
p99_dist  = np.percentile(dist, 99)

mask = (speed <= p99_speed) & (time <= p99_time) & (dist <= p99_dist)
dist_c, time_c, speed_c = dist[mask], time[mask], speed[mask]

# ── speed buckets ─────────────────────────────────────────────────────────────
bins   = [0, 10, 20, 30, 45, np.inf]
labels = ["<10 km/h\n(very slow)", "10–20 km/h\n(slow)", "20–30 km/h\n(medium)",
          "30–45 km/h\n(fast)", ">45 km/h\n(highway)"]
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
seg_idx = np.digitize(speed_c, bins) - 1

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Delivery Segment Analysis: Distance vs Time & Speed", fontsize=15, fontweight="bold")

# ── 1. Scatter: distance vs time, coloured by speed bucket ───────────────────
ax1 = fig.add_subplot(2, 2, (1, 2))
for i, (label, color) in enumerate(zip(labels, colors)):
    sel = seg_idx == i
    ax1.scatter(dist_c[sel], time_c[sel], c=color, label=label,
                alpha=0.35, s=8, rasterized=True)

# overlay ideal speed reference lines
d_range = np.linspace(0, dist_c.max(), 200)
for ref_speed, ls in [(10, "--"), (20, "-"), (40, "--")]:
    t_ref = (d_range / ref_speed) * 60
    ax1.plot(d_range, t_ref, color="black", lw=0.8, ls=ls,
             label=f"{ref_speed} km/h reference" if ref_speed == 20 else f"_{ref_speed} km/h")
    ax1.text(d_range[-1] * 0.97, t_ref[-1], f"{ref_speed} km/h",
             fontsize=7, va="center", color="black", alpha=0.7)

ax1.set_xlabel("Segment Distance (km)", fontsize=11)
ax1.set_ylabel("Segment Time (minutes)", fontsize=11)
ax1.set_title("Distance vs Time — coloured by implied speed bucket", fontsize=11)
ax1.legend(loc="upper left", markerscale=2.5, fontsize=8, framealpha=0.85)
ax1.set_xlim(0, dist_c.max() * 1.02)
ax1.set_ylim(0, time_c.max() * 1.02)
ax1.grid(True, alpha=0.25)

# ── 2. Speed distribution (KDE + histogram) ───────────────────────────────────
ax2 = fig.add_subplot(2, 2, 3)
counts, edges, patches = ax2.hist(speed_c, bins=60, density=True, alpha=0.55,
                                  color="#1f77b4", edgecolor="none")
for patch, left in zip(patches, edges[:-1]):
    i = np.digitize(left, bins) - 1
    patch.set_facecolor(colors[min(i, len(colors) - 1)])

kde_x = np.linspace(speed_c.min(), speed_c.max(), 400)
kde   = gaussian_kde(speed_c, bw_method=0.15)
ax2.plot(kde_x, kde(kde_x), color="black", lw=1.5, label="KDE")
ax2.axvline(np.median(speed_c), color="red", lw=1.2, ls="--",
            label=f"Median {np.median(speed_c):.1f} km/h")
ax2.axvline(np.mean(speed_c),   color="orange", lw=1.2, ls=":",
            label=f"Mean {np.mean(speed_c):.1f} km/h")
ax2.set_xlabel("Implied Speed (km/h)", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("Speed Distribution across all segments", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)

# ── 3. Avg time per distance bin (d/s relationship) ───────────────────────────
ax3 = fig.add_subplot(2, 2, 4)
dist_bins   = np.linspace(0, np.percentile(dist_c, 98), 20)
bin_centers = (dist_bins[:-1] + dist_bins[1:]) / 2
bin_indices = np.digitize(dist_c, dist_bins) - 1
bin_indices = np.clip(bin_indices, 0, len(dist_bins) - 2)

means  = [time_c[bin_indices == i].mean()  if (bin_indices == i).any() else np.nan for i in range(len(bin_centers))]
stds   = [time_c[bin_indices == i].std()   if (bin_indices == i).any() else np.nan for i in range(len(bin_centers))]
means  = np.array(means)
stds   = np.array(stds)

ax3.fill_between(bin_centers, means - stds, means + stds, alpha=0.25, color="#1f77b4", label="±1 std")
ax3.plot(bin_centers, means, "o-", color="#1f77b4", lw=2, ms=5, label="Mean segment time")

# fit linear trend through origin
valid = ~np.isnan(means)
slope = np.polyfit(bin_centers[valid], means[valid], 1)
trend = np.poly1d(slope)
ax3.plot(bin_centers, trend(bin_centers), "r--", lw=1.2,
         label=f"Linear fit (slope={slope[0]:.1f} min/km)")

ax3.set_xlabel("Segment Distance (km)", fontsize=11)
ax3.set_ylabel("Segment Time (minutes)", fontsize=11)
ax3.set_title("Avg Time per Distance Bin (d/s relationship)", fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.25)
ax3.set_xlim(0)
ax3.set_ylim(0)

plt.tight_layout()
plt.savefig("segment_analysis.png", dpi=150, bbox_inches="tight")
print("Saved segment_analysis.png")
plt.show()
