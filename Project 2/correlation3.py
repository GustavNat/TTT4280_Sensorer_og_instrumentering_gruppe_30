import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import

# ----------------------------
# Settings (same as correlation2.py)
# ----------------------------
fs       = 31250
Nmax     = 6
use_abs  = True
upsample = 16   # for visualization only

script_dir  = os.path.dirname(os.path.abspath(__file__))
meas_dir    = os.path.join(script_dir, "Measurements")

# Angle folders to process (all that exist)
angle_folders = sorted(
    d for d in glob.glob(os.path.join(meas_dir, "Angle*"))
    if os.path.isdir(d)
)

# ----------------------------
# Helpers (identical to correlation2.py)
# ----------------------------
def xcorr_full(x, y):
    corr = np.correlate(y, x, mode="full")
    lags = np.arange(-(len(x) - 1), len(x))
    return corr, lags

def restrict_lags(corr, lags, Nmax):
    mask = (lags >= -Nmax) & (lags <= Nmax)
    return corr[mask], lags[mask]

def peak_lag_parabolic(corr, lags, use_abs=True):
    c = np.abs(corr) if use_abs else corr
    k = int(np.argmax(c))
    if k == 0 or k == len(c) - 1:
        return float(lags[k])
    y0, y1, y2 = c[k - 1], c[k], c[k + 1]
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if denom == 0.0:
        return float(lags[k])
    delta = (y2 - y0) / denom
    return float(lags[k]) + delta

def angle_from_delays_samples(n21, n31, n32):
    num = np.sqrt(3.0) * (n31 + n21)
    den = n31 - n21 + 2.0 * n32
    return np.degrees(np.arctan2(num, den))

def estimate_angle(filepath):
    """Load one .bin file and return the estimated angle in degrees."""
    _, data = raspi_import(filepath, channels=3)
    data = data * 3.3 / 4096 - 3.3 / 2

    x0 = data[:, 0] - data[:, 0].mean()
    x1 = data[:, 1] - data[:, 1].mean()
    x2 = data[:, 2] - data[:, 2].mean()

    c01, l01 = xcorr_full(x0, x1); c01m, l01m = restrict_lags(c01, l01, Nmax)
    c02, l02 = xcorr_full(x0, x2); c02m, l02m = restrict_lags(c02, l02, Nmax)
    c12, l12 = xcorr_full(x1, x2); c12m, l12m = restrict_lags(c12, l12, Nmax)

    d01 = peak_lag_parabolic(c01m, l01m, use_abs)
    d02 = peak_lag_parabolic(c02m, l02m, use_abs)
    d12 = peak_lag_parabolic(c12m, l12m, use_abs)

    return angle_from_delays_samples(d01, d02, d12)

# ----------------------------
# Process every angle folder
# ----------------------------
results = {}   # folder_name -> list of estimated angles

for folder in angle_folders:
    name  = os.path.basename(folder)
    files = sorted(glob.glob(os.path.join(folder, "*.bin")))

    angles = []
    for f in files:
        try:
            theta = estimate_angle(f)
            angles.append(theta)
        except Exception as e:
            print(f"  WARNING: could not process {os.path.basename(f)}: {e}")

    results[name] = np.array(angles)

# ----------------------------
# Print statistics
# ----------------------------
print(f"{'Angle':>8}  {'N':>3}  {'Mean [°]':>10}  {'Std [°]':>10}  {'Var [°²]':>12}  Estimates [°]")
print("-" * 80)
for name, angles in results.items():
    if len(angles) == 0:
        print(f"{name:>8}  no valid measurements")
        continue
    mean = np.mean(angles)
    std  = np.std(angles, ddof=1)   # sample std dev
    var  = np.var(angles,  ddof=1)  # sample variance
    est_str = "  ".join(f"{a:6.2f}" for a in angles)
    print(f"{name:>8}  {len(angles):>3}  {mean:>10.2f}  {std:>10.4f}  {var:>12.6f}  [{est_str}]")

# ----------------------------
# Plot
# ----------------------------
names  = list(results.keys())
n_ang  = len(names)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Left: scatter of all estimates per angle + mean ± std
ax = axes[0]
means = []
stds  = []
for i, name in enumerate(names):
    angles = results[name]
    if len(angles) == 0:
        means.append(np.nan); stds.append(np.nan)
        continue
    m = np.mean(angles)
    s = np.std(angles, ddof=1)
    means.append(m); stds.append(s)
    ax.scatter([i] * len(angles), angles, zorder=3, label="_nolegend_")
    ax.errorbar(i, m, yerr=s, fmt="D", color="black", capsize=6, zorder=4)

ax.set_xticks(range(n_ang))
ax.set_xticklabels(names)
ax.set_ylabel("Estimated angle [°]")
ax.set_title("Estimated angles per measurement folder\n(◆ = mean, bars = ±1 std dev)")
ax.grid(True, axis="y")

# Right: bar chart of std dev per angle
ax2 = axes[1]
valid = [(name, s) for name, s in zip(names, stds) if not np.isnan(s)]
ax2.bar([v[0] for v in valid], [v[1] for v in valid], color="steelblue")
ax2.set_ylabel("Standard deviation [°]")
ax2.set_title("Measurement uncertainty (std dev) per angle")
ax2.grid(True, axis="y")

fig.suptitle("Systematic angle variation — Task 5", fontsize=13)
plt.show()
