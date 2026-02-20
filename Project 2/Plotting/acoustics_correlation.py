"""
acoustics_correlation.py
────────────────────────
Lab 2 – Acoustic time-delay estimation with three microphones.

Computes and plots cross-correlations (r_01, r_02, r_12) and the
autocorrelation (r_00) for three microphone channels, estimates peak
lags, and checks whether they fall inside the physically expected
interval ±n_max = floor((d/c) * f_s).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

from raspi_import import raspi_import

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS  –– adjust these before running
# ══════════════════════════════════════════════════════════════════════════════
d = 0.03          # microphone spacing [m]  ← set to your actual spacing
c = 343.0         # speed of sound in air [m/s]

# ── Data file ─────────────────────────────────────────────────────────────────
# Path matches pi_plotting.py exactly; edit filename to your measurement.
filepath = os.path.join(
    script_dir, "Measurements", "Lab 2", "test\\", "test3.bin"
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (identical to pi_plotting.py)
# ══════════════════════════════════════════════════════════════════════════════
sample_period, data = raspi_import(filepath, channels=3)
data = data * 3.3 / 4096 - 3.3 / 2   # convert ADC counts → centred voltage [V]

f_s = 1.0 / sample_period             # sampling frequency [Hz]
n_max = int(np.floor((d / c) * f_s))  # maximum physically allowed lag [samples]

ch0 = data[:, 0]
ch1 = data[:, 1]
ch2 = data[:, 2]

# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_corr(x, y):
    """
    Cross-correlation  r_xy(m) = Σ_n  x(n) · y(n + m).

    Uses numpy.correlate(y0, x0, 'full') which evaluates
        Σ_n  y0[n] · x0[n − (k − (N−1))]
    so that index k corresponds to lag  m = k − (N−1).

    DC offset is removed from both signals before correlating.

    Parameters
    ----------
    x, y : array-like, same length N

    Returns
    -------
    lags : ndarray, shape (2N−1,)  –– integer lags from −(N−1) to +(N−1)
    r    : ndarray, shape (2N−1,)  –– correlation values
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(
            f"compute_corr: signals must have equal length, got {x.shape} and {y.shape}"
        )
    x0 = x - x.mean()
    y0 = y - y.mean()
    N = len(x0)
    r = np.correlate(y0, x0, mode="full")
    lags = np.arange(-(N - 1), N)
    return lags, r


def peak_lag(lags, r_xy):
    """
    Return the lag at which |r_xy| is maximum (integer samples).

    Uses absolute maximum so it handles both positive and negative peaks.
    """
    idx = np.argmax(np.abs(r_xy))
    return int(lags[idx])


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
lags_01, r_01 = compute_corr(ch0, ch1)
lags_02, r_02 = compute_corr(ch0, ch2)
lags_12, r_12 = compute_corr(ch1, ch2)
lags_00, r_00 = compute_corr(ch0, ch0)   # autocorrelation

# ══════════════════════════════════════════════════════════════════════════════
# DELAY ESTIMATES & VALIDATION TABLE
# ══════════════════════════════════════════════════════════════════════════════
cross_pairs = [
    ("ch0–ch1", lags_01, r_01),
    ("ch0–ch2", lags_02, r_02),
    ("ch1–ch2", lags_12, r_12),
]

print()
print(f"  Sampling frequency  f_s   = {f_s:.2f} Hz")
print(f"  Mic spacing         d     = {d} m")
print(f"  Speed of sound      c     = {c} m/s")
print(f"  Max allowed lag     n_max = {n_max} samples  "
      f"({n_max / f_s * 1e6:.2f} µs)\n")

col = "{:<10}  {:>20}  {:>16}  {:>16}"
print(col.format("Pair", "Peak lag (samples)", "Peak time (s)", "|lag| ≤ n_max"))
print("─" * 70)

for name, lags, r in cross_pairs:
    lag   = peak_lag(lags, r)
    dt    = lag / f_s
    ok    = abs(lag) <= n_max
    flag  = "OK" if ok else "WARNING"
    print(col.format(name, lag, f"{dt:.6f}", f"{ok}   [{flag}]"))
    if not ok:
        print(
            f"   *** |lag| = {abs(lag)} > n_max = {n_max}. "
            "Possible causes: noise / low SNR, incorrect mic spacing d, "
            "wrong sampling frequency f_s, signal clipping, room reflections, "
            "wrong channel order, or an actual delay exceeding d/c."
        )

# ── Autocorrelation peak check ────────────────────────────────────────────────
auto_peak = peak_lag(lags_00, r_00)
auto_ok   = (auto_peak == 0)
print()
print(
    f"  Autocorrelation r_00 peak lag: {auto_peak} samples  →  "
    f"{'PASS – peak at lag 0 ✓' if auto_ok else 'FAIL – peak NOT at lag 0 ✗'}"
)
print()

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_corr(ax, lags, r, lag_star, title, color="steelblue", n_max_val=None):
    """Plot a single correlation sequence on *ax* with peak and ±n_max markers."""
    ax.plot(lags, r, color=color, linewidth=0.7, label="Correlation")
    ax.axvline(
        lag_star, color="crimson", linestyle="--", linewidth=1.3,
        label=f"Peak  lag = {lag_star} samples\n"
              f"({lag_star / f_s * 1e6:.2f} µs)"
    )
    ax.axhline(0, color="k", linewidth=0.4)
    if n_max_val is not None:
        ax.axvspan(
            -n_max_val, n_max_val, alpha=0.12, color="orange",
            label=f"±n_max = ±{n_max_val}"
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linewidth=0.4)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Cross-correlations
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
fig1.suptitle("Cross-correlations between microphone channels", fontsize=13)

corr_cfg = [
    (lags_01, r_01, peak_lag(lags_01, r_01), "r₀₁ : ch0 vs ch1", "steelblue"),
    (lags_02, r_02, peak_lag(lags_02, r_02), "r₀₂ : ch0 vs ch2", "seagreen"),
    (lags_12, r_12, peak_lag(lags_12, r_12), "r₁₂ : ch1 vs ch2", "darkorange"),
]

for ax, (lags, r, lag_star, title, color) in zip(axes, corr_cfg):
    plot_corr(ax, lags, r, lag_star, title, color=color, n_max_val=n_max)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Autocorrelation r_00
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(11, 4), constrained_layout=True)
fig2.suptitle("Autocorrelation r₀₀ (channel 0)", fontsize=13)

plot_corr(ax2, lags_00, r_00, auto_peak, "r₀₀ : ch0 autocorrelation",
          color="mediumpurple")
ax2.axvline(
    0, color="limegreen", linestyle=":", linewidth=1.6,
    label=f"lag = 0 reference  "
          f"({'PASS – peak here' if auto_ok else 'FAIL – peak elsewhere'})"
)
ax2.legend(fontsize=8, loc="upper right")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Delay summary bar chart (samples and seconds)
# ══════════════════════════════════════════════════════════════════════════════
pair_names  = [name for name, *_ in cross_pairs]
peak_lags   = [peak_lag(lags, r) for _, lags, r in cross_pairs]
peak_times  = [lag / f_s for lag in peak_lags]

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig3.suptitle("Estimated time delays between microphone pairs", fontsize=13)

colors_bar = ["steelblue", "seagreen", "darkorange"]

ax3a.bar(pair_names, peak_lags, color=colors_bar, edgecolor="k", linewidth=0.6)
ax3a.axhline( n_max, color="red",  linestyle="--", linewidth=1.1, label=f"+n_max = {n_max}")
ax3a.axhline(-n_max, color="red",  linestyle="--", linewidth=1.1, label=f"−n_max = {-n_max}")
ax3a.set_ylabel("Peak lag (samples)")
ax3a.set_title("Peak lag in samples")
ax3a.legend(fontsize=8)
ax3a.grid(axis="y", linewidth=0.4)

t_max = n_max / f_s
ax3b.bar(pair_names, [t * 1e6 for t in peak_times], color=colors_bar,
         edgecolor="k", linewidth=0.6)
ax3b.axhline( t_max * 1e6, color="red", linestyle="--", linewidth=1.1,
              label=f"+d/c = {t_max*1e6:.2f} µs")
ax3b.axhline(-t_max * 1e6, color="red", linestyle="--", linewidth=1.1,
              label=f"−d/c = {-t_max*1e6:.2f} µs")
ax3b.set_ylabel("Peak delay (µs)")
ax3b.set_title("Peak delay in microseconds")
ax3b.legend(fontsize=8)
ax3b.grid(axis="y", linewidth=0.4)

plt.show()
