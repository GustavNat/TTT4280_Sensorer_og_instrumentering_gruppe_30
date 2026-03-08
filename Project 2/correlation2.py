import os
import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import

# ----------------------------
# Load data
# ----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_period, data = raspi_import(
    os.path.join(script_dir, "Measurements", "Angle1", "1.bin"),
    channels=3
)

# Convert ADC codes to voltage and center roughly around 0 V
data = data * 3.3 / 4096 - 3.3 / 2

# ----------------------------
# Settings
# ----------------------------
fs = 31250          # sample rate [Hz]
Nmax = 6            # max physically possible delay in samples (search window)
use_abs_peak = True # pick peak by max |corr|
upsample = 16       # interpolation factor for visualization plot only

# ----------------------------
# Preprocess: remove average/DC
# ----------------------------
x0 = data[:, 0] - data[:, 0].mean()
x1 = data[:, 1] - data[:, 1].mean()
x2 = data[:, 2] - data[:, 2].mean()

# ----------------------------
# Correlation helpers
# ----------------------------
def xcorr_full(x, y):
    """
    Full cross-correlation using numpy.correlate.
    Returns corr, lags (integer sample lags).
    """
    corr = np.correlate(y, x, mode="full")
    lags = np.arange(-(len(x) - 1), len(x))
    return corr, lags

def restrict_lags(corr, lags, Nmax):
    mask = (lags >= -Nmax) & (lags <= Nmax)
    return corr[mask], lags[mask]

def peak_lag_integer(corr, lags, use_abs=True):
    if use_abs:
        k = np.argmax(np.abs(corr))
    else:
        k = np.argmax(corr)
    return int(lags[k]), corr[k]

def interp_corr(corr, lags, upsample):
    """
    Linearly interpolate correlation onto a finer lag grid (used for plotting only).
    """
    if upsample <= 1:
        return corr, lags.astype(float)
    lag_min = lags[0]
    lag_max = lags[-1]
    lags_fine = np.linspace(lag_min, lag_max, num=(lag_max - lag_min) * upsample + 1)
    corr_fine = np.interp(lags_fine, lags.astype(float), corr.astype(float))
    return corr_fine, lags_fine

def peak_lag_parabolic(corr, lags, use_abs=True):
    """
    Sub-sample peak via parabolic interpolation.
    Fits a parabola to the 3 points around the integer peak and
    finds the analytical vertex for fractional-sample accuracy.
    Returns (lag_peak_float, peak_value).
    """
    c = np.abs(corr) if use_abs else corr
    k = int(np.argmax(c))

    # Cannot interpolate at the boundary — fall back to integer
    if k == 0 or k == len(c) - 1:
        return float(lags[k]), corr[k]

    y0, y1, y2 = c[k - 1], c[k], c[k + 1]
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if denom == 0.0:
        return float(lags[k]), corr[k]

    delta = (y2 - y0) / denom          # fractional offset from integer peak
    return float(lags[k]) + delta, corr[k]

# ----------------------------
# Compute cross-correlations (3 pairs), restrict to ±Nmax
# ----------------------------
c01, l01 = xcorr_full(x0, x1)
c02, l02 = xcorr_full(x0, x2)
c12, l12 = xcorr_full(x1, x2)

c01m, l01m = restrict_lags(c01, l01, Nmax)
c02m, l02m = restrict_lags(c02, l02, Nmax)
c12m, l12m = restrict_lags(c12, l12, Nmax)

# Integer-sample delays
d01_i, _ = peak_lag_integer(c01m, l01m, use_abs_peak)
d02_i, _ = peak_lag_integer(c02m, l02m, use_abs_peak)
d12_i, _ = peak_lag_integer(c12m, l12m, use_abs_peak)

print(f"Integer delays:")
print(f"Delay (x0 vs x1): {d01_i} samples  ({d01_i/fs:.6f} s)")
print(f"Delay (x0 vs x2): {d02_i} samples  ({d02_i/fs:.6f} s)")
print(f"Delay (x1 vs x2): {d12_i} samples  ({d12_i/fs:.6f} s)")

# Sub-sample delays via parabolic interpolation
d01_f, _ = peak_lag_parabolic(c01m, l01m, use_abs_peak)
d02_f, _ = peak_lag_parabolic(c02m, l02m, use_abs_peak)
d12_f, _ = peak_lag_parabolic(c12m, l12m, use_abs_peak)

# Fine grid for plotting (linear interp, visualization only)
c01f, l01f = interp_corr(c01m, l01m, upsample)
c02f, l02f = interp_corr(c02m, l02m, upsample)
c12f, l12f = interp_corr(c12m, l12m, upsample)

print(f"\nParabolic interpolation delays:")
print(f"Delay (x0 vs x1): {d01_f:.4f} samples  ({d01_f/fs:.6f} s)")
print(f"Delay (x0 vs x2): {d02_f:.4f} samples  ({d02_f/fs:.6f} s)")
print(f"Delay (x1 vs x2): {d12_f:.4f} samples  ({d12_f/fs:.6f} s)")

# ----------------------------
# Autocorrelation example (should peak at lag=0)
# ----------------------------
a00, la00 = xcorr_full(x0, x0)
a00m, la00m = restrict_lags(a00, la00, Nmax)

d00_i, _ = peak_lag_integer(a00m, la00m, use_abs_peak)
d00_f, _ = peak_lag_parabolic(a00m, la00m, use_abs_peak)
a00f, la00f = interp_corr(a00m, la00m, upsample)

print(f"\nAutocorr peak lag (x0):")
print(f"Integer:             {d00_i} samples (should be 0)")
print(f"Parabolic interp:    {d00_f:.4f} samples (should be close to 0)")

# ----------------------------
# Task 4: angle estimate in [-180, 180] deg using atan2
# Use the interpolated (fractional) delays for higher resolution
# ----------------------------
def angle_from_delays_samples(n21, n31, n32):
    num = np.sqrt(3.0) * (n31 + n21)
    den = (n31 - n21 + 2.0 * n32)
    return np.degrees(np.arctan2(num, den))  # [-180, 180]

# Map to lab notation:
# mic1=0, mic2=1, mic3=2 => n21=d01, n31=d02, n32=d12
theta_i = angle_from_delays_samples(d01_i, d02_i, d12_i)
theta_f = angle_from_delays_samples(d01_f, d02_f, d12_f)

print(f"\nTheta estimate:")
print(f"From integer delays:            {theta_i:.2f} deg")
print(f"From parabolic interp delays:   {theta_f:.2f} deg")

# ----------------------------
# Plot correlations (integer and interpolated)
# ----------------------------
fig, axs = plt.subplots(4, 1, figsize=(10, 10), tight_layout=True)

axs[0].plot(l01m, c01m, label="integer grid")
axs[0].plot(l01f, c01f, label=f"interpolated x{upsample}")
axs[0].axvline(d01_f, linestyle="--")
axs[0].set_title(f"Cross-correlation x0 vs x1 (peak lag ≈ {d01_f:.3f} samples)")
axs[0].set_xlabel("Lag [samples]")
axs[0].set_ylabel("Correlation")
axs[0].legend()

axs[1].plot(l02m, c02m, label="integer grid")
axs[1].plot(l02f, c02f, label=f"interpolated x{upsample}")
axs[1].axvline(d02_f, linestyle="--")
axs[1].set_title(f"Cross-correlation x0 vs x2 (peak lag ≈ {d02_f:.3f} samples)")
axs[1].set_xlabel("Lag [samples]")
axs[1].set_ylabel("Correlation")
axs[1].legend()

axs[2].plot(l12m, c12m, label="integer grid")
axs[2].plot(l12f, c12f, label=f"interpolated x{upsample}")
axs[2].axvline(d12_f, linestyle="--")
axs[2].set_title(f"Cross-correlation x1 vs x2 (peak lag ≈ {d12_f:.3f} samples)")
axs[2].set_xlabel("Lag [samples]")
axs[2].set_ylabel("Correlation")
axs[2].legend()

axs[3].plot(la00m, a00m, label="integer grid")
axs[3].plot(la00f, a00f, label=f"interpolated x{upsample}")
axs[3].axvline(0, linestyle="--")
axs[3].axvline(d00_f, linestyle=":")
axs[3].set_title(f"Autocorrelation of x0 (peak lag ≈ {d00_f:.3f} samples; should be 0)")
axs[3].set_xlabel("Lag [samples]")
axs[3].set_ylabel("Correlation")
axs[3].legend()

plt.show()