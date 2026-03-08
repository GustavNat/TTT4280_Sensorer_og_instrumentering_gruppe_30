import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import sys
import matplotlib.pyplot as plt 
from raspi_import import raspi_import


#test4: 100Hz
#test5: 1kHz
#test6: 18kHz

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
# sample_period, data = raspi_import(os.path.join(script_dir, 'Measurements', 'ADC', '1kHz.bin'))
# sample_period, data = raspi_import(os.path.join(script_dir, 'Measurements', 'Lab 2', 'test\\' 'test3.bin'), channels=3)
sample_period, data = raspi_import(os.path.join(script_dir, 'Measurements', 'test\\' 'test14.bin'), channels=3)
data = data * 3.3/4096 - 3.3/2  # Convert to voltage and center around 0V
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Inputs / settings
# ----------------------------
fs = 31250          # sample rate [Hz] (optional, only used if you want delay in seconds)
Nmax = 6            # max physically possible delay in samples (search window)
use_abs_peak = True # True: pick peak by max |corr|, False: max corr

# ----------------------------
# Preprocess (remove average / DC)
# data must be an Nx3 array: columns are mic0, mic1, mic2
# ----------------------------
x0 = data[:, 0] - data[:, 0].mean()
x1 = data[:, 1] - data[:, 1].mean()
x2 = data[:, 2] - data[:, 2].mean()

# ----------------------------
# Correlation helpers
# ----------------------------
def xcorr_full(x, y):
    """
    Full cross-correlation r_xy[lag] = sum_n y[n] * x[n-lag]
    Returns corr, lags (in samples)
    """
    corr = np.correlate(y, x, mode="full")
    lags = np.arange(-(len(x) - 1), len(x))
    return corr, lags

def peak_lag_limited(corr, lags, Nmax, use_abs=True):
    """
    Find peak lag within [-Nmax, +Nmax].
    Returns: best_lag, best_value, corr_limited, lags_limited
    """
    mask = (lags >= -Nmax) & (lags <= Nmax)
    corr_m = corr[mask]
    lags_m = lags[mask]

    if use_abs:
        k = np.argmax(np.abs(corr_m))
    else:
        k = np.argmax(corr_m)

    return int(lags_m[k]), corr_m[k], corr_m, lags_m

# ----------------------------
# Compute cross-correlations (3 pairs)
# ----------------------------
c01, l01 = xcorr_full(x0, x1)
c02, l02 = xcorr_full(x0, x2)
c12, l12 = xcorr_full(x1, x2)

d01, v01, c01m, l01m = peak_lag_limited(c01, l01, Nmax, use_abs_peak)
d02, v02, c02m, l02m = peak_lag_limited(c02, l02, Nmax, use_abs_peak)
d12, v12, c12m, l12m = peak_lag_limited(c12, l12, Nmax, use_abs_peak)

print(f"Delay (x0 vs x1): {d01} samples  ({d01/fs:.6f} s)")
print(f"Delay (x0 vs x2): {d02} samples  ({d02/fs:.6f} s)")
print(f"Delay (x1 vs x2): {d12} samples  ({d12/fs:.6f} s)")

# ----------------------------
# Autocorrelation example (x0 with itself)
# Should peak at lag = 0
# ----------------------------
a00, la00 = xcorr_full(x0, x0)
d00, v00, a00m, la00m = peak_lag_limited(a00, la00, Nmax, use_abs_peak)

print(f"Autocorr peak lag (x0): {d00} samples (should be 0)")

# ----------------------------
# Plot (only the physically valid lag window ±Nmax)
# ----------------------------
fig, axs = plt.subplots(4, 1, figsize=(10, 10), tight_layout=True)

axs[0].plot(l01m, c01m)
axs[0].axvline(d01, linestyle="--")
axs[0].set_title(f"Cross-correlation x0 vs x1 (peak lag = {d01} samples)")
axs[0].set_xlabel("Lag [samples]")
axs[0].set_ylabel("Correlation")

axs[1].plot(l02m, c02m)
axs[1].axvline(d02, linestyle="--")
axs[1].set_title(f"Cross-correlation x0 vs x2 (peak lag = {d02} samples)")
axs[1].set_xlabel("Lag [samples]")
axs[1].set_ylabel("Correlation")

axs[2].plot(l12m, c12m)
axs[2].axvline(d12, linestyle="--")
axs[2].set_title(f"Cross-correlation x1 vs x2 (peak lag = {d12} samples)")
axs[2].set_xlabel("Lag [samples]")
axs[2].set_ylabel("Correlation")

axs[3].plot(la00m, a00m)
axs[3].axvline(0, linestyle="--")
axs[3].set_title(f"Autocorrelation of x0 (peak lag found = {d00} samples; should be 0)")
axs[3].set_xlabel("Lag [samples]")
axs[3].set_ylabel("Correlation")

plt.show()

# ----------------------------
# Task 4: Estimate incidence angle theta in [-180, 180] deg
# Using lab eq. (29) but implemented with atan2 to get full quadrant coverage.
# Map delays to n21, n31, n32 (rename if your pair choices differ)
# ----------------------------

def angle_from_delays_samples(n21, n31, n32):
    num = np.sqrt(3.0) * (n31 + n21)
    den = (n31 - n21 + 2*n32)
    theta_rad = np.arctan2(num, den)   # returns angle in [-pi, pi]
    return np.degrees(theta_rad)       # returns angle in [-180, 180]

# Your delays:
# d01 is delay for x0 vs x1, etc.
# If you label mic1=0, mic2=1, mic3=2, then:
# n21 = delay (mic2 relative mic1) -> (1 relative 0)  -> d01
# n31 = delay (mic3 relative mic1) -> (2 relative 0)  -> d02
# n32 = delay (mic3 relative mic2) -> (2 relative 1)  -> d12
n21, n31, n32 = d01, d02, d12

theta_deg = angle_from_delays_samples(n21, n31, n32)
print(f"Theta estimate: {theta_deg:.2f} deg (range [-180, 180])")

# Optional: show that the formula covers the full range by sweeping some synthetic delays
# (this is just a demo plot; your real measurements come from rotating the source)
demo = False
if demo:
    vals = []
    for n21_demo in range(-Nmax, Nmax+1):
        for n31_demo in range(-Nmax, Nmax+1):
            for n32_demo in range(-Nmax, Nmax+1):
                vals.append(angle_from_delays_samples(n21_demo, n31_demo, n32_demo))
    vals = np.array(vals)
    plt.figure()
    plt.hist(vals, bins=60)
    plt.title("Possible theta values from integer delays (demo)")
    plt.xlabel("Theta [deg]")
    plt.ylabel("Count")
    plt.show()