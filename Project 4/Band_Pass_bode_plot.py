import matplotlib.pyplot as plt
import numpy as np


file = "Active_band_pass_filter_2_frequency_response.csv"

data = np.genfromtxt(file,
                     delimiter=",", skip_header=1)

freq    = data[:, 0]
gain_dB = data[:, 2] - data[:, 1]   # Ch2 - Ch1 (dB)
phase   = data[:, 3]                 # Ch2 phase (deg)

# --- Peak ---
peak_idx = np.argmax(gain_dB)
f0 = freq[peak_idx]
g0 = gain_dB[peak_idx]

# --- Measured -3 dB cutoff frequencies ---
cutoff_level = g0 - 3.0

# Find lower cutoff: last crossing below peak where gain crosses cutoff_level
lower_mask = freq < f0
upper_mask = freq > f0

def find_cutoff(f, g, level):
    """Linear interpolation to find frequency where g crosses level."""
    for i in range(len(g) - 1):
        if (g[i] - level) * (g[i + 1] - level) < 0:
            # Linear interpolation in log-frequency space
            log_f = np.interp(level, [g[i], g[i + 1]], [np.log10(f[i]), np.log10(f[i + 1])])
            return 10 ** log_f
    return None

f_low_meas = find_cutoff(freq[lower_mask], gain_dB[lower_mask], cutoff_level)
f_high_meas = find_cutoff(freq[upper_mask], gain_dB[upper_mask], cutoff_level)

# --- Theoretical cutoff frequencies ---
f_low_theo  = 3.5
f_high_theo = 2800.0

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Active Band-Pass Filter 1 — Bode Plot", fontsize=13)

# -- Amplitude --
ax1.semilogx(freq, gain_dB, color="steelblue", linewidth=1.8, label="Measured response", zorder=3)
ax1.axhline(cutoff_level, color="gray", linestyle=":", linewidth=1.2, label=f"−3 dB level ({cutoff_level:.1f} dB)")
ax1.axvline(f0, color="steelblue", linestyle="--", linewidth=1.2, label=f"Measured peak: {f0:.1f} Hz")
if f_low_meas:
    ax1.axvline(f_low_meas,  color="darkorange", linestyle="-",  linewidth=1.4,
                label=f"Measured $f_{{c,low}}$: {f_low_meas:.1f} Hz")
if f_high_meas:
    ax1.axvline(f_high_meas, color="darkorange", linestyle="--", linewidth=1.4,
                label=f"Measured $f_{{c,high}}$: {f_high_meas:.0f} Hz")
ax1.axvline(f_low_theo,  color="tomato", linestyle="-",  linewidth=1.4,
            label=f"Theoretical $f_{{c,low}}$: {f_low_theo} Hz")
ax1.axvline(f_high_theo, color="tomato", linestyle="--", linewidth=1.4,
            label=f"Theoretical $f_{{c,high}}$: {f_high_theo/1000:.1f} kHz")
ax1.set_ylabel("Gain (dB)")
ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
ax1.legend(loc="lower left", fontsize=8.5)

# -- Phase --
ax2.semilogx(freq, phase, color="seagreen", linewidth=1.8, label="Measured phase", zorder=3)
ax2.axvline(f0, color="steelblue", linestyle="--", linewidth=1.2, label=f"Measured peak: {f0:.1f} Hz")
if f_low_meas:
    ax2.axvline(f_low_meas,  color="darkorange", linestyle="-",  linewidth=1.4,
                label=f"Measured $f_{{c,low}}$: {f_low_meas:.1f} Hz")
if f_high_meas:
    ax2.axvline(f_high_meas, color="darkorange", linestyle="--", linewidth=1.4,
                label=f"Measured $f_{{c,high}}$: {f_high_meas:.0f} Hz")
ax2.axvline(f_low_theo,  color="tomato", linestyle="-",  linewidth=1.4,
            label=f"Theoretical $f_{{c,low}}$: {f_low_theo} Hz")
ax2.axvline(f_high_theo, color="tomato", linestyle="--", linewidth=1.4,
            label=f"Theoretical $f_{{c,high}}$: {f_high_theo/1000:.1f} kHz")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Phase (deg)")
ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
ax2.legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
plt.savefig("bode_plot.png", dpi=150)
plt.show()
