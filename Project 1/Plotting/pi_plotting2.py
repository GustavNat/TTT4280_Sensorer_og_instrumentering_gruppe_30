import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import
import os

# --- Load data ---
script_dir = os.path.dirname(os.path.abspath(__file__))
fs = 31250.0  # Hz

sample_period, data = raspi_import(
    os.path.join(script_dir, 'Measurements', 'Lab 2', 'test' 'test1.bin'),
    channels=3
)

# Convert to voltage and center around 0 V
data = data * 3.3/4096 - 3.3/2

the_data0 = data[:, 0] - 0.2
the_data1 = data[:, 1]
the_data2 = data[:, 2] + 0.2

# Time in ms, 0 corresponds to 0.2 s of actual data
time = (np.linspace(0, 1, len(the_data0)) - 0.2) * 1000

# --- Plot time domain ---
plt.plot(time, the_data0, label=r"$S_1$")
plt.plot(time, the_data1, label=r"$S_2$")
plt.plot(time, the_data2, label=r"$S_3$")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.title("Measured sine wave")
plt.xlim(0, None)
plt.ylim(-2, 2)
plt.grid()
plt.legend()
plt.show()

# ---------------- SNR (time-domain, ideal-vs-noise) ----------------
x = the_data0.copy()
N = len(x)
t = np.arange(N) / fs

f0 = 8000.0  # set this to match the file you load (1kHz.bin -> 1000.0)

# Fit ideal sine: x_hat[n] = a*sin(wt) + b*cos(wt) + c
w = 2*np.pi*f0
A = np.column_stack((np.sin(w*t), np.cos(w*t), np.ones_like(t)))
theta, *_ = np.linalg.lstsq(A, x, rcond=None)  # [a, b, c]
x_ideal = A @ theta
noise = x - x_ideal

P_sig = np.mean(x_ideal**2)
P_noise = np.mean(noise**2)

SNR_lin = P_sig / P_noise
SNR_dB = 10*np.log10(SNR_lin)

print(f"Time-domain SNR (ideal fit @ {f0:.1f} Hz): {SNR_dB:.2f} dB")
print(f"  Signal RMS: {np.sqrt(P_sig):.6f} V")
print(f"  Noise  RMS: {np.sqrt(P_noise):.6f} V")

# Optional: sanity plot (measured vs ideal + residual)
# plt.figure()
# plt.plot(time, x, label="Measured")
# plt.plot(time, x_ideal, label="Ideal (fit)", linewidth=2)
# plt.plot(time, noise, label="Noise (meas-ideal)")
# plt.legend(); plt.grid(); plt.xlabel("Time (ms)"); plt.show()
