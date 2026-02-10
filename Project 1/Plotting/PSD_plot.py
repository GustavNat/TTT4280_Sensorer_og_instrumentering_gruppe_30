import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from raspi_import import raspi_import
import os

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_period, data = raspi_import(os.path.join(script_dir, 'Measurements', 'ADC', 'test.bin'))

# Convert to voltage and center around 0V
data = data * 3.3 / 4096 - 3.3 / 2

# Extract channels
channel_0 = data[:, 0]
channel_1 = data[:, 1]
channel_2 = data[:, 2]

# Sample rate
fs = 1 / sample_period  # Should be 31250 Hz

# Calculate PSD using Welch's method
# nperseg controls frequency resolution vs variance tradeoff
nperseg = 1024  # Length of each segment

f0, Pxx0 = signal.welch(channel_0, fs, nperseg=nperseg)
f1, Pxx1 = signal.welch(channel_1, fs, nperseg=nperseg)
f2, Pxx2 = signal.welch(channel_2, fs, nperseg=nperseg)

# Normalize to 0 dB (relative to maximum)
max_power = max(np.max(Pxx0), np.max(Pxx1), np.max(Pxx2))
Pxx0_norm = Pxx0 / max_power
Pxx1_norm = Pxx1 / max_power
Pxx2_norm = Pxx2 / max_power

# Plot normalized PSD in dB scale
plt.figure(figsize=(10, 6))

plt.plot(f0, 10 * np.log10(Pxx0_norm), label='Channel 0')
plt.plot(f1, 10 * np.log10(Pxx1_norm), label='Channel 1')
plt.plot(f2, 10 * np.log10(Pxx2_norm), label='Channel 2')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.title('Normalized Power Spectral Density')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
