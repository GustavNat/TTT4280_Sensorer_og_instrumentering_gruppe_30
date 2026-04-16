import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import

sample_period, data = raspi_import("Data/neg087ms_3.bin")

t = np.arange(data.shape[0]) * sample_period

fig, axes = plt.subplots(data.shape[1], 1, figsize=(12, 2.5 * data.shape[1]), sharex=True)
fig.suptitle("ADC channels — test.bin")

for i, ax in enumerate(axes):
    ax.plot(t, data[:, i], linewidth=0.8)
    ax.set_ylabel(f"Channel {i}")
    ax.grid(True, alpha=0.4)

axes[-1].set_xlabel("Time [s]")
plt.tight_layout()
plt.show()



