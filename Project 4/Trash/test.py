import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from raspi_import import raspi_import
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
sample_period, data = raspi_import(os.path.join(script_dir, 'Data', 'bruh2.bin'), channels=2)

N = data.shape[0]
t = np.arange(N) * sample_period

plt.plot(t, data[:, 0], label='Ch 1')
plt.plot(t, data[:, 1], label='Ch 2')
plt.xlabel('Time [s]')
plt.ylabel('ADC value')
plt.legend()
plt.show()

for n in range(1000):
    print(data[n,0])