import numpy as np
import sys
import matplotlib.pyplot as plt 
from raspi_import import raspi_import


#test4: 100Hz
#test5: 1kHz
#test6: 18kHz

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_period, data = raspi_import(os.path.join(script_dir, 'Measurements', 'ADC', '20Hz.bin'))
data = data * 3.3/4096 - 3.3/2  # Convert to voltage and center around 0V


the_data0 = data[:,0]-0.3
the_data1 = data[:,1]
the_data2 = data[:,2]+0.3


time = np.linspace(0, 1, len(the_data0))



N = len(the_data0)
X0 = np.fft.fft(the_data0, N*1)
f = np.fft.fftfreq(N, d=1/31250)


X0 = X0[:N//2]
f = f[:N//2]
X0[0:10] = 0


# plt.plot(f, 20*np.log10(np.abs(X0)))
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude")
# plt.title("FFT of measured sine")
# plt.grid()
# plt.show()
 

num_samples = 10000



plt.plot(time, the_data0)
plt.plot(time, the_data1)
plt.plot(time, the_data2)

plt.xlim(sample_period, num_samples*sample_period)
plt.show()



f0 = 100.0

P = np.abs(X0)**2

df = f[1] - f[0]
sig = (f > f0-df) & (f < f0+df)

SNR_dB = 20*np.log10(np.sum(P[sig]) / np.sum(P[~sig]))


