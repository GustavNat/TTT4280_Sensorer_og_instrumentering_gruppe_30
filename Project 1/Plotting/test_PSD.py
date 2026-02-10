
from raspi_import import raspi_import
import matplotlib.pyplot as plt
import numpy as np



import os





def get_idx(f):
    return np.argmin(np.abs(freqs - f))


script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'Measurements', 'ADC', 'test6.bin')

sample_period, all_data = raspi_import(path, 3)
Fs = 1 / sample_period
F = 1000
ch1 = all_data[:, 1] * 0.0008; ch1 -= np.mean(ch1)
n = np.arange(0, len(ch1))

ideal_ch1 = np.sin(2*np.pi*F / Fs * n)

# plt.plot(ideal_ch1, label="ideal signal")
# plt.plot(ch1, label = "realized signal")
# plt.legend()
# plt.xlim(0, 300)
# plt.show()
#print(all_data)
#print(len(ch1))


N_zero_pad = len(ch1)*1
art_tresh = 1e-15

ideal_fft = np.fft.rfft(ideal_ch1, N_zero_pad)
ideal_pds = np.abs(ideal_fft)**2; ideal_pds[ideal_pds <= 0] = art_tresh
ideal_pds_db = 20*np.log10(ideal_pds)

realized_fft = np.fft.rfft(ch1, N_zero_pad)
realized_pds = np.abs(realized_fft)**2
realized_pds_db = 20*np.log10(realized_pds)

freqs = np.fft.rfftfreq(N_zero_pad) * Fs

possible_noise_pds = realized_pds - ideal_pds
possible_noise_pds[possible_noise_pds < 0] = art_tresh

possible_noise_pds_db = 20*np.log10(possible_noise_pds)

noise_idx = get_idx(10_000)
print(freqs[noise_idx])


possible_actual_noise_freqs = freqs[noise_idx:]
possible_actual_noise = possible_noise_pds[noise_idx:]
possible_actual_noise_pds_db = 20*np.log10(possible_actual_noise)


plt.plot(freqs, realized_pds_db, alpha=0.7, label="Realized PDS")
plt.plot(freqs, ideal_pds_db, alpha=0.3, label = "Ideal PDS")
plt.ylim(-100, 200)
plt.legend()
plt.show()
plt.plot(freqs, possible_noise_pds_db)
plt.xlim(800, 1200)


noise_sig = ideal_ch1 - ch1
noise_fft = np.fft.rfft(noise_sig, N_zero_pad)
noise_pds = np.abs(noise_fft)**2
noise_pds = noise_pds[noise_idx:]
noise_pds_db = 20*np.log10(noise_pds)


plt.plot(possible_actual_noise_freqs, noise_pds_db)
plt.show()

print(np.mean(possible_actual_noise), np.mean(noise_pds_db) / (Fs * N_zero_pad ))


# plt.plot(possible_actual_noise_freqs, possible_actual_noise_pds_db, label="Noise PDS")
# plt.legend()
# plt.show()


# Time-domain noise
noise = ch1 - ideal_ch1

# FFT of noise, optionally zero-padded
N_fft = len(noise)
noise_fft = np.fft.rfft(noise, N_fft)

# PSD per Hz
psd_noise = (1 / (Fs * N_fft)) * np.abs(noise_fft)**2

# One-sided PSD for real signals
psd_noise[1:-1] *= 2  # double all bins except DC and Nyquist


# Average PSD
mean_psd = np.mean(psd_noise)
print(mean_psd)  # should now match σ_e^2 / Fs