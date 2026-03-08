"""
Analyse basismålingene for TTT4280 Lab 3 - Optikk.

Prosesserer 5 videoopptak (målesituasjon 1) og beregner:
  - Gjennomsnitt og standardavvik av pulsfrekvensen (BPM) på tvers av målinger
  - Spektral SNR: effekten ved pulstoppen vs. støygulvet i FFT-spekteret
  - FFT av AC-puls-signalet for hver måling og kanal

SNR (spektral) er definert som:
    SNR_dB = 10 * log10(P_peak / P_noise)

der P_peak er effekten ved dominerende pulsfrekvens og P_noise er
gjennomsnittseffekten i alle andre frekvenser innenfor fysiologisk område.

Krever: numpy, opencv-python (cv2), matplotlib, scipy
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# -------------------------------------------------------------------
# Konfigurasjon
# -------------------------------------------------------------------
MEASUREMENTS_DIR = "./Meausrements"
VIDEO_FILES = [
    # "Data_1_57bpm.mp4",
    # "Data_2_58bpm.mp4",
    # "measuremetn_11.mp4",
    # "recording_4.mp4",
    # "Data_3_56bpm.mp4",
    # "Data_4_60bpm.mp4",
    # "Data_5_58bpm.mp4",
    # "recording_5.mp4"
    "56_2.mp4",
    "56.mp4",
    "57.mp4",
    "57_2.mp4",
    "58.mp4"

]

ROI_FRACTION = 1  # sentral 50 % av rammen

CHANNEL_NAMES  = ["R", "G", "B"]
CHANNEL_COLORS = ["red", "green", "blue"]

# Fysiologisk pulsfrekvensområde [Hz]
BPM_MIN, BPM_MAX = 40, 100
F_MIN = BPM_MIN / 60
F_MAX = BPM_MAX / 60


# -------------------------------------------------------------------
# Hjelpefunksjon: ekstraher gjennomsnittssignal fra video
# -------------------------------------------------------------------
def extract_mean_signal(filepath, roi_fraction=ROI_FRACTION):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise IOError(f"Kunne ikke åpne videofil: {filepath}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    signal = np.zeros((num_frames, 3))

    ret, first_frame = cap.read()
    if not ret:
        raise IOError(f"Kunne ikke lese ramme fra: {filepath}")

    h, w = first_frame.shape[:2]
    y0 = int(h * (1 - roi_fraction) / 2)
    y1 = int(h * (1 + roi_fraction) / 2)
    x0 = int(w * (1 - roi_fraction) / 2)
    x1 = int(w * (1 + roi_fraction) / 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        roi = frame[y0:y1, x0:x1, :]
        mean_bgr = np.mean(roi, axis=(0, 1))
        signal[count, 0] = mean_bgr[2]  # R
        signal[count, 1] = mean_bgr[1]  # G
        signal[count, 2] = mean_bgr[0]  # B
        count += 1

    cap.release()
    return signal[:count], fps


# -------------------------------------------------------------------
# Hjelpefunksjon: beregn FFT og spektral SNR for ett signal
# -------------------------------------------------------------------
def analyse_pulse(signal_1d, fps):
    """
    Returnerer:
        freqs    : frekvensakse [Hz]
        power    : ensidig effektspektrum (|FFT|^2 / N^2)
        peak_hz  : dominerende pulsfrekvens [Hz]
        peak_bpm : dominerende puls [BPM]
        snr_db   : spektral SNR [dB]  (P_peak / P_noise_avg)
        mean_ac  : gjennomsnitt av AC-signalet (≈ 0 etter DC-fjerning)
        std_ac   : standardavvik av AC-signalet
    """
    N = len(signal_1d)

    # Fjern DC
    ac = signal_1d - np.mean(signal_1d)
    std_ac = np.std(ac)

    # Hann-vindu for å redusere spektrallekasje
    win = windows.hann(N)
    ac_win = ac * win

    # FFT
    fft_vals = np.fft.rfft(ac_win)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    power = (np.abs(fft_vals) ** 2) / (N ** 2)

    # Begrens til fysiologisk område
    mask = (freqs >= F_MIN) & (freqs <= F_MAX)
    if not np.any(mask):
        return freqs, power, np.nan, np.nan, np.nan, 0.0, std_ac

    peak_idx_in_mask = np.argmax(power[mask])
    peak_hz  = freqs[mask][peak_idx_in_mask]
    peak_bpm = peak_hz * 60

    # Spektral SNR: peak-effekt vs. gjennomsnittseffekt av resten
    P_peak  = power[mask][peak_idx_in_mask]
    noise_mask = mask.copy()
    # Ekskluder ±0.1 Hz rundt toppen for å unngå å ta med signalet i støyen
    noise_mask &= np.abs(freqs - peak_hz) > 0.1
    P_noise_avg = np.mean(power[noise_mask]) if np.any(noise_mask) else np.nan
    snr_db = 10 * np.log10(P_peak / P_noise_avg) if P_noise_avg and P_noise_avg > 0 else np.nan

    return freqs, power, peak_hz, peak_bpm, snr_db, 0.0, std_ac


# -------------------------------------------------------------------
# Last inn alle videoer
# -------------------------------------------------------------------
all_signals, all_fps, all_labels = [], [], []

print("=" * 65)
print("  Prosesserer videoer...")
print("=" * 65)
for fname in VIDEO_FILES:
    path = os.path.join(MEASUREMENTS_DIR, fname)
    print(f"  {fname}")
    sig, fps = extract_mean_signal(path)
    all_signals.append(sig)
    all_fps.append(fps)
    all_labels.append(os.path.splitext(fname)[0])
print()


# -------------------------------------------------------------------
# Beregn pulsstatistikk per måling
# -------------------------------------------------------------------
results = []
for sig, fps, label in zip(all_signals, all_fps, all_labels):
    entry = {"label": label, "channels": {}}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        freqs, power, peak_hz, peak_bpm, snr_db, _, std_ac = analyse_pulse(sig[:, ch_idx], fps)
        entry["channels"][ch_name] = {
            "freqs": freqs, "power": power,
            "peak_hz": peak_hz, "peak_bpm": peak_bpm,
            "snr_db": snr_db, "std_ac": std_ac,
        }
    results.append(entry)


# -------------------------------------------------------------------
# Skriv ut statistikk
# -------------------------------------------------------------------
print("=" * 65)
print("  Pulsstatistikk per måling")
print("=" * 65)
print(f"\n  {'Måling':<22} {'Kanal':<6} {'BPM':>7} {'Std AC':>9} {'SNR [dB]':>10}")
print(f"  {'-'*21:<22} {'-'*5:<6} {'-'*6:>7} {'-'*8:>9} {'-'*9:>10}")

for r in results:
    short = r['label'].replace('Data_', 'M')
    for ch_name in CHANNEL_NAMES:
        ch = r['channels'][ch_name]
        print(f"  {short:<22} {ch_name:<6} {ch['peak_bpm']:>7.1f} "
              f"{ch['std_ac']:>9.3f} {ch['snr_db']:>9.2f} dB")

# Sammendrag
print()
print("=" * 65)
print("  Gjennomsnitt og standardavvik over alle 5 målinger")
print("=" * 65)
print(f"\n  {'Kanal':<6} {'BPM mean':>10} {'BPM std':>9} {'SNR mean [dB]':>14}")
print(f"  {'-'*5:<6} {'-'*9:>10} {'-'*8:>9} {'-'*13:>14}")

for ch_name in CHANNEL_NAMES:
    bpms = [r['channels'][ch_name]['peak_bpm'] for r in results]
    snrs = [r['channels'][ch_name]['snr_db']   for r in results]
    print(f"  {ch_name:<6} {np.mean(bpms):>10.2f} {np.std(bpms):>9.2f} {np.nanmean(snrs):>13.2f} dB")


# -------------------------------------------------------------------
# Plot 1: AC-tidssignal for alle 5 målinger, G-kanal
# -------------------------------------------------------------------
fig1, axes1 = plt.subplots(5, 1, figsize=(13, 12), sharex=False)
fig1.suptitle("Basismålinger – AC-puls-signal (G-kanal)", fontsize=13)

for i, (sig, fps, label) in enumerate(zip(all_signals, all_fps, all_labels)):
    t = np.arange(len(sig)) / fps
    ac = sig[:, 1] - np.mean(sig[:, 1])   # G-kanal, DC-fjernet
    axes1[i].plot(t, ac, color="green", linewidth=0.8)
    bpm = results[i]['channels']['G']['peak_bpm']
    snr = results[i]['channels']['G']['snr_db']
    axes1[i].set_title(f"{label}  |  BPM={bpm:.1f}  |  SNR={snr:.1f} dB", fontsize=9)
    axes1[i].set_ylabel("AC")
    axes1[i].grid(True, alpha=0.3)

axes1[-1].set_xlabel("Tid [s]")
plt.tight_layout()
plt.savefig("basismaalinger_ac_tidssignal.png", dpi=150)
print("\nFigur lagret: basismaalinger_ac_tidssignal.png")


# -------------------------------------------------------------------
# Plot 2: FFT-spektrum for alle 5 målinger, alle kanaler
# -------------------------------------------------------------------
fig2, axes2 = plt.subplots(5, 3, figsize=(15, 14), sharey=False)
fig2.suptitle("Basismålinger – FFT-effektspektrum per kanal", fontsize=13)

for row, (r, fps) in enumerate(zip(results, all_fps)):
    for col, (ch_name, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS)):
        ax = axes2[row, col]
        ch = r['channels'][ch_name]
        freqs, power = ch['freqs'], ch['power']

        # Vis bare det fysiologiske området
        mask = (freqs >= F_MIN - 0.2) & (freqs <= F_MAX + 0.2)
        ax.plot(freqs[mask] * 60, power[mask], color=color, linewidth=0.9)

        # Marker toppen
        if not np.isnan(ch['peak_bpm']):
            ax.axvline(ch['peak_bpm'], color="black", linestyle="--",
                       linewidth=0.8, alpha=0.7)
            ax.text(ch['peak_bpm'] + 1, ax.get_ylim()[1] * 0.5,
                    f"{ch['peak_bpm']:.0f} BPM", fontsize=7, va='center')

        short = r['label'].replace('Data_', 'M')
        if row == 0:
            ax.set_title(f"Kanal {ch_name}", fontsize=10)
        if col == 0:
            ax.set_ylabel(short, fontsize=8)
        ax.set_xlabel("BPM", fontsize=7)
        ax.grid(True, alpha=0.3)
        snr = ch['snr_db']
        ax.set_title(f"{('Kanal ' + ch_name) if row==0 else ''}"
                     f"{'  ' if row==0 else ''}"
                     f"SNR={snr:.1f} dB", fontsize=8)

plt.tight_layout()
plt.savefig("basismaalinger_fft.png", dpi=150)
print("Figur lagret: basismaalinger_fft.png")


# -------------------------------------------------------------------
# Plot 3: Overlay av alle 5 FFT-spektra (G-kanal) + BPM-statistikk
# -------------------------------------------------------------------
fig3, (ax_fft, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Basismålinger – Sammendrag pulsanalyse (G-kanal)", fontsize=13)

bpm_vals = []
for r in results:
    ch = r['channels']['G']
    freqs, power = ch['freqs'], ch['power']
    mask = (freqs >= F_MIN - 0.2) & (freqs <= F_MAX + 0.2)
    short = r['label'].replace('Data_', 'M')
    ax_fft.plot(freqs[mask] * 60, power[mask], linewidth=1.0, label=short)
    bpm_vals.append(ch['peak_bpm'])

ax_fft.set_xlabel("BPM")
ax_fft.set_ylabel("Effekt")
ax_fft.set_title("FFT-spektrum – alle målinger")
ax_fft.legend(fontsize=8)
ax_fft.grid(True, alpha=0.3)

# Søylediagram BPM med feilbars (gjennomsnitt ± std)
short_labels = [r['label'].replace('Data_', 'M') for r in results]
ax_bar.bar(short_labels, bpm_vals, color="green", alpha=0.75, edgecolor="black")
mean_bpm = np.mean(bpm_vals)
std_bpm  = np.std(bpm_vals)
ax_bar.axhline(mean_bpm, color="red", linestyle="--", linewidth=1.5,
               label=f"Gjennomsnitt: {mean_bpm:.1f} ± {std_bpm:.1f} BPM")
ax_bar.fill_between([-0.5, len(bpm_vals) - 0.5],
                    mean_bpm - std_bpm, mean_bpm + std_bpm,
                    color="red", alpha=0.15)
ax_bar.set_ylabel("Detektert puls [BPM]")
ax_bar.set_title("Detektert BPM per måling")
ax_bar.legend(fontsize=9)
ax_bar.grid(True, axis="y", alpha=0.3)
ax_bar.set_ylim(0, 120)

plt.tight_layout()
plt.savefig("basismaalinger_sammendrag.png", dpi=150)
print("Figur lagret: basismaalinger_sammendrag.png")

plt.show()
print("\nFerdig.")
