import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

os.chdir(os.path.dirname(__file__)) 

Measurement_folder = "Measurements/"
VIDEO_FILES = [
    "56_2.mp4",
    "56.mp4",
    "57_2.mp4",
    "57.mp4",
    # "58.mp4",
    # "cold_finger_59.mp4",
    "fore_head_58.mp4",
    # "Higher_102.mp4",
    # "nothing.mp4"
    "fore_head_2.mp4"
]

VIDEO_FILES = [os.path.join(Measurement_folder, file) for file in VIDEO_FILES]


BPM_MAX = 80
BPM_MIN = 40
ZERO_PADDING = 8  


CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def file_to_RGB_values(file_path: str, crop=False) -> tuple[np.array, np.array, np.array, float]:
    """R, G, B, fps. If crop=True, opens a window to select ROI once from the first frame."""

    suffix = "_cropped" if crop else ""
    cache_path = os.path.join(CACHE_DIR, os.path.basename(file_path) + suffix + ".npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["R"], data["G"], data["B"], float(data["fps"])

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first_frame = cap.read()

    if crop:
        x, y, w, h = cv2.selectROI("Select ROI – press ENTER to confirm", first_frame, fromCenter=False)
        cv2.destroyAllWindows()
    else:
        x, y, w, h = 0, 0, first_frame.shape[1], first_frame.shape[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    means = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[y:y+h, x:x+w]
        mean_bgr = np.mean(frame, axis=(0, 1))
        means.append(mean_bgr)

    means = np.array(means)
    cap.release()

    B = means[:, 0]
    G = means[:, 1]
    R = means[:, 2]
    R = R - np.mean(R)
    G = G - np.mean(G)
    B = B - np.mean(B)

    np.savez(cache_path, R=R, G=G, B=B, fps=fps)
    return R, G, B, fps



def plot_time_domain(R, G, B, fps):
    
    N = len(G)
    t = np.arange(N) / fps
    plt.plot(t, R, label="Red")
    plt.plot(t, G, label="Green")
    plt.plot(t, B, label="Blue")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean intensity")
    plt.legend()
    plt.show()





def FFT(R, G, B, fps):
    N_padded = len(G) * ZERO_PADDING
    freq = np.fft.rfftfreq(N_padded, d=1/fps) * 60  # convert Hz to BPM

    amp_R = np.abs(np.fft.rfft(R, n=N_padded))**2
    amp_G = np.abs(np.fft.rfft(G, n=N_padded))**2
    amp_B = np.abs(np.fft.rfft(B, n=N_padded))**2

    mask = (freq >= BPM_MIN) & (freq <= BPM_MAX)

    return amp_R[mask], amp_G[mask], amp_B[mask], freq[mask]


def max_frequency(amp_R, amp_G, amp_B, freq):
    # arrays are already filtered to BPM range by FFT()
    peak_freq_R = freq[np.argmax(amp_R)]
    peak_freq_G = freq[np.argmax(amp_G)]
    peak_freq_B = freq[np.argmax(amp_B)]

    pulse_bpm_R = peak_freq_R
    pulse_bpm_G = peak_freq_G
    pulse_bpm_B = peak_freq_B

    return pulse_bpm_R, pulse_bpm_G, pulse_bpm_B


SNR_SIGNAL_WINDOW_BPM = 2*ZERO_PADDING  # ± BPM around peak counted as signal bins

def calculate_snr(amp, freq, peak_bpm):
    """SNR = mean(signal bins) / mean(noise bins), in dB."""
    signal_mask = np.abs(freq - peak_bpm) <= SNR_SIGNAL_WINDOW_BPM
    noise_mask  = ~signal_mask
    mean_signal = np.mean(amp[signal_mask])
    mean_noise  = np.mean(amp[noise_mask])
    return 10 * np.log10(mean_signal / mean_noise)



def plot_frequency_spectrum(amp_R, amp_G, amp_B, freq):
    plt.plot(freq, amp_R)
    plt.plot(freq, amp_G)
    plt.plot(freq, amp_B)
    plt.xlabel("Frequency [BPM]")
    plt.ylabel("Amplitude")
    plt.title("FFT spectrum")
    plt.show()


channel_names  = ["R", "G", "B"]
channel_colors = ["red", "green", "blue"]

# --- collect data ---
labels = []
bpm_per_channel = {"R": [], "G": [], "B": []}
all_signals = []  # store (R, G, B, fps) for time series plot

fig,  axes  = plt.subplots(len(VIDEO_FILES), 3, figsize=(15, 3 * len(VIDEO_FILES)))
fig2, axes2 = plt.subplots(len(VIDEO_FILES), 1, figsize=(12, 3 * len(VIDEO_FILES)), sharex=False)
fig.suptitle("FFT spectrum per measurement and channel")
fig2.suptitle("Time series – all colour channels")

for row, filepath in enumerate(VIDEO_FILES):
    label = os.path.splitext(os.path.basename(filepath))[0]
    labels.append(label)
    R, G, B, fps = file_to_RGB_values(filepath, crop="fore_head" in filepath)
    all_signals.append((R, G, B, fps))

    # --- time series subplot ---
    t = np.arange(len(G)) / fps
    ax_t = axes2[row]
    ax_t.plot(t, R, color="red",   linewidth=0.8, label="R")
    ax_t.plot(t, G, color="green", linewidth=0.8, label="G")
    ax_t.plot(t, B, color="blue",  linewidth=0.8, label="B")
    ax_t.set_ylabel(label, fontsize=8)
    ax_t.legend(fontsize=7, loc="upper right")
    ax_t.grid(True, alpha=0.3)
    if row == len(VIDEO_FILES) - 1:
        ax_t.set_xlabel("Time [s]")

    amp_R, amp_G, amp_B, freq = FFT(R, G, B, fps)
    bpm = max_frequency(amp_R, amp_G, amp_B, freq)
    amps = [amp_R, amp_G, amp_B]
    snrs = [calculate_snr(amp, freq, b) for amp, b in zip(amps, bpm)]
    print(f"{label}:  R={bpm[0]:.1f} BPM (SNR={snrs[0]:.1f} dB)  "
          f"G={bpm[1]:.1f} BPM (SNR={snrs[1]:.1f} dB)  "
          f"B={bpm[2]:.1f} BPM (SNR={snrs[2]:.1f} dB)")
    for ch, val in zip(channel_names, bpm):
        bpm_per_channel[ch].append(val)

    for col, (amp, name, color, snr) in enumerate(zip(amps, channel_names, channel_colors, snrs)):
        ax = axes[row, col]
        amp_db = 10 * np.log10(amp / np.max(amp))
        ax.plot(freq, amp_db, color=color, linewidth=0.9)
        ax.axvline(bpm[col], color="black", linestyle="--", linewidth=1.2,
                   label=f"{bpm[col]:.1f} BPM\nSNR={snr:.1f} dB")
        ax.plot(bpm[col], 0, "kv", markersize=7)
        ax.legend(fontsize=7)
        if row == 0:
            ax.set_title(f"Channel {name}")
        if col == 0:
            ax.set_ylabel(label, fontsize=8)
        ax.set_xlabel("BPM", fontsize=7)
        ax.set_ylabel("Amplitude [dB]", fontsize=7)
        ax.grid(True, alpha=0.3)

fig2.tight_layout()
plt.tight_layout()
plt.show()

# --- bar charts: one per channel ---
fig3, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig3.suptitle("Detected BPM per measurement and channel")

for ax, name, color in zip(axes2, channel_names, channel_colors):
    vals = bpm_per_channel[name]
    mu  = np.mean(vals)
    sig = np.std(vals)
    ax.bar(labels, vals, color=color, alpha=0.75, edgecolor="black")
    ax.axhline(mu, color="black", linestyle="--", linewidth=1.5,
               label=f"$\\mu$ = {mu:.1f} BPM\n$\\sigma$ = {sig:.1f} BPM")
    ax.fill_between([-0.5, len(vals) - 0.5], mu - sig, mu + sig,
                    color="black", alpha=0.12)
    ax.set_title(f"Channel {name}")
    ax.set_ylabel("BPM")
    ax.set_ylim(0, 120)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()

