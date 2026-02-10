
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import bartlett

from raspi_import import raspi_import


timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")

# --- Robust project paths ---
base_dir = Path(__file__).resolve().parent          # .../Project 1/Plotting
project_dir = base_dir.parent                       # .../Project 1

# Where your .bin files usually are (add more candidates if needed)
data_dirs = [
    project_dir / "Measurements" / "ADC",
    project_dir / "bin",
]

# --- Choose which file to plot ---
target_name = "test6.bin"

bin_file = None
for d in data_dirs:
    p = d / target_name
    if p.exists():
        bin_file = p
        break

if bin_file is None:
    raise FileNotFoundError(
        f"Could not find '{target_name}' in:\n" + "\n".join(str(d) for d in data_dirs)
    )

bin_stem = bin_file.stem

# --- Load data (try common channel counts) ---
def load_with_autochannels(path: Path, channel_try=(5, 3, 1)):
    last_err = None
    for ch in channel_try:
        try:
            Ts, data = raspi_import(path, channels=ch)
            return Ts, data
        except Exception as e:
            last_err = e
    raise RuntimeError(f"raspi_import failed for {path}. Last error: {last_err}")

Ts, data = load_with_autochannels(bin_file)

data = np.asarray(data)
if data.ndim == 1:
    data = data[:, None]  # ensure shape (N, channels)

# --- ADC -> Volts ---
C = 1
V_ref = 3.3
V_conv = C / 4095 * V_ref

Nsamples = data.shape[0]
Fs = 1 / Ts

# pick channel 1 (index 0). Change to 1/2/... if you want another ADC channel
ch_idx = 0
dataAnalogCh = data[:, ch_idx] * V_conv
dataAnalogOffset = dataAnalogCh - 1.5


def plot_windows(N: int, filename: Path):
    n = np.arange(N)
    w_rect = np.ones(N)
    w_hann = np.hanning(N)
    w_hamm = np.hamming(N)
    w_bart = bartlett(N)

    plt.figure(figsize=(10, 4))
    plt.plot(n, w_rect, label="rectangular")
    plt.plot(n, w_hann, label="hanning")
    plt.plot(n, w_hamm, label="hamming")
    plt.plot(n, w_bart, label="bartlett")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plotFigur(Naxs, xlimMin, xlimMax, ylimMin, ylimMax, filename, logx):
    fig, axs = plt.subplots(2, Naxs, figsize=(10, 4))
    axs = axs.ravel()
    x = dataAnalogOffset

    def fft(Nzeros, axNum, windowStyle):
        if windowStyle == "rectangular":
            xw = x
        elif windowStyle == "hanning":
            xw = x * np.hanning(Nsamples)
        elif windowStyle == "hamming":
            xw = x * np.hamming(Nsamples)
        elif windowStyle == "bartlett":
            xw = x * bartlett(Nsamples)
        else:
            raise ValueError("windowStyle must be: rectangular, hanning, hamming, bartlett")

        X = np.fft.rfft(xw, Nsamples + Nzeros)
        F = np.fft.rfftfreq(Nsamples + Nzeros, Ts)

        XmagDb = 20 * np.log10(np.abs(X) + 1e-12)
        axs[axNum].plot(F, XmagDb - np.max(XmagDb))
        axs[axNum].set_title(f"{windowStyle}, {Nzeros} zeros")
        axs[axNum].set_xlabel("Frequency [Hz]")
        if axNum in (0, 4):
            axs[axNum].set_ylabel("Relative magnitude [dB]")
        axs[axNum].set_xlim(xlimMin, xlimMax)
        axs[axNum].set_ylim(ylimMin, ylimMax)
        if logx:
            axs[axNum].set_xscale("log")
        axs[axNum].grid(True)

    fft(0, 0, "rectangular")
    fft(20000, 1, "rectangular")
    fft(0, 2, "hanning")
    fft(20000, 3, "hanning")
    fft(0, 4, "hamming")
    fft(20000, 5, "hamming")
    fft(0, 6, "bartlett")
    fft(20000, 7, "bartlett")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


# --- Output folders ---
plots_dir = project_dir / "Plots"
plots_zoom_dir = plots_dir / "Zoomed"
plots_dir.mkdir(parents=True, exist_ok=True)
plots_zoom_dir.mkdir(parents=True, exist_ok=True)

# --- Plots ---
plotFigur(
    Naxs=4,
    xlimMin=50,
    xlimMax=15000,
    ylimMin=-85,
    ylimMax=5,
    filename=plots_dir / f"FftPlot_{bin_stem}_{timestamp}.png",
    logx=True,
)

plotFigur(
    Naxs=4,
    xlimMin=990,
    xlimMax=1010,
    ylimMin=-80,
    ylimMax=10,
    filename=plots_zoom_dir / f"FftPlot_zoomedIn_{bin_stem}_{timestamp}.png",
    logx=False,
)

plot_windows(Nsamples, plots_dir / f"vindusfunksjoner_{bin_stem}_{timestamp}.png")
