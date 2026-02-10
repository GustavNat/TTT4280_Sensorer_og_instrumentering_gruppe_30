from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import bartlett

from raspi_import import raspi_import
# import plotWindows

timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")

# Prosjektstier (robust)
base_dir = Path(__file__).resolve().parent          # .../TilRapport/Kode
project_dir = base_dir.parent                       # .../TilRapport
bin_dir = project_dir / "bin"
plots_dir = project_dir / "Plots"
plots_zoom_dir = plots_dir / "Zoomed"

plots_dir.mkdir(exist_ok=True)
plots_zoom_dir.mkdir(parents=True, exist_ok=True)

# bin_file = bin_dir / "A3dot2V100Hz.bin"
bin_file = base_dir / "Measurements" / "ADC" / "100Hz.bin"

bin_stem = bin_file.stem

Ts, data = raspi_import(bin_file, channels=5)

C = 1
V_ref = 3.3
V_conv = C / 4095 * V_ref
Nsamples = data.shape[0]
Fs = 1 / Ts
t = np.arange(Nsamples) * Ts

dataAnalogCh1 = data.T[0] * V_conv
dataAnalogOffsetCh1 = dataAnalogCh1 - 1.5


def plotFigur(Naxs, xlimMin, xlimMax, ylimMin, ylimMax, filename, logx):
    fig, axs = plt.subplots(2, Naxs, figsize=(10, 4))
    axs = axs.ravel()
    x = dataAnalogOffsetCh1

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
            raise ValueError("windowStyle må være 'rectangular', 'hanning', 'hamming' eller 'bartlett'")

        X = np.fft.rfft(xw, Nsamples + Nzeros)
        F = np.fft.rfftfreq(Nsamples + Nzeros, Ts)

        XmagDb = 20 * np.log10(np.abs(X) + 1e-12)  # unngå log(0)
        axs[axNum].plot(F, XmagDb - np.max(XmagDb))
        axs[axNum].set_title(f"{windowStyle}, {Nzeros} zeros")
        axs[axNum].set_xlabel("Frequency [Hz]")
        if axNum == 0 or axNum == 4:
            axs[axNum].set_ylabel("Relative magnitude [dB]")
        axs[axNum].set_xlim(xlimMin, xlimMax)
        axs[axNum].set_ylim(ylimMin, ylimMax)
        if logx == 1:
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


plotFigur(4, 50, 15000, -85, 5, plots_dir / f"FftPlot_{bin_stem}_{timestamp}.png", 1)
plotFigur(4, 990, 1010, -80, 10, plots_zoom_dir / f"FftPlot_zoomedIn_{bin_stem}_{timestamp}.png", 0)