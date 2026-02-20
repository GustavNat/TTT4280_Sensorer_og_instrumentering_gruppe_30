from __future__ import annotations

from html import parser
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Course-provided module (downloaded from Blackboard)
from raspi_import import raspi_import

script_dir = os.path.dirname(os.path.abspath(__file__))


def rel_psd_db(psd: np.ndarray, ref: float = None) -> np.ndarray:
    """Relative PSD in dB (0 dB at the strongest non-DC bin)."""
    eps = 1e-40
    if psd.size <= 1:
        return 10 * np.log10(psd + eps)

    # Avoid DC bin dominating if any residual offset remains
    if ref is None:
        ref = np.max(psd[1:]) if np.any(psd[1:] > 0) else np.max(psd)
    ref = max(ref, eps)
    return 10 * np.log10((psd + eps) / ref)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="1kHz.bin", help="Binary file from RPi sampler")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels stored in file")
    parser.add_argument("--ch", type=int, default=0, help="Channel 1 to plot (0-based)")
    parser.add_argument("--pad", type=int, default=20000, help="Zero padding length")
    parser.add_argument("--max-f", type=float, default=None, help="Max frequency to show [Hz]")
    parser.add_argument("--no-detrend", action="store_true", help="Do not remove mean (DC)")
    parser.add_argument("--fs", type=float, default=31250.0, help="Sample rate in Hz (default: 31250)")
    parser.add_argument("--f0", type=float, default=1000.0, help="Ideal sine wave frequency in Hz (default: 1000)")
    args = parser.parse_args()

    # raspi_import already converts sample_period from µs to seconds
    file_path = os.path.join(script_dir, 'Measurements', 'ADC', args.file)
    _, data = raspi_import(file_path, channels=args.channels)

    # Use specified sample rate
    fs = args.fs
    Ts = 1.0 / fs
    print(f"Using sample rate: fs = {fs} Hz, Ts = {Ts:.6e} s")

    if not (0 <= args.ch < data.shape[1]):
        raise ValueError(f"--ch out of range. data has {data.shape[1]} channels.")

    x = data[:, args.ch].astype(np.float64)

    # Remove DC offset (as recommended in lab analysis)
    if not args.no_detrend:
        x = x - np.mean(x)

    N = x.size

    # Generate ideal sine wave with same length and amplitude
    t = np.arange(N) * Ts
    x_ideal = np.sin(2 * np.pi * args.f0 * t)
    # Scale to match measured signal amplitude
    x_ideal = x_ideal * np.std(x) / np.std(x_ideal)

    pad_lengths = [0, int(args.pad)]

    windows = [
        ("Rectangular", np.ones(N)),
        ("Hamming", np.hamming(N)),
        ("Hanning", np.hanning(N)),
        ("Bartlett", np.bartlett(N)),
    ]

    fig, axes = plt.subplots(
        nrows=len(windows),
        ncols=len(pad_lengths),
        figsize=(12, 10),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    for i, (wname, w) in enumerate(windows):
        xw = x * w
        xw_ideal = x_ideal * w

        # Window power for normalization
        window_power = np.sum(w ** 2)

        for j, p in enumerate(pad_lengths):
            xwp = np.pad(xw, (0, p), mode="constant") if p > 0 else xw
            xwp_ideal = np.pad(xw_ideal, (0, p), mode="constant") if p > 0 else xw_ideal
            M = xwp.size

            # Single-sided spectrum (real input)
            X = np.fft.rfft(xwp)
            X_ideal = np.fft.rfft(xwp_ideal)
            f = np.fft.rfftfreq(M, d=Ts)

            # Compute PSD: |X|^2 / (fs * window_power)
            # This gives units of V^2/Hz
            psd = (np.abs(X) ** 2) / (fs * window_power)
            psd_ideal = (np.abs(X_ideal) ** 2) / (fs * window_power)

            # Double the power for single-sided spectrum (except DC and Nyquist)
            psd[1:-1] *= 2
            psd_ideal[1:-1] *= 2

            # Use the same reference for both (measured signal's max)
            ref = np.max(psd[1:]) if np.any(psd[1:] > 0) else np.max(psd)
            y_db = rel_psd_db(psd, ref)
            y_db_ideal = rel_psd_db(psd_ideal, ref)

            ax = axes[i, j]
            ax.plot(f, y_db, label='Measured')
            ax.plot(f, y_db_ideal, label=f'Ideal {args.f0:.0f} Hz', linestyle='--', alpha=0.7)
            ax.set_ylim(-100, 0)
            ax.grid(True, which='both')
            ax.legend(loc='upper right', fontsize='small')

            ax.set_title(f"{wname} window, pad={p}")
            ax.set_ylabel("Rel. PSD [dB/Hz]")

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency [Hz]")

    fig.suptitle(
        f"PSD of {args.file} vs ideal {args.f0:.0f} Hz sine | channel {args.ch} | N={N} | fs={fs:.2f} Hz",
        y=1.02,
    )

    plt.show()


if __name__ == "__main__":
    main()
