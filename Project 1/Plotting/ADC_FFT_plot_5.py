from __future__ import annotations

from html import parser
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Course-provided module (downloaded from Blackboard)
from raspi_import import raspi_import


def sine_func(t, A, f, phi, offset):
    """Sine function for curve fitting."""
    return A * np.sin(2 * np.pi * f * t + phi) + offset

script_dir = os.path.dirname(os.path.abspath(__file__))


def rel_power_db(power: np.ndarray, ref: float = None) -> np.ndarray:
    """Relative power in dB (0 dB at the strongest non-DC bin)."""
    eps = 1e-40
    if power.size <= 1:
        return 10 * np.log10(power + eps)

    # Avoid DC bin dominating if any residual offset remains
    if ref is None:
        ref = np.max(power[1:]) if np.any(power[1:] > 0) else np.max(power)
    ref = max(ref, eps)
    return 10 * np.log10((power + eps) / ref)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="4kHz.bin", help="Binary file from RPi sampler")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels stored in file")
    parser.add_argument("--ch", type=int, default=0, help="Channel 1 to plot (0-based)")
    parser.add_argument("--pad", type=int, default=2000, help="Zero padding length")
    parser.add_argument("--max-f", type=float, default=None, help="Max frequency to show [Hz]")
    parser.add_argument("--no-detrend", action="store_true", help="Do not remove mean (DC)")
    parser.add_argument("--fs", type=float, default=31250.0, help="Sample rate in Hz (default: 31250)")
    parser.add_argument("--f0", type=float, default=4000.0, help="Ideal sine wave frequency in Hz (default: 1000)")
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

    # Windows: Rectangular and Hanning with padding
    p = int(args.pad)
    windows = [
        ("Rectangular", np.ones(N)),
        ("Hanning", np.hanning(N)),
    ]

    # Curve fit once (before windowing)
    t = np.arange(N) * Ts
    try:
        p0 = [np.std(x) * np.sqrt(2), args.f0, 0, 0]
        popt, _ = curve_fit(sine_func, t, x, p0=p0, maxfev=10000)
        A_fit, f_fit, phi_fit, offset_fit = popt
        x_fitted = sine_func(t, A_fit, f_fit, phi_fit, offset_fit)
        noise = x - x_fitted
        fit_success = True
        print(f"Curve fit: f = {f_fit:.1f} Hz, A = {A_fit:.4f} V")
    except Exception as e:
        print(f"Curve fit failed: {e}")
        fit_success = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, (wname, w) in zip(axes, windows):
        xw = x * w

        # Window power for normalization
        window_power = np.sum(w ** 2)

        xwp = np.pad(xw, (0, p), mode="constant")
        M = xwp.size

        # Single-sided spectrum (real input)
        X = np.fft.rfft(xwp)
        f = np.fft.rfftfreq(M, d=Ts)

        # Compute Power Spectrum: |X|^2 / window_power
        power = (np.abs(X) ** 2) / window_power

        # Double the power for single-sided spectrum (except DC and Nyquist)
        power[1:-1] *= 2

        # Calculate SNR
        if fit_success:
            noise_w = noise * w
            noise_wp = np.pad(noise_w, (0, p), mode="constant")
            Noise_fft = np.fft.rfft(noise_wp)
            power_noise = (np.abs(Noise_fft) ** 2) / window_power
            power_noise[1:-1] *= 2

            P_total = np.sum(power) / M
            P_noise = np.sum(power_noise) / M
            P_signal = P_total - P_noise

            if P_noise > 0 and P_signal > 0:
                snr_db = 10 * np.log10(P_signal / P_noise)
            else:
                snr_db = float('inf')

            print(f"{wname} window, pad={p}: SNR = {snr_db:.2f} dB")

        # Use the measured signal's max as reference
        ref = np.max(power[1:]) if np.any(power[1:] > 0) else np.max(power)
        y_db = rel_power_db(power, ref)

        ax.plot(f, y_db, label='Measured')
        ax.set_ylim(-100, 5)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Rel. Power [dB]")
        ax.set_title(f"{wname} window, pad={p}")
        ax.grid(True, which='both')
        ax.legend(loc='upper right')

    fig.suptitle(
        f"Power Spectrum of {args.file} | channel {args.ch} | N={N} | fs={fs:.2f} Hz",
    )

    plt.show()


if __name__ == "__main__":
    main()
