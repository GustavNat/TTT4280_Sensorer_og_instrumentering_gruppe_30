from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(t: np.ndarray, i_data: np.ndarray, q_data: np.ndarray) -> None:
    """Plot I and Q versus time."""
    plt.figure()
    plt.plot(t, i_data, label="I")
    plt.plot(t, q_data, label="Q")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [ADC counts]")
    plt.title("Radar I and Q time series")
    plt.grid(True)
    plt.legend()


def plot_iq_xy(i_data: np.ndarray, q_data: np.ndarray) -> None:
    """Plot Q versus I to inspect quadrature quality."""
    plt.figure()
    plt.plot(i_data, q_data)
    plt.xlabel("I [ADC counts]")
    plt.ylabel("Q [ADC counts]")
    plt.title("I-Q XY plot")
    plt.grid(True)
    plt.axis("equal")


def plot_spectrum(
    freq_axis: np.ndarray,
    spectrum_db: np.ndarray,
    peak_freq: float,
    min_search_hz: float | None,
    max_search_hz: float | None,
) -> None:
    """Plot complex FFT magnitude spectrum in dB."""
    plt.figure()
    plt.plot(freq_axis, spectrum_db, label="Spectrum")
    plt.axvline(peak_freq, linestyle="--", label=f"Peak = {peak_freq:.2f} Hz")

    if min_search_hz is not None:
        plt.axvline(min_search_hz, linestyle=":", alpha=0.7, label="Search limits")
        plt.axvline(-min_search_hz, linestyle=":", alpha=0.7)

    if max_search_hz is not None:
        plt.axvline(max_search_hz, linestyle=":", alpha=0.7)
        plt.axvline(-max_search_hz, linestyle=":", alpha=0.7)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.title("Complex FFT magnitude spectrum")
    plt.grid(True)
    plt.legend()
    plt.xlim(freq_axis[0], freq_axis[-1])