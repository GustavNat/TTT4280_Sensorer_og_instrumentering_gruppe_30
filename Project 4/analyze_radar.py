#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting import plot_iq_xy, plot_spectrum, plot_time_series
from signal_processing import (
    SCIPY_AVAILABLE,
    build_complex_signal,
    compute_fft,
    doppler_to_velocity,
    find_doppler_peak,
    get_window,
    magnitude_db,
    remove_dc,
    trim_signals,
)


RADAR_CENTER_FREQUENCY_HZ = 24.13e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Doppler radar I/Q data.")
    parser.add_argument("--file", type=Path, required=True, help="Path to .bin file")
    parser.add_argument(
        "--channels",
        type=int,
        default=5,
        help="Total number of channels stored in the file",
    )
    parser.add_argument(
        "--i-channel",
        type=int,
        default=0,
        help="Column index for I channel",
    )
    parser.add_argument(
        "--q-channel",
        type=int,
        default=1,
        help="Column index for Q channel",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start sample index for trimming",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End sample index for trimming",
    )
    parser.add_argument(
        "--f0",
        type=float,
        default=RADAR_CENTER_FREQUENCY_HZ,
        help="Radar center frequency in Hz",
    )
    parser.add_argument(
        "--dc-exclusion-hz",
        type=float,
        default=1.0,
        help="Ignore frequencies within +/- this value around DC when finding peak",
    )
    parser.add_argument(
        "--min-search-hz",
        type=float,
        default=None,
        help="Optional lower bound on |frequency| when searching for Doppler peak",
    )
    parser.add_argument(
        "--max-search-hz",
        type=float,
        default=None,
        help="Optional upper bound on |frequency| when searching for Doppler peak",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="rectangular",
        choices=["rectangular", "hann", "hamming", "kaiser"],
        help="Window to apply before FFT",
    )
    parser.add_argument(
        "--kaiser-beta",
        type=float,
        default=8.0,
        help="Beta parameter for Kaiser window",
    )
    parser.add_argument(
        "--peak-prominence-db",
        type=float,
        default=6.0,
        help="Minimum prominence in dB for peak detection when scipy is available",
    )
    return parser.parse_args()


def load_radar_file(filepath: Path, channels: int) -> tuple[float, np.ndarray]:
    """
    Read binary file in the same format as raspi_import.py.

    Format:
    - first value: sample_period as float64
    - remaining values: uint16 samples
    - data reshaped to (samples, channels)
    - sample_period stored in microseconds, converted to seconds here
    """
    with open(filepath, "rb") as fid:
        sample_period = np.fromfile(fid, count=1, dtype=np.float64)[0]
        data = np.fromfile(fid, dtype=np.uint16).astype(np.float64)

    if data.size % channels != 0:
        raise ValueError(
            f"Data length {data.size} is not divisible by channels={channels}."
        )

    data = data.reshape((-1, channels))
    sample_period *= 1e-6

    return float(sample_period), data


def extract_iq_channels(
    data: np.ndarray,
    i_channel: int,
    q_channel: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_channels = data.shape[1]

    if not (0 <= i_channel < n_channels):
        raise IndexError(f"I channel {i_channel} out of range for {n_channels} channels")
    if not (0 <= q_channel < n_channels):
        raise IndexError(f"Q channel {q_channel} out of range for {n_channels} channels")

    i_data = data[:, i_channel].copy()
    q_data = data[:, q_channel].copy()

    return i_data, q_data


def main() -> None:
    args = parse_args()

    sample_period, data = load_radar_file(args.file, args.channels)
    fs = 1.0 / sample_period

    n_samples, n_channels = data.shape
    print("\nLoaded data")
    print("-----------")
    print(f"File:            {args.file}")
    print(f"Sample period:   {sample_period:.6e} s")
    print(f"Sampling freq:   {fs:.3f} Hz")
    print(f"Data shape:      {data.shape}")
    print(f"Samples:         {n_samples}")
    print(f"Channels:        {n_channels}")

    i_data, q_data = extract_iq_channels(data, args.i_channel, args.q_channel)
    i_data, q_data = trim_signals(i_data, q_data, args.start, args.end)

    if len(i_data) == 0:
        raise ValueError("No data left after trimming.")

    t = np.arange(len(i_data)) * sample_period

    i_data = remove_dc(i_data)
    q_data = remove_dc(q_data)

    z = build_complex_signal(i_data, q_data)

    window = get_window(args.window, len(z), args.kaiser_beta)
    freq_axis, spectrum = compute_fft(z, fs, window)
    spectrum_db = magnitude_db(spectrum)

    peak_freq, peak_mag, peak_method = find_doppler_peak(
        freq_axis=freq_axis,
        spectrum=spectrum,
        dc_exclusion_hz=args.dc_exclusion_hz,
        min_search_hz=args.min_search_hz,
        max_search_hz=args.max_search_hz,
        peak_prominence_db=args.peak_prominence_db,
    )
    radial_velocity = doppler_to_velocity(peak_freq, args.f0)

    observation_time = len(z) * sample_period
    freq_resolution = 1.0 / observation_time

    print("\nAnalysis results")
    print("----------------")
    print(f"Trimmed samples:         {len(z)}")
    print(f"Observation time:        {observation_time:.6f} s")
    print(f"Theoretical df = 1/T:    {freq_resolution:.6f} Hz")
    print(f"Window:                  {args.window}")
    if args.window == "kaiser":
        print(f"Kaiser beta:             {args.kaiser_beta:.3f}")
    print(f"Peak Doppler frequency:  {peak_freq:.6f} Hz")
    print(f"Peak magnitude:          {peak_mag:.6e}")
    print(f"Estimated radial speed:  {radial_velocity:.6f} m/s")
    print(f"Peak method:             {peak_method}")
    print(f"SciPy available:         {'yes' if SCIPY_AVAILABLE else 'no'}")

    plot_time_series(t, i_data, q_data)
    plot_iq_xy(i_data, q_data)
    plot_spectrum(
        freq_axis=freq_axis,
        spectrum_db=spectrum_db,
        peak_freq=peak_freq,
        min_search_hz=args.min_search_hz,
        max_search_hz=args.max_search_hz,
    )

    plt.show()


if __name__ == "__main__":
    main()