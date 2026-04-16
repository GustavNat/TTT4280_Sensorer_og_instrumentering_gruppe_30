from __future__ import annotations

import numpy as np

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


SPEED_OF_LIGHT = 299_792_458.0


def trim_signals(
    i_data: np.ndarray,
    q_data: np.ndarray,
    start: int,
    end: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim both channels to the selected sample range."""
    return i_data[start:end], q_data[start:end]


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """Remove DC offset by subtracting the mean value."""
    return signal - np.mean(signal)


def get_window(window_name: str, n: int, kaiser_beta: float = 8.0) -> np.ndarray:
    """Create the selected window."""
    if window_name == "rectangular":
        return np.ones(n)
    if window_name == "hann":
        return np.hanning(n)
    if window_name == "hamming":
        return np.hamming(n)
    if window_name == "kaiser":
        return np.kaiser(n, kaiser_beta)

    raise ValueError(f"Unsupported window: {window_name}")


def build_complex_signal(i_data: np.ndarray, q_data: np.ndarray) -> np.ndarray:
    """Build complex I/Q signal z[n] = I[n] + j Q[n]."""
    return i_data + 1j * q_data


def compute_fft(
    signal: np.ndarray,
    fs: float,
    window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute fftshifted FFT and matching frequency axis.
    """
    if len(signal) != len(window):
        raise ValueError("Signal and window must have the same length.")

    signal_windowed = signal * window
    n = len(signal_windowed)

    spectrum = np.fft.fftshift(np.fft.fft(signal_windowed))
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))

    return freq_axis, spectrum


def magnitude_db(spectrum: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert magnitude to dB."""
    return 20.0 * np.log10(np.abs(spectrum) + eps)


def build_search_mask(
    freq_axis: np.ndarray,
    dc_exclusion_hz: float,
    min_search_hz: float | None,
    max_search_hz: float | None,
) -> np.ndarray:
    """
    Build mask for which FFT bins are allowed in peak search.
    """
    abs_freq = np.abs(freq_axis)
    mask = abs_freq >= dc_exclusion_hz

    if min_search_hz is not None:
        mask &= abs_freq >= min_search_hz

    if max_search_hz is not None:
        mask &= abs_freq <= max_search_hz

    return mask


def parabolic_peak_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    peak_index: int,
) -> tuple[float, float]:
    """
    Improve peak estimate using simple parabolic interpolation.

    x: frequency axis
    y: linear magnitude values
    """
    if peak_index <= 0 or peak_index >= len(y) - 1:
        return float(x[peak_index]), float(y[peak_index])

    y_m1 = y[peak_index - 1]
    y_0 = y[peak_index]
    y_p1 = y[peak_index + 1]

    denominator = y_m1 - 2.0 * y_0 + y_p1
    if np.isclose(denominator, 0.0):
        return float(x[peak_index]), float(y[peak_index])

    delta = 0.5 * (y_m1 - y_p1) / denominator
    dx = x[1] - x[0]

    x_interp = x[peak_index] + delta * dx
    y_interp = y_0 - 0.25 * (y_m1 - y_p1) * delta

    return float(x_interp), float(y_interp)


def find_doppler_peak(
    freq_axis: np.ndarray,
    spectrum: np.ndarray,
    dc_exclusion_hz: float,
    min_search_hz: float | None,
    max_search_hz: float | None,
    peak_prominence_db: float,
) -> tuple[float, float, str]:
    """
    Find the Doppler peak in the complex spectrum.

    Returns:
    - peak frequency [Hz]
    - peak magnitude [linear]
    - method description
    """
    mask = build_search_mask(
        freq_axis=freq_axis,
        dc_exclusion_hz=dc_exclusion_hz,
        min_search_hz=min_search_hz,
        max_search_hz=max_search_hz,
    )

    if not np.any(mask):
        raise ValueError("No FFT bins left after applying search mask.")

    freq_valid = freq_axis[mask]
    mag_valid = np.abs(spectrum[mask])
    mag_valid_db = 20.0 * np.log10(mag_valid + 1e-12)

    method = "argmax"

    if SCIPY_AVAILABLE and len(mag_valid_db) >= 3:
        peaks, _ = find_peaks(mag_valid_db, prominence=peak_prominence_db)
        if len(peaks) > 0:
            peak_index = int(peaks[np.argmax(mag_valid[peaks])])
            method = "scipy.find_peaks"
        else:
            peak_index = int(np.argmax(mag_valid))
    else:
        peak_index = int(np.argmax(mag_valid))

    peak_freq, peak_mag = parabolic_peak_interpolation(
        freq_valid,
        mag_valid,
        peak_index,
    )

    method += " + parabolic interpolation"
    return peak_freq, peak_mag, method


def doppler_to_velocity(f_d: float, f0: float) -> float:
    """
    Monostatic CW Doppler radar:
        f_D = 2 * v_r / lambda
        v_r = f_D * lambda / 2
    """
    wavelength = SPEED_OF_LIGHT / f0
    return f_d * wavelength / 2.0