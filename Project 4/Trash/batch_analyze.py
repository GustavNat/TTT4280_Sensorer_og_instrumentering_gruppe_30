#!/usr/bin/env python3
"""
Section 9.3, points 2-5: Doppler radar batch analysis.

Point 2 – Measured vs. theoretical velocity (scatter plot + repeated runs)
Point 3 – Raw I/Q time series, IQ amplitude scaling, Doppler spectrum in dB, SNR
Point 4 – Doppler resolution: theoretical df=1/T vs. measured 3-dB peak width
Point 5 – Velocity accuracy: mean and standard deviation per speed group
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from analyze_radar import extract_iq_channels, load_radar_file
from signal_processing import (
    build_complex_signal,
    compute_fft,
    doppler_to_velocity,
    find_doppler_peak,
    get_window,
    magnitude_db,
    remove_dc,
    trim_signals,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR           = Path(__file__).parent / "Data"
RADAR_F0_HZ        = 24.13e9
CHANNELS           = 5
I_CH               = 0
Q_CH               = 1
WINDOW_NAME        = "hann"
DC_EXCLUSION_HZ    = 1.0
MIN_SEARCH_HZ      = 3.5
MAX_SEARCH_HZ      = 2800.0
PEAK_PROMINENCE_DB = 6.0

# SNR: how many frequency bins either side of the peak count as "signal"
SNR_SIGNAL_BINS = 3

# Motion detection: parameters for automatically finding the constant-speed window
MOTION_WIN_S       = 0.5    # short-time window length for energy scan [s]
MOTION_THRESH_DB   = 10.0   # window must be >= noise_floor + this value to count as motion
MOTION_MIN_FD_HZ   = 20.0   # minimum Doppler frequency to count as valid motion (not stationary)
MOTION_SKIP_S      = 0.5    # always skip this many seconds at the very start of the file
MOTION_SKIP_FIRST  = 1      # within detected motion block, skip this many chunks (acceleration)

# Groups: (display label, nominal velocity m/s, filename prefix)
GROUPS = [
    ("0.36 m/s (toward)",  +0.36, "036ms"),
    ("1.18 m/s (toward)",  +1.18, "118ms"),
    ("0.87 m/s (away)",    -0.87, "neg087ms"),
]
# ──────────────────────────────────────────────────────────────────────────────


# ── Helper: automatic motion-window detection ────────────────────────────────
def detect_motion_window(
    signal: np.ndarray,
    fs: float,
    min_fd: float = MIN_SEARCH_HZ,
    max_fd: float = MAX_SEARCH_HZ,
    win_s: float = MOTION_WIN_S,
    thresh_db: float = MOTION_THRESH_DB,
    min_fd_motion: float = MOTION_MIN_FD_HZ,
    skip_s: float = MOTION_SKIP_S,
    skip_first: int = MOTION_SKIP_FIRST,
) -> tuple[int, int]:
    """
    Find the first constant-speed window in the recording.

    A short-time window counts as "motion" only when:
      (a) its peak Doppler magnitude exceeds the noise floor by thresh_db, AND
      (b) the dominant Doppler frequency >= min_fd_motion (i.e. not stationary).

    The noise floor is estimated from the minimum chunk magnitude rather than
    the median, so the threshold is not inflated when motion occupies most of
    the recording.

    Returns (start_sample, end_sample) with the first skip_first chunks of
    the detected block discarded to skip the acceleration ramp.
    """
    skip_samples = int(skip_s * fs)
    chunk = int(win_s * fs)
    if chunk == 0:
        return 0, len(signal)

    # work from skip_samples onwards
    search = signal[skip_samples:]
    n_chunks = len(search) // chunk
    if n_chunks < 1:
        return 0, len(signal)

    peak_mags  = []
    dom_freqs  = []
    for k in range(n_chunks):
        seg  = search[k * chunk:(k + 1) * chunk]
        spec = np.fft.rfft(seg * np.hanning(len(seg)))
        fa   = np.fft.rfftfreq(len(seg), d=1.0 / fs)
        mask = (fa >= min_fd) & (fa <= max_fd)
        if np.any(mask):
            peak_mags.append(float(20.0 * np.log10(np.abs(spec[mask]).max() + 1e-12)))
            dom_freqs.append(float(fa[mask][np.argmax(np.abs(spec[mask]))]))
        else:
            peak_mags.append(-np.inf)
            dom_freqs.append(0.0)

    peak_mags = np.array(peak_mags)
    dom_freqs = np.array(dom_freqs)

    # noise floor = minimum chunk magnitude (avoids inflation from long motion)
    noise_floor = float(np.min(peak_mags[np.isfinite(peak_mags)]) if np.any(np.isfinite(peak_mags)) else -np.inf)
    threshold   = noise_floor + thresh_db

    motion = (peak_mags >= threshold) & (dom_freqs >= min_fd_motion)

    # find the FIRST contiguous True run
    first_start, first_len = -1, 0
    cur_start,   cur_len   = 0, 0
    for k, m in enumerate(motion):
        if m:
            if cur_len == 0:
                cur_start = k
            cur_len += 1
            if first_start == -1:    # take first run only
                first_start = cur_start
        else:
            if cur_len > 0:
                if first_start != -1:
                    first_len = cur_len
                    break            # stop after first run ends
            cur_len = 0
    else:
        # loop ended while still in a run
        if cur_len > 0 and first_start != -1:
            first_len = cur_len

    if first_start == -1 or first_len == 0:
        return 0, len(signal)        # no motion detected – use everything

    # skip the first few chunks (acceleration ramp)
    actual_start = first_start + min(skip_first, first_len - 1)
    abs_start    = skip_samples + int(actual_start * chunk)
    abs_end      = skip_samples + int((first_start + first_len) * chunk)
    return abs_start, abs_end


# ── Helper: 3-dB bandwidth of the Doppler peak ───────────────────────────────
def measure_3db_bandwidth(freq_axis: np.ndarray, spec_db: np.ndarray,
                          peak_freq: float) -> float:
    """Return the 3-dB bandwidth (Hz) around peak_freq."""
    peak_idx = int(np.argmin(np.abs(freq_axis - peak_freq)))
    peak_level = spec_db[peak_idx]
    threshold  = peak_level - 3.0

    # walk left
    left = peak_idx
    while left > 0 and spec_db[left] >= threshold:
        left -= 1

    # walk right
    right = peak_idx
    while right < len(spec_db) - 1 and spec_db[right] >= threshold:
        right += 1

    # linear interpolation at both crossings
    def interp_crossing(idx_in, idx_out):
        f1, f2 = freq_axis[idx_in], freq_axis[idx_out]
        s1, s2 = spec_db[idx_in],   spec_db[idx_out]
        if s1 == s2:
            return (f1 + f2) / 2
        return f1 + (threshold - s1) * (f2 - f1) / (s2 - s1)

    f_left  = interp_crossing(left + 1, left)
    f_right = interp_crossing(right - 1, right)
    return float(f_right - f_left)


# ── Helper: SNR from spectrum ─────────────────────────────────────────────────
def compute_snr(freq_axis: np.ndarray, spectrum: np.ndarray,
                peak_freq: float, n_signal_bins: int = SNR_SIGNAL_BINS) -> float:
    """
    SNR = mean power in signal bins / mean power in noise bins (dB).
    Signal bins: the n_signal_bins closest to peak_freq.
    Noise bins : everything in [MIN_SEARCH_HZ, MAX_SEARCH_HZ] that is not signal.
    """
    power = np.abs(spectrum) ** 2

    # valid search range (same as peak detection)
    valid = (np.abs(freq_axis) >= MIN_SEARCH_HZ) & (np.abs(freq_axis) <= MAX_SEARCH_HZ)

    dist        = np.abs(freq_axis - peak_freq)
    sorted_idx  = np.where(valid)[0][np.argsort(dist[valid])]
    signal_idx  = sorted_idx[:n_signal_bins]

    signal_mask = np.zeros(len(freq_axis), dtype=bool)
    signal_mask[signal_idx] = True

    noise_mask = valid & ~signal_mask

    mean_sig   = np.mean(power[signal_mask])
    mean_noise = np.mean(power[noise_mask])

    if mean_noise == 0:
        return float("inf")
    return float(10.0 * np.log10(mean_sig / mean_noise))


# ── Core analysis function ────────────────────────────────────────────────────
def analyze_file(filepath: Path, direction: int = 1) -> dict:
    """
    direction: +1 for targets moving toward the sensor (positive Doppler),
               -1 for targets moving away (negative Doppler).
    Because the I and Q channels in this hardware are not in quadrature,
    the complex FFT is symmetric.  We search the positive-frequency half and
    apply the direction sign afterwards.
    """
    sample_period, data = load_radar_file(filepath, CHANNELS)
    fs = 1.0 / sample_period

    i_raw, q_raw = extract_iq_channels(data, I_CH, Q_CH)
    # Automatically detect the constant-speed motion window (skips acceleration
    # ramp and stationary/return phases).
    i_for_detect = i_raw - np.mean(i_raw)
    start_sample, end_sample = detect_motion_window(i_for_detect, fs)
    i_raw, q_raw = trim_signals(i_raw, q_raw, start_sample, end_sample)

    i_dc = remove_dc(i_raw)
    q_dc = remove_dc(q_raw)

    # ── IQ amplitude balance (point 3) ───────────────────────────────────────
    rms_i = float(np.sqrt(np.mean(i_dc ** 2)))
    rms_q = float(np.sqrt(np.mean(q_dc ** 2)))
    scale = rms_i / rms_q if rms_q > 0 else 1.0
    q_scaled = q_dc * scale          # Q scaled to match I amplitude

    t = np.arange(len(i_dc)) * sample_period
    observation_time = len(i_dc) * sample_period
    freq_resolution  = 1.0 / observation_time

    # ── FFT with scaled IQ ───────────────────────────────────────────────────
    z      = build_complex_signal(i_dc, q_scaled)
    window = get_window(WINDOW_NAME, len(z))
    freq_axis, spectrum = compute_fft(z, fs, window)
    spec_db = magnitude_db(spectrum)

    # The I and Q channels in this hardware are not in quadrature (0° phase
    # difference instead of 90°), so the complex FFT spectrum is symmetric
    # around DC.  Searching both halves causes the algorithm to randomly pick
    # the negative-frequency mirror image.  We therefore zero-out the negative
    # half before peak search so the detected peak is always at a positive
    # frequency, giving the correct Doppler magnitude.
    spec_pos = spectrum.copy()
    spec_pos[freq_axis < 0] = 0.0

    peak_freq_pos, _, method = find_doppler_peak(
        freq_axis          = freq_axis,
        spectrum           = spec_pos,
        dc_exclusion_hz    = DC_EXCLUSION_HZ,
        min_search_hz      = MIN_SEARCH_HZ,
        max_search_hz      = MAX_SEARCH_HZ,
        peak_prominence_db = PEAK_PROMINENCE_DB,
    )
    # Apply the known direction (sign) — positive-half search gives magnitude only
    peak_freq = float(direction) * abs(peak_freq_pos)
    velocity  = doppler_to_velocity(peak_freq, RADAR_F0_HZ)

    snr_db  = compute_snr(freq_axis, spectrum, peak_freq)
    bw_3db  = measure_3db_bandwidth(freq_axis, spec_db, peak_freq)

    return {
        "file":             filepath.name,
        "fs":               fs,
        "n_samples":        len(z),
        "t":                t,
        "i_dc":             i_dc,
        "q_dc":             q_dc,
        "q_scaled":         q_scaled,
        "rms_i":            rms_i,
        "rms_q":            rms_q,
        "iq_scale":         scale,
        "freq_axis":        freq_axis,
        "spectrum":         spectrum,
        "spec_db":          spec_db,
        "peak_freq":        peak_freq,
        "velocity":         velocity,
        "snr_db":           snr_db,
        "bw_3db":           bw_3db,
        "freq_resolution":  freq_resolution,
        "observation_time": observation_time,
        "method":           method,
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _example_label(group_results):
    """Pick one representative result from each group for detailed plots."""
    return [(label, nominal, results[0])
            for label, nominal, results in group_results if results]


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Load all files ────────────────────────────────────────────────────────
    group_results: list[tuple[str, float, list[dict]]] = []
    all_results:   list[dict] = []

    for label, nominal, prefix in GROUPS:
        files = sorted(DATA_DIR.glob(f"{prefix}_*.bin"))
        if not files:
            print(f"WARNING: no files for prefix '{prefix}'")
        direction = 1 if nominal >= 0 else -1
        results = []
        for f in files:
            print(f"Analysing {f.name} ...", end="  ")
            r = analyze_file(f, direction=direction)
            results.append(r)
            print(f"fD = {r['peak_freq']:+.2f} Hz  v = {r['velocity']:+.4f} m/s  "
                  f"SNR = {r['snr_db']:.1f} dB  BW_3dB = {r['bw_3db']:.3f} Hz")
        group_results.append((label, nominal, results))
        all_results.extend(results)

    # ═════════════════════════════════════════════════════════════════════════
    # POINT 5 – printed statistics
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print(f"{'File':<22} {'fD (Hz)':>9} {'v_meas (m/s)':>13} "
          f"{'v_nom (m/s)':>12} {'SNR (dB)':>9} {'BW_3dB (Hz)':>12}")
    print("-" * 72)
    for label, nominal, results in group_results:
        for r in results:
            print(f"{r['file']:<22} {r['peak_freq']:>+9.3f} {r['velocity']:>+13.4f} "
                  f"{nominal:>+12.3f} {r['snr_db']:>9.1f} {r['bw_3db']:>12.4f}")
        vels = [r["velocity"] for r in results]
        mu   = np.mean(vels)
        sig  = np.std(vels, ddof=1) if len(vels) > 1 else 0.0
        err  = mu - nominal
        print(f"  mean = {mu:+.4f} m/s   std = {sig:.4f} m/s   "
              f"bias = {err:+.4f} m/s   (nominal = {nominal:+.3f} m/s)")
        # Doppler resolution comparison
        obs_times = [r["observation_time"] for r in results]
        bws       = [r["bw_3db"] for r in results]
        df_theo   = np.mean([1.0 / T for T in obs_times])
        bw_mean   = np.mean(bws)
        print(f"  Theoretical df = 1/T = {df_theo:.4f} Hz   "
              f"Measured 3-dB BW = {bw_mean:.4f} Hz")
        print()
    print("=" * 72)

    # ═════════════════════════════════════════════════════════════════════════
    # POINT 2 – measured vs. theoretical velocity
    # ═════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Point 2 – Measured vs. theoretical velocity", fontsize=12)

    # Left: scatter plot measured vs nominal
    ax = axes2[0]
    group_colors = ["steelblue", "darkorange", "seagreen"]
    for (label, nominal, results), col in zip(group_results, group_colors):
        vels = [r["velocity"] for r in results]
        ax.scatter([nominal] * len(vels), vels, color=col, s=60,
                   zorder=3, label=label)
        mu  = np.mean(vels)
        sig = np.std(vels, ddof=1) if len(vels) > 1 else 0.0
        ax.errorbar(nominal, mu, yerr=sig, fmt="D", color=col,
                    capsize=6, markersize=8, linewidth=2)

    # ideal line
    all_nom = [nom for _, nom, _ in group_results]
    lim = max(abs(min(all_nom)), abs(max(all_nom))) * 1.4
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="Ideal (measured = nominal)")
    ax.set_xlabel("Nominal velocity [m/s]")
    ax.set_ylabel("Measured velocity [m/s]")
    ax.set_title("Measured vs. theoretical (diamonds = mean ± std)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # Right: bar chart of all measurements
    ax = axes2[1]
    x_labels, x_vals, bar_colors = [], [], []
    idx = 0
    for (label, nominal, results), col in zip(group_results, group_colors):
        for r in results:
            x_labels.append(r["file"].replace(".bin", ""))
            x_vals.append(r["velocity"])
            bar_colors.append(col)
        n = len(results)
        if n:
            ax.hlines(nominal, idx - 0.5, idx + n - 0.5,
                      colors="red", linestyles="--", linewidth=1.5,
                      label=f"Nominal {nominal:+.2f} m/s")
        idx += n

    ax.bar(x_labels, x_vals, color=bar_colors, edgecolor="black", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Radial velocity [m/s]")
    ax.set_title("All measurements")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.4)
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()

    # ═════════════════════════════════════════════════════════════════════════
    # POINT 3 – I/Q time series + IQ scaling + Doppler spectrum + SNR
    # (one representative run per group)
    # ═════════════════════════════════════════════════════════════════════════
    examples = _example_label(group_results)
    n_ex = len(examples)

    fig3, axes3 = plt.subplots(3, n_ex, figsize=(5 * n_ex, 10))
    fig3.suptitle("Point 3 – Raw I/Q, IQ scaling, Doppler spectrum", fontsize=12)
    if n_ex == 1:
        axes3 = axes3[:, np.newaxis]

    for col, (label, nominal, r) in enumerate(examples):
        t_plot = r["t"]

        # Row 0: raw I and Q time series
        ax = axes3[0, col]
        ax.plot(t_plot, r["i_dc"],  linewidth=0.7, label=f"I  (rms={r['rms_i']:.1f})")
        ax.plot(t_plot, r["q_dc"],  linewidth=0.7, label=f"Q  (rms={r['rms_q']:.1f})", alpha=0.85)
        ax.set_title(f"{r['file']}\n{label}", fontsize=9)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("ADC counts")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 1: I and Q after amplitude scaling
        ax = axes3[1, col]
        ax.plot(t_plot, r["i_dc"],      linewidth=0.7, label="I (unchanged)")
        ax.plot(t_plot, r["q_scaled"],  linewidth=0.7, alpha=0.85,
                label=f"Q×{r['iq_scale']:.3f}  (rms now = {r['rms_i']:.1f})")
        ax.set_title("After IQ amplitude scaling")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("ADC counts")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 2: Doppler spectrum with peak and SNR
        ax = axes3[2, col]
        fa = r["freq_axis"]
        ax.plot(fa, r["spec_db"], linewidth=0.8, color="steelblue")
        ax.axvline(r["peak_freq"], color="red", linestyle="--", linewidth=1.3,
                   label=f"fD = {r['peak_freq']:+.2f} Hz\n"
                         f"v  = {r['velocity']:+.3f} m/s\n"
                         f"SNR = {r['snr_db']:.1f} dB")
        # mark 3-dB level
        peak_bin = int(np.argmin(np.abs(fa - r["peak_freq"])))
        peak_lv  = r["spec_db"][peak_bin]
        ax.axhline(peak_lv - 3.0, color="gray", linestyle=":", linewidth=1.0,
                   label=f"−3 dB level\nBW = {r['bw_3db']:.3f} Hz")
        # zoom so the Doppler peak is clearly visible (5× the peak frequency,
        # minimum ±100 Hz, maximum ±MAX_SEARCH_HZ)
        zoom = min(max(abs(r["peak_freq"]) * 5, 100), MAX_SEARCH_HZ)
        ax.set_xlim(-zoom, zoom)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_title("Doppler spectrum (IQ-scaled)")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ═════════════════════════════════════════════════════════════════════════
    # POINT 4 – Doppler resolution: theoretical vs. measured 3-dB width
    # ═════════════════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    fig4.suptitle("Point 4 – Doppler frequency resolution", fontsize=12)

    x_pos, x_tick_labels = [], []
    idx = 0
    for (label, nominal, results), col in zip(group_results, group_colors):
        for r in results:
            theo = r["freq_resolution"]
            meas = r["bw_3db"]
            ax4.bar(idx,       theo, color="steelblue", alpha=0.6,
                    label="Theoretical df = 1/T" if idx == 0 else "_")
            ax4.bar(idx + 0.4, meas, color="darkorange", alpha=0.8,
                    label="Measured 3-dB BW"    if idx == 0 else "_")
            x_pos.append(idx + 0.2)
            x_tick_labels.append(r["file"].replace(".bin", ""))
            idx += 1
        idx += 0.5   # gap between groups

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x_tick_labels, rotation=35, ha="right", fontsize=8)
    ax4.set_ylabel("Frequency [Hz]")
    ax4.set_title("Theoretical resolution (df = 1/T) vs. measured 3-dB peak width\n"
                  "(blue = theoretical, orange = measured)")
    ax4.legend(fontsize=9)
    ax4.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()

    # ═════════════════════════════════════════════════════════════════════════
    # POINT 5 – velocity accuracy: mean, std, error per group
    # ═════════════════════════════════════════════════════════════════════════
    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))
    fig5.suptitle("Point 5 – Velocity measurement accuracy", fontsize=12)

    # Left: all individual measurements with mean ± std per group
    ax = axes5[0]
    group_x, group_means, group_stds, group_noms = [], [], [], []
    x = 0
    xticks, xticklabels = [], []
    for (label, nominal, results), col in zip(group_results, group_colors):
        vels = [r["velocity"] for r in results]
        mu   = np.mean(vels)
        sig  = np.std(vels, ddof=1) if len(vels) > 1 else 0.0

        for i, v in enumerate(vels):
            ax.scatter(x + i, v, color=col, s=50, zorder=3)
            xticks.append(x + i)
            xticklabels.append(results[i]["file"].replace(".bin", ""))

        xc = x + (len(vels) - 1) / 2
        ax.hlines(mu, x - 0.3, x + len(vels) - 0.7,
                  colors=col, linewidth=2.0,
                  label=f"{label}: mean={mu:+.4f}, std={sig:.4f} m/s")
        ax.fill_between([x - 0.3, x + len(vels) - 0.7], mu - sig, mu + sig,
                        color=col, alpha=0.15)
        ax.hlines(nominal, x - 0.3, x + len(vels) - 0.7,
                  colors="red", linestyles="--", linewidth=1.2)
        group_x.append(xc)
        group_means.append(mu)
        group_stds.append(sig)
        group_noms.append(nominal)
        x += len(vels) + 1

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Radial velocity [m/s]")
    ax.set_title("Individual measurements\n(solid line = mean, shading = ±std, dashed = nominal)")
    ax.legend(fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.7)

    # Right: error bars (mean ± std) vs. nominal
    ax = axes5[1]
    for i, ((label, nominal, _), col) in enumerate(zip(group_results, group_colors)):
        ax.errorbar(group_noms[i], group_means[i],
                    yerr=group_stds[i],
                    fmt="o", color=col, capsize=8, markersize=9,
                    linewidth=2.0, label=f"{label}\nstd = {group_stds[i]:.4f} m/s")
    lim2 = max(abs(v) for v in group_noms) * 1.5
    ax.plot([-lim2, lim2], [-lim2, lim2], "k--", linewidth=1.0, label="Ideal")
    ax.set_xlabel("Nominal velocity [m/s]")
    ax.set_ylabel("Measured mean velocity [m/s]")
    ax.set_title("Mean ± std vs. nominal velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
