"""
Bode plot (measured) for low-pass filter.

CSV format (starting from line 2 / skip header line):
Frequency (Hz), Channel 1 Magnitude (dB), Channel 2 Magnitude (dB), Channel 2 Phase (deg)

Magnitude transfer is computed as:
|H(f)|_dB = Mag2_dB - Mag1_dB = 20*log10(|V2/V1|)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "Measurements" / "Low-pass_filter" / "Low-pass_filter_10_ohm_resistor.csv"

# --- Component values for theoretical model ---
L_H = 100e-3          # 100 mH
C2_F = 470e-6         # 470 uF
C3_F = 100e-9         # 100 nF
C_OUT_F = C2_F + C3_F
R_OHM = 10          # Damping resistor (ohms)


def H_mag_damped(f_hz: np.ndarray, R_ohm: float, L_h: float, C_out_f: float) -> np.ndarray:
    """Theoretical transfer function magnitude for damped 2nd-order low-pass."""
    omega = 2.0 * np.pi * f_hz
    real = 1.0 - (omega**2) * L_h * C_out_f
    imag = omega * R_ohm * C_out_f
    denom = np.sqrt(real**2 + imag**2)
    denom = np.maximum(denom, 1e-18)
    return 1.0 / denom


def main() -> None:
    # Read from line 2 (skip line 1)
    df = pd.read_csv(CSV_FILE, skiprows=1, header=None)

    if df.shape[1] < 4:
        raise ValueError(
            f"Expected at least 4 columns, got {df.shape[1]}. "
            "Check delimiter/format in the CSV file."
        )

    df = df.iloc[:, :4].copy()
    df.columns = ["f_Hz", "ch1_mag_dB", "ch2_mag_dB", "ch2_phase_deg"]

    # Convert to numeric (drop non-numeric rows safely)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["f_Hz"]).sort_values("f_Hz")
    f = df["f_Hz"].to_numpy()
    ch1_mag_dB = df["ch1_mag_dB"].to_numpy()
    ch2_mag_dB = df["ch2_mag_dB"].to_numpy()
    phase_deg = df["ch2_phase_deg"].to_numpy()

    # Transfer function magnitude in dB (Ch2/Ch1)
    H_mag_dB = ch2_mag_dB - ch1_mag_dB

    # Find -3dB cutoff frequency (where amplitude = -3 dB)
    cutoff_level = -3.0

    # Find where magnitude crosses -3dB level
    f_cutoff = None
    for i in range(1, len(H_mag_dB)):
        if H_mag_dB[i-1] >= cutoff_level and H_mag_dB[i] < cutoff_level:
            # Linear interpolation between the two points
            slope = (f[i] - f[i-1]) / (H_mag_dB[i] - H_mag_dB[i-1])
            f_cutoff = f[i-1] + slope * (cutoff_level - H_mag_dB[i-1])
            break

    if f_cutoff is None:
        f_cutoff = f[-1]  # Cutoff beyond measured range

    print(f"Measured cutoff frequency (-3dB): {f_cutoff:.2f} Hz")

    # --- Compute theoretical response ---
    f_theory = np.logspace(np.log10(f.min()), np.log10(f.max()), 500)
    H_theory = H_mag_damped(f_theory, R_OHM, L_H, C_OUT_F)
    H_theory_dB = 20.0 * np.log10(H_theory)

    # Find theoretical -3dB cutoff frequency
    f_cutoff_theory = None
    for i in range(1, len(H_theory_dB)):
        if H_theory_dB[i-1] >= cutoff_level and H_theory_dB[i] < cutoff_level:
            slope = (f_theory[i] - f_theory[i-1]) / (H_theory_dB[i] - H_theory_dB[i-1])
            f_cutoff_theory = f_theory[i-1] + slope * (cutoff_level - H_theory_dB[i-1])
            break

    if f_cutoff_theory is None:
        f_cutoff_theory = f_theory[-1]

    print(f"Theoretical cutoff frequency (-3dB): {f_cutoff_theory:.2f} Hz")

    # --- Plot amplitude response ---
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogx(f, H_mag_dB, label=r"Measured $|H(j2\pi f)|$")
    ax.semilogx(f_theory, H_theory_dB, '--', linewidth=1.5, label=r"Theoretical $|H(j2\pi f)|$")

    # Mark measured cutoff frequency
    ax.axvline(f_cutoff, color='r', linestyle=':', linewidth=1.5, label=f"Measured $f_c$ = {f_cutoff:.1f} Hz")
    ax.plot(f_cutoff, cutoff_level, 'ro', markersize=8)

    # Mark theoretical cutoff frequency
    ax.axvline(f_cutoff_theory, color='g', linestyle=':', linewidth=1.5, label=f"Theoretical $f_c$ = {f_cutoff_theory:.1f} Hz")
    ax.plot(f_cutoff_theory, cutoff_level, 'go', markersize=8)

    # Mark -3dB level
    ax.axhline(cutoff_level, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, which="both")
    ax.legend()

    fig.suptitle("Amplitude response – Low-pass filter (Measured vs Theoretical)")
    fig.tight_layout()
    plt.show()

    # Optional: save figure
    # fig.savefig("bode_lowpass_measured.png", dpi=200)


if __name__ == "__main__":
    main()