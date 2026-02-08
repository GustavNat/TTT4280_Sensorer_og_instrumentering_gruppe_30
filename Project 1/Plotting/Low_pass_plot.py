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

    # --- Plot Bode ---
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_mag.semilogx(f, H_mag_dB, label=r"$|H(f)|$ (Ch2/Ch1) [dB]")
    # Optional: show the raw channel magnitudes too (comment out if you want a cleaner plot)
    ax_mag.semilogx(f, ch1_mag_dB, "--", linewidth=1.0, label="Ch1 magnitude [dB]")
    ax_mag.semilogx(f, ch2_mag_dB, "--", linewidth=1.0, label="Ch2 magnitude [dB]")

    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True, which="both")
    ax_mag.legend()

    ax_phase.semilogx(f, phase_deg, label="Phase [deg]")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.grid(True, which="both")
    ax_phase.legend()

    fig.suptitle("Measured Bode plot – Low-pass filter")
    fig.tight_layout()
    plt.show()

    # Optional: save figure
    # fig.savefig("bode_lowpass_measured.png", dpi=200)


if __name__ == "__main__":
    main()