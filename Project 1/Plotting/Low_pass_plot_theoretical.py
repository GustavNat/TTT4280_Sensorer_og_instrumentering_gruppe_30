"""
Theoretical amplitude response for the *undamped* 2nd-order low-pass model:

    |H(jω)| = 1 / | 1 - ω^2 * L * C_out |

where:
    ω = 2πf
    C_out = C2 + C3

Note:
- This ideal model has a singularity at f0 = 1/(2π√(L C_out)).
  Real circuits do not diverge due to losses (ESR/DCR, source impedance).
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Set component values here ---
L_H = 100e-3          # 100 mH
C2_F = 470e-6         # 470 uF
C3_F = 100e-9         # 100 nF
C_OUT_F = C2_F + C3_F

# Frequency axis
F_MIN = 1.0
F_MAX = 1e5
N_POINTS = 2000

# Clip to keep plot readable near ideal resonance
MAX_DB = 40.0


def H_mag_undamped(f_hz: np.ndarray, L_h: float, C_out_f: float) -> np.ndarray:
    omega = 2.0 * np.pi * f_hz
    denom = np.abs(1.0 - (omega**2) * L_h * C_out_f)

    # Avoid division by zero numerically
    denom = np.maximum(denom, 1e-18)
    return 1.0 / denom

import numpy as np

def H_mag_damped(f_hz: np.ndarray, R_ohm: float, L_h: float, C_out_f: float) -> np.ndarray:

    omega = 2.0 * np.pi * f_hz
    real = 1.0 - (omega**2) * L_h * C_out_f
    imag = omega * R_ohm * C_out_f
    denom = np.sqrt(real**2 + imag**2)

    # Avoid division by zero numerically (should not happen for R>0, but safe)
    denom = np.maximum(denom, 1e-18)
    return 1.0 / denom





def main() -> None:
    # --- Frequency axis ---
    f = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_POINTS)

    # --- Undamped ---
    H_mag_u = H_mag_undamped(f, L_H, C_OUT_F)
    H_mag_u_dB = 20.0 * np.log10(H_mag_u)
    H_mag_u_dB = np.clip(H_mag_u_dB, -200.0, MAX_DB)

    f0 = 1.0 / (2.0 * np.pi * np.sqrt(L_H * C_OUT_F))

    fig_u, ax_u = plt.subplots(figsize=(9, 4.8))
    ax_u.semilogx(
        f, H_mag_u_dB,
        label=r"$20\log_{10}\!\left(\frac{1}{|1-\omega^2LC_{out}|}\right)$"
    )
    ax_u.axvline(
        f0,
        color="orange",
        linestyle="-",
        linewidth=2.0,
        alpha=0.9,
        zorder=5,
        label=rf"$f_0 \approx {f0:.1f}\ \mathrm{{Hz}}$",
    )
    ax_u.set_xlabel("Frequency (Hz)")
    ax_u.set_ylabel("Magnitude (dB)")
    ax_u.grid(True, which="both")
    ax_u.legend()
    ax_u.set_title("Theoretical amplitude response (undamped model)")
    fig_u.tight_layout()


    # --- Damped ---
    # Set your series damping resistor here:
    R_OHM = 10.0  # e.g. 10 ohm

    H_mag_d = H_mag_damped(f, R_OHM, L_H, C_OUT_F)
    H_mag_d_dB = 20.0 * np.log10(H_mag_d)
    H_mag_d_dB = np.clip(H_mag_d_dB, -200.0, MAX_DB)

    # Optional: frequency for peak magnitude (only if R is small enough)
    # omega_r = sqrt(omega0^2 - (R/L)^2 / 2)
    omega0 = 1.0 / np.sqrt(L_H * C_OUT_F)
    inside = omega0**2 - 0.5 * (R_OHM / L_H) ** 2
    f_r = np.sqrt(inside) / (2.0 * np.pi) if inside > 0 else None

    fig_d, ax_d = plt.subplots(figsize=(9, 4.8))
    ax_d.semilogx(
        f, H_mag_d_dB,
        label=r"$20\log_{10}\!\left(\frac{1}{\sqrt{(1-\omega^2LC_{out})^2+(\omega RC_{out})^2}}\right)$"
    )

    # ax_d.axvline(
    #     f0,
    #     color="orange",
    #     linestyle="-",
    #     linewidth=2.0,
    #     alpha=0.9,
    #     zorder=5,
    #     label=rf"$f_0 \approx {f0:.1f}\ \mathrm{{Hz}}$",
    # )

    # if f_r is not None:
    #     ax_d.axvline(
    #         f_r,
    #         color="black",
    #         linestyle="--",
    #         linewidth=1.5,
    #         alpha=0.9,
    #         zorder=5,
    #         label=rf"$f_r \approx {f_r:.1f}\ \mathrm{{Hz}}$",
    #     )

    ax_d.set_xlabel("Frequency (Hz)")
    ax_d.set_ylabel("Magnitude (dB)")
    ax_d.grid(True, which="both")
    ax_d.legend()
    ax_d.set_title(f"Theoretical amplitude response (damped model, R = {R_OHM:g} Ω)")
    fig_d.tight_layout()

    plt.show()



if __name__ == "__main__":
    main()
