"""Generate illustrative plots of the synthetic ECG/PPG generators.

Outputs PNGs under ``results/_synthetic_plots/``:

* ``ecg_clean_vs_noisy.png`` -- clean ECG + same signal at SNR 20/10/0 dB.
* ``ppg_clean_vs_noisy.png`` -- same panel layout for PPG.
* ``ecg_morphology_zoom.png`` -- single beat with QRS / ST / T markers.
* ``ppg_morphology_zoom.png`` -- single pulse with systolic / notch / diastolic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compressionkit.synthetic import (
    NoiseSpec,
    add_noise,
    ecg_mcsharry,
    ppg_dynamical,
)


OUT_DIR = Path("results/_synthetic_plots")


def _plot_clean_vs_noisy(
    clean: np.ndarray,
    fs: float,
    snr_levels: list[float],
    noise_spec: NoiseSpec,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    t = np.arange(clean.size) / fs
    n_panels = 1 + len(snr_levels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 1.6 * n_panels), sharex=True)
    axes[0].plot(t, clean, color="#1f77b4", lw=1.2)
    axes[0].set_title(f"{title} — clean synthetic (ground truth)", fontsize=10, loc="left")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(alpha=0.3)

    for ax, snr in zip(axes[1:], snr_levels):
        noisy, _ = add_noise(clean, sample_rate=fs, snr_db=snr, spec=noise_spec, seed=1)
        ax.plot(t, noisy, color="#d62728", lw=0.9, alpha=0.9)
        ax.plot(t, clean, color="#1f77b4", lw=1.0, alpha=0.5, label="clean")
        ax.set_title(f"noisy @ SNR = {snr:.0f} dB", fontsize=10, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_ecg_morphology_zoom(fs: float, path: Path) -> None:
    """Single-beat zoom annotating P / Q / R / S / T regions."""
    sig = ecg_mcsharry(duration_s=3.0, sample_rate=fs, hr_mean=72.0, hr_std=0.0, seed=42)
    sig = sig * 100.0  # scale to plot-friendly amplitude
    # Pick the middle beat (most settled).
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(sig, height=0.3 * sig.max(), distance=int(0.4 * fs))
    if peaks.size < 2:
        print(f"  WARNING: only {peaks.size} peak(s) detected for zoom plot")
        return
    p = int(peaks[len(peaks) // 2])
    half = int(0.45 * fs)
    a = max(p - half, 0)
    b = min(p + half, sig.size)
    seg = sig[a:b]
    t = (np.arange(seg.size) - (p - a)) / fs * 1000.0  # ms relative to R

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, seg, color="#1f77b4", lw=1.4)
    ax.axvline(0, color="#444", lw=0.6, ls="--", alpha=0.5)
    ptp = float(np.ptp(sig))
    ax.annotate("R", xy=(0, seg[p - a]), xytext=(0, seg[p - a] + 0.15 * ptp),
                ha="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#444"))
    # Q at -30 ms, S at +30 ms, T peak ~+250 ms, P peak ~-150 ms (defaults).
    for label, t_ms in [("P", -150), ("Q", -30), ("S", +30), ("T", +250)]:
        i = int(round((t_ms / 1000.0) * fs)) + (p - a)
        if 0 <= i < seg.size:
            ax.annotate(label, xy=(t_ms, seg[i]),
                        xytext=(t_ms, seg[i] - 0.10 * ptp),
                        ha="center", fontsize=9,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#888"))
    ax.set_title("Synthetic ECG — single-beat morphology (McSharry-Clifford)", fontsize=10, loc="left")
    ax.set_xlabel("time relative to R (ms)")
    ax.set_ylabel("amplitude (a.u.)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_ppg_morphology_zoom(fs: float, path: Path) -> None:
    """Single-pulse zoom annotating systolic / dicrotic notch / diastolic."""
    sig = ppg_dynamical(duration_s=3.0, sample_rate=fs, hr_mean=72.0, hr_std=0.0, seed=42)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(sig, height=0.3 * sig.max(), distance=int(0.4 * fs))
    if peaks.size < 2:
        print(f"  WARNING: only {peaks.size} systolic peak(s) detected for zoom plot")
        return
    p = int(peaks[len(peaks) // 2])
    half = int(0.6 * fs)
    a = max(p - half, 0)
    b = min(p + half, sig.size)
    seg = sig[a:b]
    t = (np.arange(seg.size) - (p - a)) / fs * 1000.0

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, seg, color="#2ca02c", lw=1.5)
    ax.axvline(0, color="#444", lw=0.6, ls="--", alpha=0.5)
    ptp = float(np.ptp(sig))
    ax.annotate("systolic", xy=(0, seg[p - a]),
                xytext=(0, seg[p - a] + 0.10 * ptp),
                ha="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#444"))
    # Dicrotic notch at theta=pi/3 -> roughly +160 ms at HR=72.
    notch_ms = +160
    dia_ms = +260
    for label, t_ms in [("dicrotic\nnotch", notch_ms), ("diastolic", dia_ms)]:
        i = int(round((t_ms / 1000.0) * fs)) + (p - a)
        if 0 <= i < seg.size:
            ax.annotate(label, xy=(t_ms, seg[i]),
                        xytext=(t_ms, seg[i] + 0.18 * ptp),
                        ha="center", fontsize=8,
                        arrowprops=dict(arrowstyle="->", lw=0.5, color="#888"))
    ax.set_title("Synthetic PPG — single-pulse morphology (dynamical, 3-Gaussian)", fontsize=10, loc="left")
    ax.set_xlabel("time relative to systolic peak (ms)")
    ax.set_ylabel("amplitude (a.u.)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ECG: 8-second window, 256 Hz.
    fs_e = 256.0
    ecg = ecg_mcsharry(duration_s=8.0, sample_rate=fs_e, hr_mean=72.0, hr_std=1.5, seed=42)
    ecg = ecg * 100.0  # scale to a.u. that's easy to read
    ecg_noise = NoiseSpec(
        weights={"baseline_wander": 1.0, "emg": 1.0, "powerline": 0.3, "gauss": 0.3},
        powerline_hz=60.0,
    )
    _plot_clean_vs_noisy(
        ecg, fs_e, snr_levels=[20.0, 10.0, 0.0], noise_spec=ecg_noise,
        title="Synthetic ECG", ylabel="amplitude (a.u.)",
        path=OUT_DIR / "ecg_clean_vs_noisy.png",
    )

    # PPG: 10-second window, 64 Hz.
    fs_p = 64.0
    ppg = ppg_dynamical(duration_s=10.0, sample_rate=fs_p, hr_mean=72.0, hr_std=1.0, seed=42)
    ppg_noise = NoiseSpec(weights={"baseline_wander": 1.0, "motion": 1.5, "gauss": 0.5})
    _plot_clean_vs_noisy(
        ppg, fs_p, snr_levels=[20.0, 10.0, 0.0], noise_spec=ppg_noise,
        title="Synthetic PPG", ylabel="amplitude (a.u.)",
        path=OUT_DIR / "ppg_clean_vs_noisy.png",
    )

    _plot_ecg_morphology_zoom(fs_e, OUT_DIR / "ecg_morphology_zoom.png")
    _plot_ppg_morphology_zoom(fs_p, OUT_DIR / "ppg_morphology_zoom.png")
    print(f"\nAll plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
