"""Adversarial input bank for codec evaluation.

Generators (pathological inputs):

- :func:`make_zero_input`         — silence: tests "does the codec hallucinate from nothing?"
- :func:`make_gaussian_noise`     — pure noise: tests denoising / mode collapse
- :func:`make_dc_offset`          — constant level
- :func:`make_sinusoid`           — narrowband tone(s) at chosen frequencies
- :func:`make_step`               — step discontinuity
- :func:`make_zero_burst`         — packet-loss simulation: real signal with zero gaps
- :func:`make_signal_zero_mix`    — random fraction of samples zeroed out
- :func:`inject_gaussian`         — additive Gaussian noise at target input SNR
- :func:`inject_baseline_wander`  — low-frequency sinusoid added to signal
- :func:`inject_powerline`        — 50/60 Hz sinusoid added to signal

Battery runner:

- :func:`run_adversarial_battery` — applies a Codec to all generators and
  returns a list of :class:`AdversarialResult`. The result captures the
  measurable "no-hallucination" evidence:

      energy_ratio       = ||y||^2 / ||x||^2     (low for zero/noise input)
      hallucinated_peaks = #peaks in output when input has none
      band_power         = power in physiological band(s)

These numbers are what we put on a slide for customer meetings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from compressionkit.evaluation.codec import Codec

__all__ = [
    "PHYSIO_BANDS",
    "AdversarialResult",
    "inject_baseline_wander",
    "inject_gaussian",
    "inject_powerline",
    "make_dc_offset",
    "make_gaussian_noise",
    "make_signal_zero_mix",
    "make_sinusoid",
    "make_step",
    "make_zero_burst",
    "make_zero_input",
    "run_adversarial_battery",
]


# Physiological bands for hallucination scoring
PHYSIO_BANDS: dict[str, tuple[float, float]] = {
    "ppg": (0.5, 3.0),  # pulse fundamental
    "ecg": (0.5, 40.0),  # diagnostic band
}


# ---------------------------------------------------------------------------
# Generators (no signal in)
# ---------------------------------------------------------------------------


def make_zero_input(frame_size: int, n_frames: int = 8) -> np.ndarray:
    """All-zeros frames. Shape ``(n_frames, frame_size)``."""
    return np.zeros((int(n_frames), int(frame_size)), dtype=np.float32)


def make_gaussian_noise(
    frame_size: int,
    n_frames: int = 8,
    *,
    std: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Pure Gaussian noise frames."""
    rng = np.random.default_rng(seed)
    return (std * rng.standard_normal((int(n_frames), int(frame_size)))).astype(np.float32)


def make_dc_offset(frame_size: int, offsets: list[float]) -> np.ndarray:
    """One frame per offset value, all samples equal to the offset."""
    return np.stack([np.full(int(frame_size), float(v), dtype=np.float32) for v in offsets])


def make_sinusoid(
    frame_size: int,
    fs: int,
    *,
    freqs_hz: list[float],
    amp: float = 1.0,
) -> np.ndarray:
    """One frame per requested frequency: ``amp * sin(2π f t)``."""
    t = np.arange(int(frame_size)) / float(fs)
    return np.stack([(amp * np.sin(2 * np.pi * float(f) * t)).astype(np.float32) for f in freqs_hz])


def make_step(
    frame_size: int,
    *,
    step_positions: list[int],
    step_amp: float = 1.0,
) -> np.ndarray:
    """One frame per step position: 0 before, ``step_amp`` after."""
    frames = []
    for p in step_positions:
        frame = np.zeros(int(frame_size), dtype=np.float32)
        p = max(0, min(int(p), int(frame_size)))
        frame[p:] = float(step_amp)
        frames.append(frame)
    return np.stack(frames)


# ---------------------------------------------------------------------------
# Generators (modify a real signal)
# ---------------------------------------------------------------------------


def make_zero_burst(
    signal: np.ndarray,
    *,
    burst_len: int,
    n_bursts: int,
    seed: int = 0,
) -> np.ndarray:
    """Insert ``n_bursts`` zero gaps of length ``burst_len`` into *signal*.

    *signal* is ``(N, frame_size)``. Burst start positions are uniform random.
    Useful as a packet-loss / sensor-disconnect surrogate.
    """
    out = np.asarray(signal, dtype=np.float32).copy()
    n_frames, frame_size = out.shape
    burst_len = max(1, int(burst_len))
    rng = np.random.default_rng(seed)
    for i in range(n_frames):
        for _ in range(int(n_bursts)):
            start = int(rng.integers(0, max(1, frame_size - burst_len + 1)))
            out[i, start : start + burst_len] = 0.0
    return out


def make_signal_zero_mix(
    signal: np.ndarray,
    *,
    zero_fraction: float,
    seed: int = 0,
) -> np.ndarray:
    """Zero out a random fraction of samples in each frame.

    Args:
        signal: ``(N, frame_size)``.
        zero_fraction: Fraction in ``[0, 1]`` of samples to set to zero.
    """
    out = np.asarray(signal, dtype=np.float32).copy()
    if not 0.0 <= float(zero_fraction) <= 1.0:
        raise ValueError("zero_fraction must be in [0, 1]")
    n_frames, frame_size = out.shape
    rng = np.random.default_rng(seed)
    n_zero = round(zero_fraction * frame_size)
    if n_zero <= 0:
        return out
    for i in range(n_frames):
        idx = rng.choice(frame_size, size=n_zero, replace=False)
        out[i, idx] = 0.0
    return out


def inject_gaussian(
    signal: np.ndarray,
    *,
    input_snr_db: float,
    seed: int = 0,
) -> np.ndarray:
    """Add Gaussian noise calibrated to a target input SNR per frame."""
    out = np.asarray(signal, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    for i in range(out.shape[0]):
        sig_power = float(np.mean(out[i] ** 2))
        if sig_power <= 0:
            continue
        noise_power = sig_power / (10.0 ** (float(input_snr_db) / 10.0))
        noise = rng.standard_normal(out.shape[1]) * np.sqrt(noise_power)
        out[i] += noise.astype(np.float32)
    return out


def inject_baseline_wander(
    signal: np.ndarray,
    *,
    fs: int,
    amp: float,
    freq_hz: float = 0.2,
) -> np.ndarray:
    """Add a low-frequency sinusoid (baseline wander) to each frame."""
    out = np.asarray(signal, dtype=np.float32).copy()
    t = np.arange(out.shape[1]) / float(fs)
    wander = (float(amp) * np.sin(2 * np.pi * float(freq_hz) * t)).astype(np.float32)
    return out + wander[None, :]


def inject_powerline(
    signal: np.ndarray,
    *,
    fs: int,
    amp: float,
    freq_hz: float = 60.0,
) -> np.ndarray:
    """Add a powerline-frequency sinusoid to each frame."""
    return inject_baseline_wander(signal, fs=fs, amp=amp, freq_hz=freq_hz)


# ---------------------------------------------------------------------------
# Result + battery
# ---------------------------------------------------------------------------


@dataclass
class AdversarialResult:
    """One row in the adversarial report.

    Attributes:
        test_name: Human-readable label (e.g. ``"zero_input"``).
        n_frames: Number of frames evaluated.
        input_energy: Mean ``||x||^2`` per frame.
        output_energy: Mean ``||y||^2`` per frame.
        energy_ratio: ``output_energy / input_energy`` (``inf`` if input is zero
            and output is non-zero — that is the hallucination signal).
        output_l2_when_input_zero: Mean ``||y||_2`` on zero input (only filled
            when ``input_energy == 0``).
        hallucinated_peaks: Mean number of peaks detected in the output when
            no peaks exist in the input (or in the physio band).
        output_band_power: Mean output power within the modality's physio
            band as a fraction of total output power.
        reconstruction_prd_mean: When input has signal, reconstruction PRD;
            ``None`` for non-signal inputs.
        params: The generator parameters used (for reproducibility).
    """

    test_name: str
    n_frames: int
    input_energy: float
    output_energy: float
    energy_ratio: float
    output_l2_when_input_zero: float | None = None
    hallucinated_peaks: float = 0.0
    output_band_power: float = 0.0
    reconstruction_prd_mean: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _peak_count_in_band(
    signal_1d: np.ndarray,
    *,
    fs: int,
    band: tuple[float, float],
    min_height: float = 0.05,
) -> int:
    """Count peaks consistent with the physiological band.

    Minimum peak distance corresponds to the band's upper frequency. The
    height threshold is ``max(0.5 * std, min_height)`` — the relative term
    suppresses tiny ripples within a peaky signal, the absolute term
    ``min_height`` prevents low-amplitude noise (e.g. zero-input
    hallucination at sub-physiological scale) from registering as peaks.

    Args:
        signal_1d: 1-D signal to scan.
        fs: Sample rate (Hz).
        band: ``(low_hz, high_hz)`` — only ``high_hz`` is used (sets the
            minimum inter-peak distance).
        min_height: Absolute peak-amplitude floor. The default of ``0.05``
            assumes layer-normalized signals (scale ~1.0): any peak below
            5 % of typical signal amplitude is considered noise.
    """
    if signal_1d.size == 0 or np.allclose(signal_1d, 0.0):
        return 0
    f_hi = max(0.1, float(band[1]))
    min_dist = max(1, round(fs / f_hi))
    std = float(np.std(signal_1d))
    if std <= 1e-9:
        return 0
    height = max(0.5 * std, float(min_height))
    peaks, _ = scipy_signal.find_peaks(signal_1d, distance=min_dist, height=height)
    return int(peaks.size)


def _band_power_fraction(signal_1d: np.ndarray, *, fs: int, band: tuple[float, float]) -> float:
    """Fraction of output power that sits inside *band*."""
    if signal_1d.size == 0:
        return 0.0
    total = float(np.sum(signal_1d.astype(np.float64) ** 2))
    if total <= 1e-12:
        return 0.0
    nperseg = min(len(signal_1d), 256)
    f, pxx = scipy_signal.welch(signal_1d, fs=fs, nperseg=nperseg)
    df = f[1] - f[0] if len(f) > 1 else 1.0
    mask = (f >= band[0]) & (f <= band[1])
    band_power = float(np.sum(pxx[mask]) * df)
    total_power = float(np.sum(pxx) * df)
    return band_power / total_power if total_power > 1e-12 else 0.0


def _evaluate_frames(
    codec: Codec,
    inputs: np.ndarray,
    *,
    test_name: str,
    band: tuple[float, float],
    params: dict[str, Any],
    measure_prd: bool,
) -> AdversarialResult:
    n_frames = int(inputs.shape[0])
    in_energy: list[float] = []
    out_energy: list[float] = []
    out_l2_zero: list[float] = []
    hallucinated: list[int] = []
    band_powers: list[float] = []
    prds: list[float] = []

    for i in range(n_frames):
        x = inputs[i]
        enc = codec.encode(x)
        y = codec.decode(enc)
        # Crop/pad y to match x
        if y.shape != x.shape:
            n = min(y.shape[0], x.shape[0])
            y = y[:n]
            x = x[:n]
        ex = float(np.sum(x.astype(np.float64) ** 2))
        ey = float(np.sum(y.astype(np.float64) ** 2))
        in_energy.append(ex)
        out_energy.append(ey)
        if ex <= 1e-12:
            out_l2_zero.append(float(np.linalg.norm(y)))
            hallucinated.append(_peak_count_in_band(np.asarray(y), fs=codec.sample_rate, band=band))
        band_powers.append(_band_power_fraction(np.asarray(y), fs=codec.sample_rate, band=band))
        if measure_prd and ex > 1e-12:
            err = x - y
            prds.append(100.0 * np.sqrt(float(np.sum(err**2)) / (ex + 1e-12)))

    mean_in = float(np.mean(in_energy)) if in_energy else 0.0
    mean_out = float(np.mean(out_energy)) if out_energy else 0.0
    if mean_in <= 1e-12:
        ratio = float("inf") if mean_out > 1e-12 else 0.0
    else:
        ratio = mean_out / mean_in

    return AdversarialResult(
        test_name=test_name,
        n_frames=n_frames,
        input_energy=mean_in,
        output_energy=mean_out,
        energy_ratio=ratio,
        output_l2_when_input_zero=float(np.mean(out_l2_zero)) if out_l2_zero else None,
        hallucinated_peaks=float(np.mean(hallucinated)) if hallucinated else 0.0,
        output_band_power=float(np.mean(band_powers)) if band_powers else 0.0,
        reconstruction_prd_mean=float(np.mean(prds)) if prds else None,
        params=params,
    )


def run_adversarial_battery(
    codec: Codec,
    *,
    n_frames: int = 16,
    seed: int = 0,
    signal_frames: np.ndarray | None = None,
    input_snr_db_sweep: list[float] | None = None,
) -> list[AdversarialResult]:
    """Apply the full adversarial battery to *codec*.

    Args:
        codec: Any object satisfying :class:`Codec`.
        n_frames: Frames per generator (where applicable).
        seed: RNG seed for reproducibility.
        signal_frames: Optional ``(N, frame_size)`` real-signal frames. Required
            for the injection / cutout / SNR-sweep tests. When ``None`` those
            tests are skipped.
        input_snr_db_sweep: SNRs (dB) at which to sweep additive Gaussian
            noise on the supplied signal frames. Defaults to ``[20, 10, 5, 0]``.

    Returns:
        A list of :class:`AdversarialResult`, one per test.
    """
    fs = codec.sample_rate
    band = PHYSIO_BANDS.get(codec.modality, (0.5, 40.0))
    results: list[AdversarialResult] = []

    # Zero input
    results.append(
        _evaluate_frames(
            codec,
            make_zero_input(codec.frame_size, n_frames=n_frames),
            test_name="zero_input",
            band=band,
            params={"n_frames": n_frames},
            measure_prd=False,
        )
    )

    # Pure Gaussian noise
    results.append(
        _evaluate_frames(
            codec,
            make_gaussian_noise(codec.frame_size, n_frames=n_frames, std=1.0, seed=seed),
            test_name="gaussian_noise",
            band=band,
            params={"std": 1.0, "n_frames": n_frames, "seed": seed},
            measure_prd=False,
        )
    )

    # DC offset
    results.append(
        _evaluate_frames(
            codec,
            make_dc_offset(codec.frame_size, offsets=[-1.0, 0.5, 1.0]),
            test_name="dc_offset",
            band=band,
            params={"offsets": [-1.0, 0.5, 1.0]},
            measure_prd=False,
        )
    )

    # Sinusoid at the physio band's lower and upper edge
    sin_freqs = [band[0], 0.5 * (band[0] + band[1]), band[1]]
    results.append(
        _evaluate_frames(
            codec,
            make_sinusoid(codec.frame_size, fs=fs, freqs_hz=sin_freqs, amp=1.0),
            test_name="sinusoid",
            band=band,
            params={"freqs_hz": sin_freqs, "amp": 1.0},
            measure_prd=False,
        )
    )

    # Step
    positions = [codec.frame_size // 4, codec.frame_size // 2, 3 * codec.frame_size // 4]
    results.append(
        _evaluate_frames(
            codec,
            make_step(codec.frame_size, step_positions=positions, step_amp=1.0),
            test_name="step",
            band=band,
            params={"step_positions": positions, "step_amp": 1.0},
            measure_prd=False,
        )
    )

    # Signal-based tests
    if signal_frames is not None and signal_frames.size > 0:
        sig = np.asarray(signal_frames, dtype=np.float32)

        # Input-SNR sweep (additive Gaussian)
        snr_db_list = input_snr_db_sweep or [20.0, 10.0, 5.0, 0.0]
        for snr_db in snr_db_list:
            noisy = inject_gaussian(sig, input_snr_db=float(snr_db), seed=seed)
            results.append(
                _evaluate_frames(
                    codec,
                    noisy,
                    test_name=f"signal_plus_gaussian_{int(snr_db)}dB",
                    band=band,
                    params={"input_snr_db": float(snr_db), "n_frames": int(sig.shape[0])},
                    measure_prd=True,
                )
            )

        # Zero-burst (packet loss surrogate)
        burst_len = max(1, codec.frame_size // 16)
        bursty = make_zero_burst(sig, burst_len=burst_len, n_bursts=2, seed=seed)
        results.append(
            _evaluate_frames(
                codec,
                bursty,
                test_name="signal_with_zero_bursts",
                band=band,
                params={"burst_len": burst_len, "n_bursts": 2},
                measure_prd=True,
            )
        )

        # Signal-zero mix
        mixed = make_signal_zero_mix(sig, zero_fraction=0.3, seed=seed)
        results.append(
            _evaluate_frames(
                codec,
                mixed,
                test_name="signal_zero_mix_30pct",
                band=band,
                params={"zero_fraction": 0.3},
                measure_prd=True,
            )
        )

    return results
