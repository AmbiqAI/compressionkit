"""Analytical (non-learned) synthetic physiological signal generators.

These models produce ground-truth-clean PPG / ECG waveforms with explicit,
parametric control over morphology, heart rate, HRV, and respiration. They are
intended as deterministic test fixtures for codec evaluation -- the clean
output pairs with a controlled noise harness to give us
``(clean_truth, noisy_input)`` pairs that disambiguate fidelity-to-noise from
fidelity-to-truth.

Models implemented:

* :func:`ecg_mcsharry` -- McSharry-Clifford 2003 dynamical ECG model.
* :func:`ppg_dynamical` -- Sister 3-Gaussian dynamical PPG model in the same
  style.
* :func:`add_noise` -- Calibrated noise injection (baseline wander, EMG,
  motion, powerline) at target SNR.
"""

from __future__ import annotations

from compressionkit.synthetic.ecg_mcsharry import (
    EcgMorphologyParams,
    ecg_mcsharry,
)
from compressionkit.synthetic.noise import NoiseSpec, add_noise
from compressionkit.synthetic.ppg_dynamical import (
    PpgMorphologyParams,
    ppg_dynamical,
)

__all__ = [
    "EcgMorphologyParams",
    "NoiseSpec",
    "PpgMorphologyParams",
    "add_noise",
    "ecg_mcsharry",
    "ppg_dynamical",
]
