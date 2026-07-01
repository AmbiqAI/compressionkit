"""Tests for the compression playbook catalog and run engine."""

from __future__ import annotations

import numpy as np
import pytest

from compressionkit.playbook import (
    Faithfulness,
    Lane,
    Status,
    get_method,
    list_methods,
    run_method_on_signal,
)
from compressionkit.playbook.run import load_signal


@pytest.fixture
def ecg_signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(2560) / 256.0
    sig = 0.1 * np.sin(2 * np.pi * 0.3 * t)
    for k in range(int(t[-1] * 1.2)):
        sig += np.exp(-((t - k / 1.2) ** 2) / (2 * 0.02**2))
    sig += 0.05 * rng.standard_normal(t.size)
    return sig.astype(np.float32)


def test_catalog_lists_explicit_and_golden_methods() -> None:
    cards = list_methods()
    ids = {c.id for c in cards}
    # explicit classical cards
    assert {"spiht", "filter_spiht", "identity"} <= ids
    # golden hook surfaces registered goldens
    assert any(c.golden_id is not None for c in cards)
    assert any(c.status is Status.SHIPPED for c in cards)


def test_lane_and_status_filters() -> None:
    clean = list_methods(lane=Lane.CLEAN)
    assert clean and all(c.lane is Lane.CLEAN for c in clean)
    planned = list_methods(status=Status.PLANNED)
    assert all(c.status is Status.PLANNED for c in planned)


def test_get_method_and_faithfulness() -> None:
    assert get_method("spiht").faithfulness is Faithfulness.FAITHFUL
    assert get_method("filter_spiht").faithfulness is Faithfulness.DENOISE
    with pytest.raises(KeyError):
        get_method("does_not_exist")


def test_run_dual_reference_scorecard(ecg_signal: np.ndarray) -> None:
    card = get_method("spiht")
    res = run_method_on_signal(card, ecg_signal, modality="ecg", target_cr=8.0)
    assert res.true_cr == pytest.approx(8.0, rel=0.05)
    assert np.isfinite(res.faithful_prd)
    assert np.isfinite(res.proxy_prd)
    assert res.reconstruction.shape == ecg_signal.shape


def test_filter_is_cleaner_spiht_is_more_faithful(ecg_signal: np.ndarray) -> None:
    spiht = run_method_on_signal(get_method("spiht"), ecg_signal, modality="ecg", target_cr=8.0)
    filt = run_method_on_signal(get_method("filter_spiht"), ecg_signal, modality="ecg", target_cr=8.0)
    # SPIHT preserves the noisy input more faithfully; the bandpass filter
    # tracks the clean proxy better. This is the faithful-vs-clean axis.
    assert spiht.faithful_prd < filt.faithful_prd
    assert filt.proxy_prd < spiht.proxy_prd


def test_run_rejects_unrunnable_method(ecg_signal: np.ndarray) -> None:
    card = get_method("ssm_decoder")  # experimental, no builder
    assert not card.runnable
    with pytest.raises(ValueError, match="trained weights"):
        run_method_on_signal(card, ecg_signal, modality="ecg", target_cr=8.0)


def test_load_signal_round_trip(tmp_path) -> None:
    arr = np.arange(100, dtype=np.float32)
    path = tmp_path / "sig.npy"
    np.save(path, arr)
    loaded = load_signal(str(path))
    np.testing.assert_array_equal(loaded, arr)


def test_tiers_present_and_sorted_within_lane() -> None:
    from compressionkit.playbook import Lane, Tier
    from compressionkit.playbook.catalog import TIER_ORDER

    faithful = list_methods(lane=Lane.FAITHFUL)
    tiers = [c.tier for c in faithful]
    # at least one baseline and one robust in the faithful lane
    assert Tier.BASELINE in tiers
    assert Tier.ROBUST in tiers
    # sorted baseline -> robust -> experimental within the lane
    orders = [TIER_ORDER[t] for t in tiers]
    assert orders == sorted(orders)


def test_robust_methods_have_rationale() -> None:
    from compressionkit.playbook import Tier

    for card in list_methods():
        if card.tier is Tier.ROBUST:
            assert card.rationale, f"robust method {card.id} must justify itself"


def test_autocorr_structure_detects_periodicity() -> None:
    from compressionkit.playbook.benchmark import autocorr_structure

    t = np.arange(512) / 256.0
    sine = np.sin(2 * np.pi * 1.2 * t).astype(np.float32)  # 1.2 Hz ~ 72 bpm
    noise = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    assert autocorr_structure(sine, 256.0) > 0.5
    assert autocorr_structure(noise, 256.0) < 0.5


def test_run_benchmark_triple_metric() -> None:
    from compressionkit.playbook.benchmark import run_benchmark

    results = run_benchmark(["spiht"], n_windows=4, family="mains", severity=0.5)
    assert len(results) == 1
    r = results[0]
    assert np.isfinite(r.truth_prd) and r.truth_prd > 0
    assert np.isfinite(r.faithful_prd) and r.faithful_prd >= 0
    assert r.imprint >= 0.0  # DSP coder must not hallucinate
    assert r.true_cr == pytest.approx(8.0, rel=0.1)
