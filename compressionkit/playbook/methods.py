"""Seed cards for the method catalog.

Importing this module registers the explicit (non-golden) cards: classical
DSP codecs that run with no trained weights, plus *experimental* and
*planned* entries that advertise a technique that already has building
blocks in the tree (or is on the roadmap) even before it ships as a golden.

Keeping these here — rather than inside :mod:`catalog` — preserves the rule
that the catalog owns no algorithm logic; the builders below just assemble
codecs from the existing ``evaluation`` building blocks.
"""

from __future__ import annotations

from compressionkit.evaluation.codec import (
    BayesShrinkSpihtCodec,
    Codec,
    FilterSpihtCodec,
    IdentityCodec,
    LearnedShrinkSpihtCodec,
    SpihtAcCodec,
)
from compressionkit.pipeline import BandpassPreprocessor, ZNormPreprocessor, build_dwt_deadzone_codec
from compressionkit.playbook.catalog import (
    Faithfulness,
    MethodCard,
    Status,
    Tier,
    register_method,
)
from compressionkit.playbook.lanes import Lane


def _build_identity(*, sample_rate: int, frame_size: int, target_cr: float, modality: str, **_: object) -> Codec:
    return IdentityCodec(name="identity", modality=modality, sample_rate=sample_rate, frame_size=frame_size)



def _build_spiht(*, sample_rate: int, frame_size: int, target_cr: float, modality: str, **_: object) -> Codec:
    return SpihtAcCodec(
        name=f"spiht_{target_cr:g}x",
        modality=modality,
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
    )


def _build_filter_spiht(*, sample_rate: int, frame_size: int, target_cr: float, modality: str, **_: object) -> Codec:
    high_hz = 40.0 if modality == "ecg" else 8.0
    return FilterSpihtCodec(
        name=f"filter_spiht_{target_cr:g}x",
        modality=modality,
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        low_hz=0.5,
        high_hz=high_hz,
    )


def _build_bayes_shrink_spiht(
    *, sample_rate: int, frame_size: int, target_cr: float, modality: str, **_: object
) -> Codec:
    return BayesShrinkSpihtCodec(
        name=f"bayes_shrink_spiht_{target_cr:g}x",
        modality=modality,
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
    )


def _build_dwt_deadzone_deflate(
    *, sample_rate: int, frame_size: int, target_cr: float, modality: str, **_: object
) -> Codec:
    return build_dwt_deadzone_codec(
        modality=modality,
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        preprocess=ZNormPreprocessor(),
        name=f"dwt_deadzone_deflate_{target_cr:g}x",
    )


def _build_bandpass_dwt_deadzone_deflate(
    *, sample_rate: int, frame_size: int, target_cr: float, modality: str, **_: object
) -> Codec:
    high_hz = 40.0 if modality == "ecg" else 8.0
    return build_dwt_deadzone_codec(
        modality=modality,
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        preprocess=BandpassPreprocessor(sample_rate=sample_rate, low_hz=0.5, high_hz=high_hz),
        name=f"bandpass_dwt_deadzone_deflate_{target_cr:g}x",
    )


# Default location for the shipped ECG wavelet-gain denoiser (git-ignored
# results/). Override by retraining via experiments/scripts/train_wavelet_denoiser_ecg.py.
_ECG_DENOISER_PATH = "results/wavelet_denoiser_ecg_noharm/gain_model.keras"
# Higher-capacity v2 denoiser (gated-residual, wide-augmentation training);
# trained via experiments/scripts/train_wavelet_denoiser_v2_ecg.py.
_ECG_DENOISER_V2_PATH = "results/wavelet_denoiser_ecg_v2/gain_model.keras"
# PPG wavelet-gain denoiser, trained via experiments/scripts/train_wavelet_denoiser_ppg.py.
# v2: amplitude-preserving + no-harm loss with motion augmentation (protects the
# clean pulse shape and the single-channel SpO2 AC proxy). Baseline preserved at
# results/wavelet_denoiser_ppg/ for provenance.
_PPG_DENOISER_PATH = "results/wavelet_denoiser_ppg_v2_amp_noharm/gain_model.keras"


def _denoiser_path(modality: str) -> str:
    """Default trained gain-model path for the learned-shrink lane by modality."""
    return _PPG_DENOISER_PATH if modality == "ppg" else _ECG_DENOISER_PATH


def _build_learned_denoise_dwt_deadzone_deflate(
    *, sample_rate: int, frame_size: int, target_cr: float, modality: str, **_: object
) -> Codec:
    from compressionkit.pipeline import load_wavelet_gain_preprocessor

    pre = load_wavelet_gain_preprocessor(_ECG_DENOISER_PATH, frame_size=frame_size)
    return build_dwt_deadzone_codec(
        modality=modality,
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        preprocess=pre,
        name=f"learned_denoise_dwt_deadzone_deflate_{target_cr:g}x",
    )


def _build_learned_shrink_spiht(
    *, sample_rate: int, frame_size: int, target_cr: float, modality: str, **_: object
) -> Codec:
    from compressionkit.pipeline import load_wavelet_gain_preprocessor

    # Reuse the same wavelet-gain denoiser, but on the strong SPIHT backend so the
    # only difference from the bandpass golden is the denoise front-end. The
    # trained gain model is modality-specific (ECG vs PPG).
    pre = load_wavelet_gain_preprocessor(_denoiser_path(modality), frame_size=frame_size)
    return LearnedShrinkSpihtCodec(
        name=f"learned_shrink_spiht_{target_cr:g}x",
        modality=modality,
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        coeff_denoiser=pre.coeff_denoiser,
    )


def _build_learned_shrink_spiht_v2(
    *, sample_rate: int, frame_size: int, target_cr: float, modality: str, **_: object
) -> Codec:
    from compressionkit.pipeline import load_wavelet_gain_preprocessor

    # Higher-capacity gated-residual denoiser trained on the wide artifact
    # distribution, on the same SPIHT backend as the bandpass golden.
    pre = load_wavelet_gain_preprocessor(_ECG_DENOISER_V2_PATH, frame_size=frame_size)
    return LearnedShrinkSpihtCodec(
        name=f"learned_shrink_spiht_v2_{target_cr:g}x",
        modality=modality,
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        coeff_denoiser=pre.coeff_denoiser,
    )



# ---------------------------------------------------------------------------
# Runnable classical cards (no trained weights required)
# ---------------------------------------------------------------------------

register_method(
    MethodCard(
        id="identity",
        display_name="Identity (passthrough)",
        lane=Lane.FAITHFUL,
        family="reference",
        faithfulness=Faithfulness.FAITHFUL,
        summary="Lossless passthrough; sanity-check anchor (CR = 1).",
        status=Status.SHIPPED,
        tier=Tier.BASELINE,
        rationale="Lossless anchor (CR=1) to sanity-check the harness and bound the metrics.",
        target_crs=(1,),
        edge_notes="Trivial; no compute.",
        builder=_build_identity,
    )
)

register_method(
    MethodCard(
        id="spiht",
        display_name="SPIHT (bior4.4 + AC)",
        lane=Lane.FAITHFUL,
        family="classical-dsp",
        faithfulness=Faithfulness.FAITHFUL,
        summary="Wavelet zerotree coder; preserves morphology and noise alike.",
        status=Status.SHIPPED,
        tier=Tier.ROBUST,
        rationale="Best bit-allocation of the classical coders (significance-ordered bitplanes); "
        "faithful-lane champion on real ECG.",
        target_crs=(2, 4, 8, 16, 32),
        edge_notes="Fixed-point DWT + zerotree scan; malloc-free, embedded-portable.",
        preprocess="znorm",
        transform="dwt(bior4.4)",
        encoder_stage="spiht (fused)",
        entropy="arithmetic (fused)",
        builder=_build_spiht,
    )
)

register_method(
    MethodCard(
        id="filter_spiht",
        display_name="Bandpass + SPIHT",
        lane=Lane.CLEAN,
        family="classical-dsp",
        faithfulness=Faithfulness.DENOISE,
        summary="Zero-phase Butterworth bandpass, then SPIHT. Denoises without inventing structure.",
        status=Status.SHIPPED,
        tier=Tier.ROBUST,
        rationale="Bandpass denoise + SPIHT: the conventional clean-lane golden to beat — strong "
        "truth fidelity, no hallucination risk (linear front-end). Novel denoisers must beat this.",
        target_crs=(2, 4, 8, 16, 32),
        edge_notes="Butterworth SOS (fixed coeffs) + SPIHT; embedded-portable.",
        preprocess="bandpass",
        transform="dwt(bior4.4)",
        encoder_stage="spiht (fused)",
        entropy="arithmetic (fused)",
        builder=_build_filter_spiht,
    )
)

register_method(
    MethodCard(
        id="bayes_shrink_spiht",
        display_name="BayesShrink + SPIHT",
        lane=Lane.CLEAN,
        family="classical-dsp",
        faithfulness=Faithfulness.DENOISE,
        summary="Wavelet-domain BayesShrink denoise, then SPIHT.",
        status=Status.EXPERIMENTAL,
        tier=Tier.EXPERIMENTAL,
        rationale="Adaptive wavelet-threshold denoiser kept for comparison against the bandpass front-end.",
        target_crs=(2, 4, 8, 16, 32),
        edge_notes="Per-band threshold + SPIHT; embedded-portable.",
        preprocess="bayes-shrink",
        transform="dwt(bior4.4)",
        encoder_stage="spiht (fused)",
        entropy="arithmetic (fused)",
        builder=_build_bayes_shrink_spiht,
    )
)

register_method(
    MethodCard(
        id="dwt_deadzone_deflate",
        display_name="DWT + dead-zone quant + deflate",
        lane=Lane.FAITHFUL,
        family="classical-dsp",
        faithfulness=Faithfulness.FAITHFUL,
        summary="Cleanly separable all-DSP pipeline: znorm \u2192 DWT \u2192 dead-zone quantize \u2192 "
        "deflate. First-class, swappable entropy slot.",
        status=Status.EXPERIMENTAL,        tier=Tier.BASELINE,
        rationale="Separable all-DSP reference: exposes the swappable encoder/entropy slots that "
        "SPIHT fuses. Substrate for experiments, not yet competitive with SPIHT.",        target_crs=(2, 4, 8, 16),
        edge_notes="Uniform quantizer + DEFLATE (portable C); malloc-free.",
        preprocess="znorm",
        transform="dwt(bior4.4)",
        encoder_stage="dead-zone quant",
        entropy="deflate",
        builder=_build_dwt_deadzone_deflate,
    )
)

register_method(
    MethodCard(
        id="bandpass_dwt_deadzone_deflate",
        display_name="Bandpass + DWT + dead-zone quant + deflate",
        lane=Lane.CLEAN,
        family="classical-dsp",
        faithfulness=Faithfulness.DENOISE,
        summary="Denoising variant of the separable DSP pipeline: bandpass \u2192 DWT \u2192 dead-zone "
        "quantize \u2192 deflate.",
        status=Status.EXPERIMENTAL,
        target_crs=(2, 4, 8, 16),
        edge_notes="Butterworth SOS + uniform quantizer + DEFLATE; embedded-portable.",
        preprocess="bandpass",
        transform="dwt(bior4.4)",
        encoder_stage="dead-zone quant",
        entropy="deflate",
        builder=_build_bandpass_dwt_deadzone_deflate,
    )
)

register_method(
    MethodCard(
        id="learned_denoise_dwt_deadzone_deflate",
        display_name="Learned denoise + DWT + dead-zone quant + deflate",
        lane=Lane.CLEAN,
        family="hybrid-ai-dsp",
        faithfulness=Faithfulness.DENOISE,
        summary="Phase 2: AI denoiser (wavelet-gain net) in the preprocess slot, then the same "
        "all-DSP DWT \u2192 dead-zone \u2192 deflate chain. Tests learned vs bandpass denoising.",
        status=Status.EXPERIMENTAL,
        tier=Tier.EXPERIMENTAL,
        rationale="Secondary AI-denoise probe on the separable pipeline; viable but encoder is weaker "
        "than SPIHT. See learned_shrink_spiht for the apples-to-apples contender.",
        modality=("ecg",),
        sample_rate=256,
        window_samples=512,
        target_crs=(2, 4, 8, 16),
        edge_notes="Small wavelet-gain net (INT8) + DSP chain. Needs trained denoiser weights.",
        preprocess="learned_denoise",
        transform="dwt(bior4.4)",
        encoder_stage="dead-zone quant",
        entropy="deflate",
        builder=_build_learned_denoise_dwt_deadzone_deflate,
    )
)

register_method(
    MethodCard(
        id="learned_shrink_spiht",
        display_name="AI denoise + SPIHT",
        lane=Lane.CLEAN,
        family="hybrid-ai-dsp",
        faithfulness=Faithfulness.DENOISE,
        summary="Learned wavelet-gain denoise on the SAME SPIHT backend as the bandpass golden. "
        "The only change vs the golden is the denoise front-end — a fair AI-vs-DSP test.",
        status=Status.EXPERIMENTAL,
        tier=Tier.EXPERIMENTAL,
        rationale="Viable contender to the bandpass+SPIHT golden: identical SPIHT backend, AI denoise "
        "front-end. Kept as a first-class challenger — not yet proven to beat conventional filtering.",
        modality=("ecg",),
        sample_rate=256,
        window_samples=512,
        target_crs=(2, 4, 8, 16, 32),
        edge_notes="Small wavelet-gain net (INT8) + SPIHT; embedded-portable. Needs trained denoiser weights.",
        preprocess="learned_denoise",
        transform="dwt(bior4.4)",
        encoder_stage="spiht (fused)",
        entropy="arithmetic (fused)",
        builder=_build_learned_shrink_spiht,
    )
)

register_method(
    MethodCard(
        id="learned_shrink_spiht_v2",
        display_name="AI denoise v2 (gated-residual) + SPIHT",
        lane=Lane.CLEAN,
        family="hybrid-ai-dsp",
        faithfulness=Faithfulness.DENOISE,
        summary="Higher-capacity gated-residual denoiser (attenuate + bounded correct), trained on a "
        "wide artifact distribution, on the same SPIHT backend as the bandpass golden.",
        status=Status.EXPERIMENTAL,
        tier=Tier.EXPERIMENTAL,
        rationale="Strongest AI-denoise challenger to the bandpass+SPIHT golden: more capacity and a "
        "richer mode of operation, trained across the full artifact family distribution.",
        modality=("ecg",),
        sample_rate=256,
        window_samples=512,
        target_crs=(2, 4, 8, 16, 32),
        edge_notes="~13k-param gated-residual net (INT8) + SPIHT; embedded-portable. Needs v2 denoiser weights.",
        preprocess="learned_denoise_v2",
        transform="dwt(bior4.4)",
        encoder_stage="spiht (fused)",
        entropy="arithmetic (fused)",
        builder=_build_learned_shrink_spiht_v2,
    )
)

# ---------------------------------------------------------------------------
# Experimental / planned cards (building blocks exist or are on the roadmap)
# ---------------------------------------------------------------------------

register_method(
    MethodCard(
        id="rvq_token_prior",
        display_name="RVQ + learned token entropy prior",
        lane=Lane.COMPACT_LEARNED,
        family="learned-entropy",
        faithfulness=Faithfulness.PERCEPTUAL,
        summary="RVQ codec whose token stream is arithmetic-coded under a learned causal prior "
        "(two-stage). Pushes CR beyond the uniform bits/token ceiling.",
        status=Status.EXPERIMENTAL,
        tier=Tier.EXPERIMENTAL,
        rationale="Novel, highest-value direction: RVQ tokens under a learned entropy prior "
        "(EnCodec/SoundStream-style). Building blocks already exist (runtime.two_stage).",
        target_crs=(8, 16, 32),
        edge_notes="Small causal prior (INT8, LiteRT) + arithmetic coder; see runtime.two_stage.",
    )
)

register_method(
    MethodCard(
        id="ssm_decoder",
        display_name="Compact encoder + SSM decoder",
        lane=Lane.LONG_CONTEXT,
        family="learned-asymmetric",
        faithfulness=Faithfulness.PERCEPTUAL,
        summary="Compact conv encoder, heavier diagonal-SSM (S4D) decoder that mixes time at every "
        "scale. Decoder pulls its weight on phone/cloud.",
        status=Status.EXPERIMENTAL,
        rationale="Long-context champion candidate: asymmetric compact-encoder / heavy-decoder; "
        "deferred until the trustworthy scorecard lands.",
        target_crs=(8, 16, 32),
        edge_notes="S4D recurrent decoder; fixed state size, streaming-friendly. See models.build_decoder_2d_ssm.",
    )
)

register_method(
    MethodCard(
        id="linattn_decoder",
        display_name="Compact encoder + linear-attention decoder",
        lane=Lane.LONG_CONTEXT,
        family="learned-asymmetric",
        faithfulness=Faithfulness.PERCEPTUAL,
        summary="Compact conv encoder, heavy CNN + linear-attention transformer decoder. "
        "Linear (kernel-feature) attention is matmul-only and INT8-friendly.",
        status=Status.PLANNED,
        rationale="Long-context alternative to SSM; deferred (biggest lift, needs scorecard first).",
        target_crs=(8, 16, 32),
        edge_notes="Linear attention avoids softmax; targets LiteRT INT8. To be built.",
    )
)

register_method(
    MethodCard(
        id="predictive_entropy",
        display_name="Predictive coding + entropy coder",
        lane=Lane.FAITHFUL,
        family="classical-dsp",
        faithfulness=Faithfulness.FAITHFUL,
        summary="Short/long-term predictor with entropy-coded residual; the honest near-lossless "
        "anchor (FLAC-style) we currently lack.",
        status=Status.PLANNED,
        rationale="Planned faithful-lane near-lossless anchor; embedded-ideal (a few MACs + rANS).",
        target_crs=(2, 4),
        edge_notes="A few MACs + rANS; ideal embedded fit. To be built.",
    )
)


__all__: list[str] = []
