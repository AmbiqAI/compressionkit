"""DWT-domain RVQ autoencoder: wavelet front-end + learned codec + wavelet back-end.

Architecture:
    Signal → DWT (fixed) → Encoder (learned) → RVQ → Decoder (learned) → iDWT (fixed) → Reconstruction

The encoder/decoder operate in wavelet domain where the signal is sparse and
decorrelated, reducing the autoencoder reconstruction floor. The DWT/iDWT
provide perfect-reconstruction transform boundaries.
"""

from __future__ import annotations

import keras
from helia_edge.trainers import VQAutoencoder

from compressionkit.layers.dwt import DWT1D, InverseSubbandNorm1D, SubbandNorm1D, iDWT1D
from compressionkit.models.rvq_autoencoder import build_rvq_autoencoder


@keras.saving.register_keras_serializable(package="compressionkit")
class DWTVQAutoencoder(VQAutoencoder):
    """VQAutoencoder that wraps encoder/decoder with DWT/iDWT transforms.

    The composite forward pass is:
        x → DWT → encoder → VQ → decoder → iDWT → reconstruction

    Loss is computed in time domain (after iDWT), ensuring fair comparison
    with time-domain baselines.
    """

    def __init__(
        self,
        encoder: keras.Model,
        vq,
        decoder: keras.Model,
        *,
        dwt_layer: DWT1D,
        idwt_layer: iDWT1D,
        subband_norm: SubbandNorm1D | None = None,
        inv_subband_norm: InverseSubbandNorm1D | None = None,
        **kwargs,
    ):
        super().__init__(encoder=encoder, vq=vq, decoder=decoder, **kwargs)
        self.dwt_layer = dwt_layer
        self.idwt_layer = idwt_layer
        self.subband_norm = subband_norm
        self.inv_subband_norm = inv_subband_norm

    def call(self, x, training=False, return_indices=False):
        # Transform to wavelet domain
        z_wavelet = self.dwt_layer(x)
        # Normalize subbands to similar scale
        if self.subband_norm is not None:
            z_wavelet = self.subband_norm(z_wavelet)
        # Encode + VQ
        z = self.encoder(z_wavelet, training=training)
        if callable(self.vq):
            if return_indices:
                zq, indices = self.vq(z, training=training, return_indices=True)
            else:
                zq = self.vq(z, training=training)
        else:
            zq = z
            indices = None
        # Decode in wavelet domain (normalized)
        y_wavelet = self.decoder(zq, training=training)
        # Denormalize back to original scale
        if self.inv_subband_norm is not None:
            y_wavelet = self.inv_subband_norm(y_wavelet)
        # Transform back to time domain
        y = self.idwt_layer(y_wavelet)
        if return_indices:
            return y, indices
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "dwt_layer": keras.saving.serialize_keras_object(self.dwt_layer),
            "idwt_layer": keras.saving.serialize_keras_object(self.idwt_layer),
            "subband_norm": keras.saving.serialize_keras_object(self.subband_norm) if self.subband_norm else None,
            "inv_subband_norm": keras.saving.serialize_keras_object(self.inv_subband_norm) if self.inv_subband_norm else None,
        })
        return config


def build_dwt_rvq_autoencoder(
    frame_size: int = 512,
    *,
    wavelet: str = "bior4.4",
    dwt_levels: int = 6,
    subband_norm: bool = True,
    embedding_dim: int = 16,
    latent_width: int = 256,
    num_levels: int = 2,
    num_stages: int = 2,
    base_filters: int = 48,
    multiplier: float = 1.25,
    beta: float = 0.25,
    use_ema: bool = True,
    ema_decay: float = 0.99,
    encoder_block_norm: str = "batch",
    encoder_head_norm: str = "none",
    decoder_block_norm: str = "none",
    decoder_head_norm: str = "layer",
    use_residual: bool = False,
    decoder_activation: str = "relu",
    encoder_blocks_per_stage: int = 1,
    revive_dead_codes: bool = False,
    revive_threshold: float = 0.03,
    kmeans_init: bool = False,
) -> tuple[keras.Model, object, keras.Model, DWTVQAutoencoder]:
    """Build DWT-domain RVQ autoencoder.

    The encoder/decoder operate on wavelet coefficients (same frame_size).
    The DWT/iDWT layers handle the transform boundaries.

    Returns:
        ``(encoder, rvq, decoder, model)`` where model is DWTVQAutoencoder.
    """
    # Build the inner encoder/decoder/RVQ (operates in wavelet domain)
    encoder, rvq, decoder, _inner_model = build_rvq_autoencoder(
        frame_size=frame_size,
        embedding_dim=embedding_dim,
        latent_width=latent_width,
        num_levels=num_levels,
        num_stages=num_stages,
        base_filters=base_filters,
        multiplier=multiplier,
        beta=beta,
        use_ema=use_ema,
        ema_decay=ema_decay,
        encoder_block_norm=encoder_block_norm,
        encoder_head_norm=encoder_head_norm,
        decoder_block_norm=decoder_block_norm,
        decoder_head_norm=decoder_head_norm,
        use_residual=use_residual,
        decoder_activation=decoder_activation,
        encoder_blocks_per_stage=encoder_blocks_per_stage,
        revive_dead_codes=revive_dead_codes,
        revive_threshold=revive_threshold,
        kmeans_init=kmeans_init,
    )

    # Build DWT/iDWT layers
    dwt_layer = DWT1D(wavelet=wavelet, levels=dwt_levels, signal_len=frame_size, name="dwt_front")
    idwt_layer = iDWT1D(wavelet=wavelet, levels=dwt_levels, signal_len=frame_size, name="idwt_back")

    # Build optional subband normalization
    norm_layer = None
    inv_norm_layer = None
    if subband_norm:
        norm_layer = SubbandNorm1D(wavelet=wavelet, levels=dwt_levels, signal_len=frame_size, name="subband_norm")
        inv_norm_layer = InverseSubbandNorm1D(wavelet=wavelet, levels=dwt_levels, signal_len=frame_size, name="inv_subband_norm")

    # Composite model
    downsample_factor = 2**num_stages
    model = DWTVQAutoencoder(
        encoder=encoder,
        vq=rvq,
        decoder=decoder,
        dwt_layer=dwt_layer,
        idwt_layer=idwt_layer,
        subband_norm=norm_layer,
        inv_subband_norm=inv_norm_layer,
        name=f"DWT_RVQAE_ds{downsample_factor}",
    )

    return encoder, rvq, decoder, model
