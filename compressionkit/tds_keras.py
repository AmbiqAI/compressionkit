"""
Keras3 implementation of the EMG2Pose TDS network.

The architecture mirrors ``config/network/tds.yaml`` from
https://github.com/facebookresearch/emg2pose and follows the definitions in
``emg2pose/networks.py``. The model operates on sequences shaped
``(batch, timesteps, channels)`` and uses the TensorFlow backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import tensorflow as tf
from keras import Input, Model, layers


# Default configuration mirroring config/network/tds.yaml
DEFAULT_CONV_BLOCK_CFGS = (
    dict(in_channels=16, out_channels=256, kernel_size=11, stride=5),
    dict(in_channels=256, out_channels=256, kernel_size=5, stride=2),
)

DEFAULT_TDS_STAGE_CFGS = (
    dict(
        in_channels=256,
        in_conv_kernel_width=17,
        in_conv_stride=4,
        num_blocks=2,
        channels=16,
        feature_width=16,
        kernel_width=9,
        out_channels=None,
    ),
    dict(
        in_channels=256,
        in_conv_kernel_width=9,
        in_conv_stride=2,
        num_blocks=2,
        channels=16,
        feature_width=16,
        kernel_width=5,
        out_channels=64,
    ),
)


class Conv1DBlock(layers.Layer):
    """1D convolution followed by optional normalization, ReLU, and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: str = "layer",
        dropout: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.in_channels = in_channels
        self.norm_type = norm_type
        self.conv = layers.Conv1D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
        )
        if norm_type == "batch":
            self.batch_norm = layers.BatchNormalization(axis=-1)
        else:
            self.batch_norm = None
        if norm_type == "layer":
            self.layer_norm = layers.LayerNormalization(axis=-1)
        else:
            self.layer_norm = None
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape: tf.TensorShape) -> None:
        if input_shape[-1] and input_shape[-1] != self.in_channels:
            raise ValueError(
                f"Conv1DBlock expected {self.in_channels} channels, "
                f"received {input_shape[-1]}"
            )
        super().build(input_shape)

    def call(self, inputs, training: bool | None = None):
        x = self.conv(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)
        x = self.relu(x)
        x = self.dropout(x, training=training)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class TDSConv2DBlock(layers.Layer):
    """Temporal depth separable convolutional block."""

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        if kernel_width % 2 == 0:
            raise ValueError("kernel_width must be odd.")
        super().__init__()
        self.channels = channels
        self.width = width
        self.kernel_width = kernel_width
        self.conv2d = layers.Conv2D(
            filters=channels,
            kernel_size=(1, kernel_width),
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
            use_bias=True,
        )
        self.relu = layers.ReLU()
        self.layer_norm = layers.LayerNormalization(axis=-1)

    def call(self, inputs, training: bool | None = None):
        # inputs: (B, T, channels * width)
        input_bt = tf.transpose(inputs, perm=[0, 2, 1])  # (B, C, T)
        total_features = tf.shape(input_bt)[1]
        expected_features = self.channels * self.width
        tf.debugging.assert_equal(
            total_features,
            expected_features,
            message="channels * width must equal feature dimension.",
        )

        batch_size = tf.shape(input_bt)[0]
        time_steps = tf.shape(input_bt)[2]

        x = tf.reshape(input_bt, [batch_size, self.channels, self.width, time_steps])
        x = tf.transpose(x, perm=[0, 2, 3, 1])  # (B, width, T, channels)
        x = self.conv2d(x)
        x = self.relu(x)
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # (B, channels, width, T_out)
        x = tf.reshape(x, [batch_size, expected_features, -1])  # (B, C, T_out)

        t_out = tf.shape(x)[-1]
        t_in = tf.shape(input_bt)[-1]
        tf.debugging.assert_less_equal(
            t_out, t_in, message="TDS block output length cannot exceed input length."
        )
        start = t_in - t_out
        skip = tf.slice(input_bt, [0, 0, start], [-1, -1, -1])
        x = x + skip

        x = tf.transpose(x, perm=[0, 2, 1])  # (B, T_out, C)
        x = self.layer_norm(x)
        return x


class TDSFullyConnectedBlock(layers.Layer):
    """Two-layer feed-forward residual block operating per timestep."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.dense1 = layers.Dense(num_features, activation="relu")
        self.dense2 = layers.Dense(num_features)
        self.layer_norm = layers.LayerNormalization(axis=-1)

    def call(self, inputs, training: bool | None = None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = x + inputs
        return self.layer_norm(x)


class TDSConvEncoder(layers.Layer):
    """Stack of alternating TDSConv2DBlock and TDSFullyConnectedBlock."""

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int],
        kernel_width: int,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.kernel_width = kernel_width
        self.blocks: list[layers.Layer] = []
        for channels in block_channels:
            if num_features % channels != 0:
                raise ValueError(
                    f"channels {channels} must evenly divide num_features {num_features}"
                )
            feature_width = num_features // channels
            self.blocks.append(TDSConv2DBlock(channels, feature_width, kernel_width))
            self.blocks.append(TDSFullyConnectedBlock(num_features))

    def call(self, inputs, training: bool | None = None):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x


class TdsStage(layers.Layer):
    """Stage of Conv1DBlock -> TDSConvEncoder -> optional projection."""

    def __init__(
        self,
        in_channels: int,
        in_conv_kernel_width: int,
        in_conv_stride: int,
        num_blocks: int,
        channels: int,
        feature_width: int,
        kernel_width: int,
        out_channels: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        target_channels = channels * feature_width
        self.layers_list: list[layers.Layer] = []
        if in_conv_kernel_width > 0:
            self.layers_list.append(
                Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=target_channels,
                    kernel_size=in_conv_kernel_width,
                    stride=in_conv_stride,
                    name=f"{name}_conv1d" if name else None,
                )
            )
        elif in_channels != target_channels:
            raise ValueError(
                "in_channels must equal channels * feature_width "
                "when in_conv_kernel_width <= 0."
            )

        self.layers_list.append(
            TDSConvEncoder(
                num_features=target_channels,
                block_channels=[channels] * num_blocks,
                kernel_width=kernel_width,
            )
        )
        self.projection = (
            layers.Dense(out_channels, name=f"{name}_projection")
            if out_channels is not None
            else None
        )

    def call(self, inputs, training: bool | None = None):
        x = inputs
        for block in self.layers_list:
            x = block(x, training=training)
        if self.projection is not None:
            x = self.projection(x)
        return x


class TdsNetwork(Model):
    """Keras implementation of the EMG2Pose TDS network."""

    def __init__(
        self,
        conv_blocks: Iterable[Conv1DBlock],
        tds_stages: Iterable[TdsStage],
        input_shape: tuple[int | None, int],
    ) -> None:
        inputs = Input(shape=input_shape, name="emg_input")
        x = inputs
        for block in conv_blocks:
            x = block(x)
        for stage in tds_stages:
            x = stage(x)
        super().__init__(inputs=inputs, outputs=x, name="tds_network")


def build_default_tds_network(
    timesteps: int | None = None,
    *,
    conv_block_cfgs: Sequence[dict] = DEFAULT_CONV_BLOCK_CFGS,
    tds_stage_cfgs: Sequence[dict] = DEFAULT_TDS_STAGE_CFGS,
) -> Model:
    """
    Build the exact network described in config/network/tds.yaml.

    Args:
        timesteps: Optional fixed number of timesteps for the input. ``None``
            keeps the time dimension dynamic.
    """

    conv_blocks = [
        Conv1DBlock(**cfg, name=f"conv_block_{idx}") for idx, cfg in enumerate(conv_block_cfgs)
    ]
    tds_stages = [
        TdsStage(**cfg, name=f"tds_stage_{idx}") for idx, cfg in enumerate(tds_stage_cfgs)
    ]

    input_shape = (timesteps, conv_block_cfgs[0]["in_channels"])
    return TdsNetwork(conv_blocks, tds_stages, input_shape=input_shape)


def _compute_left_context(
    conv_block_cfgs: Sequence[dict], tds_stage_cfgs: Sequence[dict]
) -> int:
    """Replicate the PyTorch helper that measures required left context."""

    left = 0
    cumulative_stride = 1
    for cfg in conv_block_cfgs:
        left += (cfg["kernel_size"] - 1) * cumulative_stride
        cumulative_stride *= cfg["stride"]

    for stage_cfg in tds_stage_cfgs:
        left += (stage_cfg["in_conv_kernel_width"] - 1) * cumulative_stride
        cumulative_stride *= stage_cfg["in_conv_stride"]
        for _ in range(stage_cfg["num_blocks"]):
            left += (stage_cfg["kernel_width"] - 1) * cumulative_stride

    return left


DEFAULT_MIN_TIMESTEPS = _compute_left_context(
    DEFAULT_CONV_BLOCK_CFGS, DEFAULT_TDS_STAGE_CFGS
) + 1
REFERENCE_WINDOW_LENGTH = 2_000  # https://github.com/facebookresearch/emg2pose config


def _representative_dataset(
    timesteps: int, num_samples: int, num_channels: int
) -> tf.lite.RepresentativeDataset:
    """Generate random samples for post-training quantization."""

    def dataset():
        for _ in range(num_samples):
            sample = tf.random.normal([1, timesteps, num_channels], dtype=tf.float32)
            yield [sample]

    return dataset


def convert_tds_to_int8_tflite(
    output_path: str | Path,
    timesteps: int,
    representative_samples: int = 128,
) -> Path:
    """
    Build the default TDS network, perform int8 quantization, and export TFLite.

    Args:
        output_path: Destination for the flatbuffer.
        timesteps: Fixed temporal dimension for the exported model.
        representative_samples: Number of random samples for calibration.
    """

    if timesteps < DEFAULT_MIN_TIMESTEPS:
        raise ValueError(
            "timesteps must be at least "
            f"{DEFAULT_MIN_TIMESTEPS} to keep the TDS receptive field valid. "
            f"Try timesteps >= {REFERENCE_WINDOW_LENGTH} (reference repo default)."
        )
    if representative_samples <= 0:
        raise ValueError("representative_samples must be positive.")
    output_path = Path(output_path)
    model = build_default_tds_network(timesteps=timesteps)
    print(model.summary())
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = _representative_dataset(
        timesteps=timesteps,
        num_samples=representative_samples,
        num_channels=model.input_shape[-1],
    )

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the EMG2Pose TDS network and export an int8 TFLite model."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=REFERENCE_WINDOW_LENGTH,
        help=(
            "Fixed number of timesteps for the exported model. "
            f"Must be >= {DEFAULT_MIN_TIMESTEPS}. Default matches the reference repo (2000)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/tds_network_int8.tflite"),
        help="Destination path for the TFLite flatbuffer.",
    )
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=128,
        help="Number of random samples for quantization calibration.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = convert_tds_to_int8_tflite(
        output_path=args.output,
        timesteps=args.timesteps,
        representative_samples=args.representative_samples,
    )
    print(f"Saved int8 TDS network to {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "Conv1DBlock",
    "TDSConv2DBlock",
    "TDSFullyConnectedBlock",
    "TDSConvEncoder",
    "TdsStage",
    "TdsNetwork",
    "build_default_tds_network",
    "convert_tds_to_int8_tflite",
]
