# %% import statements
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 3
import contextlib
import tempfile
from pathlib import Path
import pandas as pd

import keras
import h5py
import numpy as np
import tensorflow as tf
import helia_edge as helia
import matplotlib.pyplot as plt
import logging

from compression_kit.layers import VectorQuantizer, GumbelSoftmaxBottleneck
from compression_kit.layers.residual_vector_quantizer import ResidualVectorQuantizer
from compression_kit.trainers import VQAutoencoder, GSAutoencoder
from compression_kit.callbacks import TemperatureAnneal

# %% Constants

# File paths
project_root = Path(__file__).resolve().parent
datasets_dir = Path("/home/vscode/datasets")
job_dir = Path(tempfile.gettempdir()) / "hk-ecg-compressor"
results_root = project_root / "results"
model_file = job_dir / "model.keras"
val_file = job_dir / "val.pkl"

# Data settings
sampling_rate = 250  # 250 Hz
input_size = 5000  # 20 seconds
frame_size = 1024  # ~4.1 seconds

# Training settings
batch_size = 128  # Batch size for training
buffer_size = 10000  # How many samples are shuffled each epoch
epochs = 100  # Increase this to 100+
steps_per_epoch = 150  # # Steps per epoch (must set since ds has unknown size)
samples_per_patient = 5  # Number of samples per patient
val_metric = "loss"  # Metric to monitor for early stopping
val_mode = "min"  # Mode for early stopping min for loss, max for accuracy
val_size = 10000  # Number of samples used for validation
learning_rate = 1e-3  # Learning rate for Adam optimizer
epsilon = 0.001

# Model settings
embedding_dim = 16
latent_width = 256
temperature = 0.1
num_levels = 3
filename_suffix = f"ecg-rvq-l{num_levels}"

# Other settings
seed = 42  # Seed for reproducibility
verbose = 1  # Verbosity level
rng = np.random.default_rng(seed)

# Configure logger
logger = logging.getLogger("compression")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
# Remove the source name from the formatter to avoid duplication
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

run_results_dir = results_root / filename_suffix
os.makedirs(job_dir, exist_ok=True)
results_root.mkdir(parents=True, exist_ok=True)
run_results_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Job directory: {job_dir}")

# %% Load PTBXL dataset

ptbxl_path = datasets_dir / "ptbxl"
ptbxl_files = list(ptbxl_path.glob("*.h5"))
logger.debug(f"Found {len(ptbxl_files)} PTB-XL files.")
# Each file represents different person. Grab randomly 80% for training, 10% for validation, 10% for testing
# Shuffle the files
ptbxl_files = np.random.permutation(ptbxl_files)
train_pts = ptbxl_files[: int(len(ptbxl_files) * 0.8)]
val_pts = ptbxl_files[int(len(ptbxl_files) * 0.8) : int(len(ptbxl_files) * 0.9)]
test_pts = ptbxl_files[int(len(ptbxl_files) * 0.9) :]

# Print the number of files in each set
logger.info(f"Training files: {len(train_pts)}")
logger.info(f"Validation files: {len(val_pts)}")
logger.info(f"Testing files: {len(test_pts)}")

def load_ptbxl_data(file_paths):
    data = []
    for file_path in file_paths:
        with h5py.File(file_path, "r") as h5:
            pt_data = h5["data"][:]
            # pt_data = pt_data.reshape(-1, 12, 5000)
            data.append(pt_data)

    data = np.concatenate(data, axis=0)
    return data

train_data = load_ptbxl_data(train_pts)
val_data = load_ptbxl_data(val_pts)
test_data = load_ptbxl_data(test_pts)

train_data = train_data[:, :, np.newaxis]
val_data = val_data[:, :, np.newaxis]
test_data = test_data[:, :, np.newaxis]

# %% Augmentation and preprocessing pipeline

# nstdb = hk.datasets.nstdb.NstdbNoise(target_rate=sampling_rate)
# noises = np.hstack(
#     (nstdb.get_noise(noise_type="bw"), nstdb.get_noise(noise_type="ma"), nstdb.get_noise(noise_type="em"))
# )
# noises = noises.astype(np.float32)

# Preprocess shalll take random crop, layer norm and reshape to 2D
preprocessor = helia.layers.preprocessing.AugmentationPipeline(
    layers=[
        helia.layers.preprocessing.RandomCrop1D(duration=frame_size, name="RandomCropPreprocess"),
        helia.layers.preprocessing.LayerNormalization1D(epsilon=epsilon, name="LayerNormalization"),
    ]
)
# preprocessor = helia.layers.preprocessing.LayerNormalization1D(epsilon=epsilon, name="LayerNormalization")

augmenter = helia.layers.preprocessing.AugmentationPipeline(
    layers=[
        # helia.layers.preprocessing.RandomNoiseDistortion1D(
        #     sample_rate=sampling_rate, amplitude=(0, 1.0), frequency=(0.5, 1.5), name="BaselineWander"
        # ),
        # helia.layers.preprocessing.RandomSineWave(
        #     sample_rate=sampling_rate, amplitude=(0, 0.05), frequency=(45, 50), name="PowerlineNoise"
        # ),
        # helia.layers.preprocessing.AmplitudeWarp(
        #     sample_rate=sampling_rate, amplitude=(0.05, 0.1), frequency=(0.5, 1.5), name="AmplitudeWarp"
        # ),
        helia.layers.preprocessing.RandomGaussianNoise1D(factor=(0.01, 0.1), name="GaussianNoise"),
        # helia.layers.preprocessing.RandomBackgroundNoises1D(
        #     noises=noises, amplitude=(0.05, 0.2), num_noises=2, name="RandomBackgroundNoises"
        # ),
        # helia.layers.preprocessing.RandomCutout1D(
        #     factor=(0.01, 0.05), cutouts=2, fill_mode="constant", fill_value=0.0, name="RandomCutout"
        # ),
        # helia.layers.preprocessing.RandomCrop1D(duration=frame_size, name="RandomCrop", auto_vectorize=True),
    ],
)



# %% Apply preprocessing and augmentation

train_ds = tf.data.Dataset.from_tensor_slices(train_data)
val_ds = tf.data.Dataset.from_tensor_slices(val_data)

train_ds = (
    train_ds.shuffle(
        buffer_size=buffer_size,
        reshuffle_each_iteration=True,
    )
    .batch(
        batch_size=batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .map(
        lambda x: (
            preprocessor(x, training=True),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .map(
        lambda x: (
            keras.layers.Reshape((1, frame_size, 1), name="To2D")(augmenter(x, training=True)),
            keras.layers.Reshape((1, frame_size, 1), name="To2D")(x),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds.batch(
        batch_size=batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .map(
        lambda x: (
            preprocessor(x, training=True)
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .map(
        lambda x: (
            keras.layers.Reshape((1, frame_size, 1), name="To2D")(augmenter(x, training=True)),
            keras.layers.Reshape((1, frame_size, 1), name="To2D")(x),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .prefetch(tf.data.AUTOTUNE)
)

# Cache the validation dataset
# val_ds = val_ds.take(val_size // batch_size).cache()


def _collect_random_samples(dataset, sample_count, rng, pool_multiplier=5):
    """Gather a random subset of samples from a tf.data.Dataset."""
    pool_size = max(sample_count * pool_multiplier, sample_count)
    inputs, targets = [], []
    for batch_inputs, batch_targets in dataset:
        batch_inputs_np = batch_inputs.numpy()
        batch_targets_np = batch_targets.numpy()
        for idx in range(batch_inputs_np.shape[0]):
            inputs.append(batch_inputs_np[idx])
            targets.append(batch_targets_np[idx])
            if len(inputs) >= pool_size:
                break
        if len(inputs) >= pool_size:
            break

    if len(inputs) < sample_count:
        raise ValueError("Unable to collect the requested number of samples from the dataset.")

    selected_idx = rng.choice(len(inputs), size=sample_count, replace=False)
    sampled_inputs = np.stack([inputs[i] for i in selected_idx])
    sampled_targets = np.stack([targets[i] for i in selected_idx])
    return sampled_inputs, sampled_targets

#%% Visualize augmented samples
aug_ecg, act_ecg = next(iter(train_ds))
aug_ecg = aug_ecg.numpy()[0,0,:,0]
act_ecg = act_ecg.numpy()[0,0,:,0]
ts = np.arange(0, aug_ecg.shape[0]) / sampling_rate
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
ax[0].plot(ts, aug_ecg, lw=2, label="Augmented ECG Signal")
ax[1].plot(ts, act_ecg, lw=2, alpha=0.5, label="Actual ECG Signal")
fig.suptitle("Sample Preprocessed + Augmented ECG Signal")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amplitude")
ax[1].legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.show()
# %% Define VQ-VAE model components

# ---------- tiny building blocks ----------
def conv2d_block(x, filters, k_w=7, stride_w=1, name=None):
    x = keras.layers.Conv2D(filters, (1, k_w), strides=(1, stride_w), padding="same",
                      name=None if name is None else f"{name}_conv")(x)
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-5,
                      name=None if name is None else f"{name}_ln")(x)
    x = keras.layers.Activation("relu",
                      name=None if name is None else f"{name}_act")(x)
    return x

def depthwise2d_block(x, filters, k_w=7, stride_w=1, name=None):
    x = keras.layers.DepthwiseConv2D((1, k_w), strides=(1, stride_w), padding="same",
                      name=None if name is None else f"{name}_dw")(x)
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-5,
                      name=None if name is None else f"{name}_dw_bn")(x)
    x = keras.layers.Activation("relu",
                      name=None if name is None else f"{name}_dw_act")(x)
    x = keras.layers.Conv2D(filters, (1,1), padding="same",
                      name=None if name is None else f"{name}_pw")(x)
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-5,
                      name=None if name is None else f"{name}_pw_bn")(x)
    x = keras.layers.Activation("relu",
                      name=None if name is None else f"{name}_pw_act")(x)
    return x

def up2d_block(x, filters, k_w=7, name=None):
    x = keras.layers.UpSampling2D(size=(1, 2),
                      name=None if name is None else f"{name}_up")(x)
    x = keras.layers.Conv2D(filters, (1, k_w), padding="same",
                      name=None if name is None else f"{name}_conv")(x)
    x = keras.layers.Activation("relu",
                      name=None if name is None else f"{name}_act")(x)
    # light anti-alias
    x = keras.layers.SeparableConv2D(filters, (1, 5), padding="same",
                      name=None if name is None else f"{name}_aa")(x)
    return x

def conv1d_block(x, filters, k=7, stride=1, name=None):
    x = keras.layers.Conv1D(
        filters,
        k,
        strides=stride,
        padding="same",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-5, name=None if name is None else f"{name}_ln"
    )(x)
    x = keras.layers.Activation(
        "gelu", name=None if name is None else f"{name}_act"
    )(x)
    return x

def up1d_block(x, filters, k=7, name=None):
    x = keras.layers.UpSampling1D(
        size=2, name=None if name is None else f"{name}_up"
    )(x)
    x = keras.layers.Conv1D(
        filters,
        k,
        padding="same",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = keras.layers.Activation(
        "gelu", name=None if name is None else f"{name}_act"
    )(x)
    x = keras.layers.SeparableConv1D(
        filters,
        5,
        padding="same",
        name=None if name is None else f"{name}_aa",
    )(x)
    return x

def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

# ---------- Encoder: (B,4096,1) -> (B,1,256,D) ----------
def build_encoder_16x_2d(input_len=4096, in_ch=1, base=32, embedding_dim=16, multiplier=2):
    inp = keras.layers.Input(shape=(1, input_len, in_ch), name="ecg_in_1d")
    # x = keras.layers.Reshape((1, input_len, in_ch), name="to_2d")(inp)
    x = inp

    # 4 stages of stride 2 along width: 4096→2048→1024→512→256
    filters = base
    for i in range(0, 2):
        x = conv2d_block(x, filters, k_w=7, stride_w=2, name=f"enc_s{i+1}")
        filters = make_divisible(filters * multiplier, 8)
        print(f"Encoder stage {i+1}: shape={x.shape}, filters={filters}")

    for i in range(2, 4):
        x = depthwise2d_block(x, filters, k_w=7, stride_w=2, name=f"enc_s{i+1}")
        filters = make_divisible(filters * multiplier, 8)
        print(f"Encoder stage {i+1}: shape={x.shape}, filters={filters}")


    # project to VQ embedding dim (channels = D)
    x = keras.layers.Conv2D(embedding_dim, (1,1), padding="same", name="to_vq")(x)
    return keras.Model(inp, x, name="Encoder2D_16x")

# ---------- Decoder: (B,1,256,D) -> (B,4096,1) ----------
def build_decoder_16x_2d(output_len=4096, out_ch=1, base=32, embedding_dim=16, multiplier=2):
    z = keras.layers.Input(shape=(1, output_len // 16, embedding_dim), name="latent_2d")
    x = z

    # 4 up stages: 256→512→1024→2048→4096
    filters = make_divisible(base * (multiplier ** 3), 8)
    for i in range(4):
        x = up2d_block(x, filters, k_w=7, name=f"dec_s{i+1}")
        filters = make_divisible(filters // multiplier, 8)
        print(f"Decoder stage {i+1}: shape={x.shape}, filters={filters}")

    x = keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="head_ln")(x)
    x = keras.layers.Conv2D(out_ch, (1,1), padding="same", name="recon_2d")(x)
    out = x
    # out = keras.layers.Reshape((output_len, out_ch), name="from_2d")(x)
    return keras.Model(z, out, name="Decoder2D_16x")

def build_encoder_16x_1d(input_len=4096, in_ch=1, base=32, embedding_dim=16):
    inp = keras.layers.Input(shape=(input_len, in_ch), name="ecg_in_1d")
    x = inp

    # 4 stages of stride-2 downsampling: 4096→256
    x = conv1d_block(x, base, k=7, stride=2, name="enc1d_s0")
    x = conv1d_block(x, base * 2, k=7, stride=2, name="enc1d_s1")
    x = conv1d_block(x, base * 3, k=7, stride=2, name="enc1d_s2")
    x = conv1d_block(x, base * 4, k=7, stride=2, name="enc1d_s3")

    x = keras.layers.Conv1D(
        embedding_dim, 1, padding="same", name="enc1d_to_latent"
    )(x)
    return keras.Model(inp, x, name="Encoder1D_16x")

def build_decoder_16x_1d(output_len=4096, out_ch=1, base=32, embedding_dim=16):
    z = keras.layers.Input(
        shape=(output_len // 16, embedding_dim), name="latent_1d"
    )
    x = z

    x = up1d_block(x, base * 4, k=7, name="dec1d_s0")
    x = up1d_block(x, base * 3, k=7, name="dec1d_s1")
    x = up1d_block(x, base * 2, k=7, name="dec1d_s2")
    x = up1d_block(x, base, k=7, name="dec1d_s3")

    x = keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="dec1d_head_ln")(x)
    out = keras.layers.Conv1D(out_ch, 1, padding="same", name="dec1d_recon")(x)
    return keras.Model(z, out, name="Decoder1D_16x")

def build_vqae_2d_16x(input_len=4096, embedding_dim=16, num_embeddings=256, base=32):
    enc = build_encoder_16x_2d(input_len, 1, base, embedding_dim)
    dec = build_decoder_16x_2d(input_len, 1, base, embedding_dim)
    vq  = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, beta=0.25)

    inp = keras.layers.Input(shape=(input_len, 1), name="ecg_in")
    z   = enc(inp)                      # (B, 1, 256, D)
    zq  = vq(z)                         # quantized, same shape
    out = dec(zq)                       # (B, 4096, 1)
    model = keras.Model(inp, out, name="VQAE_2D_16x")
    return enc, vq, dec, model

def build_rvqae_2d_16x(
    input_len=4096,
    embedding_dim=16,
    num_embeddings=256,
    base=32,
    multiplier=2,
    num_levels=num_levels,
    beta=0.25,
):
    enc = build_encoder_16x_2d(input_len, 1, base, embedding_dim, multiplier)
    dec = build_decoder_16x_2d(input_len, 1, base, embedding_dim, multiplier)
    rvq = ResidualVectorQuantizer(
        num_levels=num_levels,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        beta=beta,
    )

    inp = keras.layers.Input(shape=(1, input_len, 1), name="ecg_in")
    z = enc(inp)
    zq = rvq(z)
    out = dec(zq)
    model = keras.Model(inp, out, name="RVQAE_2D_16x")
    return enc, rvq, dec, model

# ------------------------------------------------------------
# DW → ACT → PW → NORM block (NHWC). Residual only if stride==1
# and in/out channels match.
# ------------------------------------------------------------
def dw_pw_block_nhwc(
    x,
    out_channels,
    dw_kernel=3,
    stride=1,                  # downsample along width when >1
    activation="relu",
    norm="bn",                 # "bn" or "ln"
    bn_momentum=0.9,
    name="blk",
    use_skip=True,
):
    ch_in = int(x.shape[-1])

    # Depthwise (1, k) with optional stride along width
    y = keras.layers.DepthwiseConv2D(
        kernel_size=(1, dw_kernel),
        strides=(1, stride),
        padding="same",
        use_bias=False,
        name=f"{name}_dw",
    )(x)

    # Activation
    y = keras.layers.Activation(activation, name=f"{name}_act")(y)

    # Pointwise 1x1
    y = keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False,
        name=f"{name}_pw",
    )(y)

    # Normalization at the end
    if norm == "bn":
        y = keras.layers.BatchNormalization(momentum=bn_momentum, name=f"{name}_bn")(y)
    elif norm == "ln":
        y = keras.layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln")(y)

    # Identity residual only if not downsampling and channels match
    if use_skip and stride == 1 and ch_in == out_channels:
        y = keras.layers.Add(name=f"{name}_res")([x, y])

    return y


# ------------------------------------------------------------
# Encoder: (H=1, W=4096, C=in_ch) -> (H=1, W=4096 / Πstrides, C=last_channels)
# N blocks of DW→ACT→PW→NORM, downsampling via stride on DW.
# ------------------------------------------------------------
def build_encoder_nhwc(
    input_len=4096,
    in_channels=1,
    channels=(16, 16, 16, 16),   # out channels per block
    strides=(2, 2, 2, 2),        # width strides per block (product should be 16 to reach W=256)
    activation="relu",
    norm="bn",
    bn_momentum=0.9,
    name="encoder",
    latent_dim=16,
):
    assert len(channels) == len(strides), "channels and strides must be same length"

    inp = keras.layers.Input(shape=(input_len, in_channels), name=f"{name}_input")

    # Reshape to NHWC: (B, 1, input_len, in_channels)
    x = keras.layers.Reshape((1, input_len, in_channels), name=f"{name}_to_nhwc")(inp)

    for i, (c, s) in enumerate(zip(channels, strides), start=1):
        x = dw_pw_block_nhwc(
            x,
            out_channels=c,
            dw_kernel=7,
            stride=s,
            activation=activation,
            norm=norm,
            bn_momentum=bn_momentum,
            name=f"{name}_b{i}",
            use_skip=True,  # will only add skip if s==1 and ch match
        )

    # Final projection to latent_dim channels (linear)
    x = keras.layers.Conv2D(
        filters=latent_dim,
        kernel_size=(1, 1),
        padding="same",
        use_bias=True,
        name=f"{name}_head",
    )(x)

    return keras.Model(inp, x, name=name)


# ------------------------------------------------------------
# Decoder: mirrors encoder with upsampling on width, then DW→ACT→PW→NORM
# Input width * Πup_factors should equal 4096.
# ------------------------------------------------------------
def build_decoder_nhwc(
    output_len=4096,
    out_channels=1,
    channels=(16, 16, 16, 16),   # channels per block (mirror of encoder)
    up_factors=(2, 2, 2, 2),     # width upsampling per block
    activation="relu",
    norm="bn",
    bn_momentum=0.9,
    up_interpolation="nearest",  # or "bilinear"
    name="decoder",
    latent_dim=16,
):
    assert len(channels) == len(up_factors), "channels and up_factors must be same length"

    total_up = 1
    for u in up_factors:
        total_up *= u
    tokens = output_len // total_up
    assert tokens * total_up == output_len, "output_len must be divisible by product of up_factors"

    # Expect input shape from the encoder head: (1, tokens, latent_dim)
    inp = keras.layers.Input(shape=(1, tokens, latent_dim), name=f"{name}_input")
    x = inp

    for i, (c, u) in enumerate(zip(channels, up_factors), start=1):
        if u > 1:
            x = keras.layers.UpSampling2D(size=(1, u), interpolation=up_interpolation, name=f"{name}_up{i}")(x)

        # After upsampling, apply the same DW→ACT→PW→NORM block (stride=1)
        x = dw_pw_block_nhwc(
            x,
            out_channels=c,
            dw_kernel=7,
            stride=1,
            activation=activation,
            norm=norm,
            bn_momentum=bn_momentum,
            name=f"{name}_b{i}",
            use_skip=True,  # residual applies (stride==1) and ch match
        )

    # Final projection to out_channels (linear)
    out = keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(1, 1),
        padding="same",
        use_bias=True,
        name=f"{name}_head",
    )(x)

    # Reshape back to (B, output_len, out_channels)
    out = keras.layers.Reshape((output_len, out_channels), name=f"{name}_from_nhwc")(out)

    return keras.Model(inp, out, name=name)


# ------------------------------------------------------------
# Optional: end-to-end autoencoder wiring
# For your (4096 -> 256) latent width target, use strides=(2,2,2,2) and up_factors=(2,2,2,2).
# ------------------------------------------------------------
def build_autoencoder_nhwc(
    input_len=4096,
    in_channels=1,
    latent_dim=16,
    num_embeddings=256,
    num_levels=2,
    beta=0.25,
    channels=(16, 16, 16, 16),
    strides=(2, 2, 2, 2),
    activation="relu",
    norm="bn",
    bn_momentum=0.9,
    up_interpolation="nearest",
    name="ae",
):
    enc = build_encoder_nhwc(
        input_len=input_len,
        in_channels=in_channels,
        channels=channels,
        strides=strides,
        activation=activation,
        norm=norm,
        bn_momentum=bn_momentum,
        name=f"{name}_enc",
        latent_dim=latent_dim,
    )

    # Mirror decoder: same channel tuple; up_factors mirror strides to recover 4096
    dec = build_decoder_nhwc(
        output_len=input_len,
        out_channels=in_channels,
        channels=channels[::-1],                 # optional: keep channels same order or reverse
        up_factors=strides[::-1],                # upsample inversely to encoder strides
        activation=activation,
        norm=norm,
        bn_momentum=bn_momentum,
        up_interpolation=up_interpolation,
        name=f"{name}_dec",
        latent_dim=latent_dim,
    )

    rvq = ResidualVectorQuantizer(
        num_levels=num_levels,
        num_embeddings=num_embeddings,
        embedding_dim=latent_dim,
        beta=beta,
    )

    inp = keras.layers.Input(shape=(input_len, 1), name="ecg_in")
    z = enc(inp)
    zq = rvq(z)
    out = dec(zq)
    model = keras.Model(inp, out, name=name)
    return enc, rvq, dec, model


def build_gsae_2d_16x(
    input_len: int = 4096,
    embedding_dim: int = 16,
    num_embeddings: int = 256,
    base: int = 32,
    *,
    temperature: float = 1.0,
    hard: bool = True,
    kl_weight: float = 1.0,
    input_is_logits: bool = False,
    name: str = "GSAE_2D_16x",
    use_wrapper: bool = True,   # return GSAutoencoder by default
):
    """
    Build a height=1, 2D Gumbel-Softmax autoencoder with 16× temporal down/upsampling.

    Returns:
      enc (keras.Model), gs (GumbelSoftmaxBottleneck), dec (keras.Model),
      model (GSAutoencoder if use_wrapper=True; otherwise a plain keras.Model)
    """
    # Encoder: produce D channels normally; or K logits if input_is_logits=True
    enc = build_encoder_16x_2d(
        input_len=input_len,
        in_ch=1,
        base=base,
        embedding_dim=(num_embeddings if input_is_logits else embedding_dim),
    )
    dec = build_decoder_16x_2d(
        output_len=input_len,
        out_ch=1,
        base=base,
        embedding_dim=embedding_dim,
    )

    gs = GumbelSoftmaxBottleneck(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        temperature=temperature,
        hard=hard,
        input_is_logits=input_is_logits,   # True if encoder outputs K logits
        kl_weight=kl_weight,
    )

    if use_wrapper:
        # Return the convenience wrapper so compile(extra_losses=..., extra_metrics=...) works.
        model = GSAutoencoder(encoder=enc, gs=gs, decoder=dec, name=name)
    else:
        # Plain functional model (no extra_losses/extra_metrics support in compile)
        inp = keras.layers.Input(shape=(input_len, 1), name="ecg_in")
        z   = enc(inp)     # (B, 1, input_len/16, D) or (..., K) if input_is_logits
        zq  = gs(z)        # (B, 1, input_len/16, embedding_dim)
        out = dec(zq)      # (B, input_len, 1)
        model = keras.Model(inp, out, name=name)

    return enc, gs, dec, model

def build_gsae_1d_16x(
    input_len: int = 4096,
    embedding_dim: int = 16,
    num_embeddings: int = 256,
    base: int = 32,
    *,
    temperature: float = 1.0,
    hard: bool = True,
    kl_weight: float = 1.0,
    input_is_logits: bool = False,
    name: str = "GSAE_1D_16x",
    use_wrapper: bool = True,   # return GSAutoencoder by default
):
    """
    Build a height=1, 1D Gumbel-Softmax autoencoder with 16× temporal down/upsampling.

    Returns:
      enc (keras.Model), gs (GumbelSoftmaxBottleneck), dec (keras.Model),
      model (GSAutoencoder if use_wrapper=True; otherwise a plain keras.Model)
    """
    # Encoder: produce D channels normally; or K logits if input_is_logits=True
    enc = build_encoder_16x_1d(
        input_len=input_len,
        in_ch=1,
        base=base,
        embedding_dim=(num_embeddings if input_is_logits else embedding_dim),
    )
    dec = build_decoder_16x_1d(
        output_len=input_len,
        out_ch=1,
        base=base,
        embedding_dim=embedding_dim,
    )

    gs = GumbelSoftmaxBottleneck(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        temperature=temperature,
        hard=hard,
        input_is_logits=input_is_logits,   # True if encoder outputs K logits
        kl_weight=kl_weight,
    )

    if use_wrapper:
        # Return the convenience wrapper so compile(extra_losses=..., extra_metrics=...) works.
        model = GSAutoencoder(encoder=enc, gs=gs, decoder=dec, name=name)
    else:
        # Plain functional model (no extra_losses/extra_metrics support in compile)
        inp = keras.layers.Input(shape=(input_len, 1), name="ecg_in")
        z   = enc(inp)     # (B, 1, input_len/16, D) or (..., K) if input_is_logits
        zq  = gs(z)        # (B, 1, input_len/16, embedding_dim)
        out = dec(zq)      # (B, input_len, 1)
        model = keras.Model(inp, out, name=name)

    return enc, gs, dec, model

# %%

# enc, vq, dec, model = build_autoencoder_nhwc(
#     input_len=frame_size,
#     in_channels=1,
#     latent_dim=embedding_dim,
#     num_embeddings=latent_width,
#     num_levels=2,
#     beta=0.25,
#     channels=(32, 64, 96, 128),
#     strides=(2, 2, 2, 2),
#     activation="relu",
#     norm="bn",
#     bn_momentum=0.9,
#     up_interpolation="nearest",
#     name="AE_NHWC_16x",
# )
# enc.summary()
# dec.summary()
# model.summary()

# %% Build VQ-VAE model
enc, vq, dec, model = build_rvqae_2d_16x(
    input_len=frame_size,
    embedding_dim=embedding_dim,
    num_embeddings=latent_width,
    base=32,
    multiplier=1.25,
    num_levels=2,
)
# enc, gs, dec, model = build_gsae_1d_16x(
#     input_len=frame_size,
#     embedding_dim=embedding_dim,
#     num_embeddings=latent_width,
#     base=8,
#     input_is_logits=False,
#     use_wrapper=True,
# )
enc.summary()
dec.summary()
model.summary()

# %% Compile the model

def get_scheduler():
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=steps_per_epoch * epochs,
    )


optimizer = keras.optimizers.Adam(get_scheduler())
loss = helia.losses.simclr.SimCLRLoss(temperature=temperature)

def d_loss(y, yhat):
    dy  = y[:, 1:, :] - y[:, :-1, :]
    dyh = yhat[:, 1:, :] - yhat[:, :-1, :]
    return keras.ops.mean(keras.ops.abs(dy - dyh)) * 0.2

def prd_percent(y, yhat):
    mse = keras.ops.mean((y - yhat) ** 2)
    return 100.0 * keras.ops.sqrt(mse)

metrics = [
    keras.metrics.MeanSquaredError(name="mse"),
    keras.metrics.CosineSimilarity(name="cos", axis=-2),
]

model_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor=f"val_{val_metric}",
        patience=max(int(0.25 * epochs), 1),
        mode=val_mode,
        restore_best_weights=True,
        verbose=verbose - 1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(model_file), monitor=f"val_{val_metric}", save_best_only=True, mode=val_mode, verbose=verbose - 1
    ),
    keras.callbacks.CSVLogger(job_dir / "history.csv"),
]
if helia.utils.env_flag("TENSORBOARD"):
    model_callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir=job_dir,
            write_steps_per_second=True,
        )
    )

# model.compile(
#     optimizer=keras.optimizers.Adam(1e-3),
#     loss=keras.losses.MeanSquaredError(),
#     metrics=metrics,
# )

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.MeanSquaredError(),
    metrics=metrics,
    # extra_losses=[d_loss],
    # extra_metrics=[prd_percent],
)

# %% Train the model

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    verbose=verbose,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=model_callbacks,
)

# %% Persist artifacts

model_save_path = run_results_dir / "model.keras"
model.save(model_save_path)
logger.info(f"Saved trained model to {model_save_path}")

sample_inputs, sample_targets = _collect_random_samples(val_ds, sample_count=20, rng=rng)
recon_samples = model.predict(sample_inputs, verbose=0)

rows = []
for sample_id, (target, recon) in enumerate(zip(sample_targets, recon_samples)):
    target_flat = target.reshape(-1)
    recon_flat = recon.reshape(-1)
    for time_index, (orig_val, recon_val) in enumerate(zip(target_flat, recon_flat)):
        rows.append(
            {
                "sample_id": sample_id,
                "time_index": time_index,
                "original": float(orig_val),
                "reconstructed": float(recon_val),
            }
        )

recon_df = pd.DataFrame(rows)
recon_csv_path = run_results_dir / f"reconstructions_{filename_suffix}.csv"
recon_df.to_csv(recon_csv_path, index=False)
logger.info(f"Saved reconstruction CSV to {recon_csv_path}")

# %% Visualize training history

fig, _ = helia.plotting.plot_history_metrics(
    history.history,
    metrics=["loss", "rvq_bits_per_index_sum"],
    title="Training History",
    stack=True,
    figsize=(9, 5),
)
fig.tight_layout()
fig.show()

# %%
# Lets plot some reconstructions from the test set
test_ecg, orig_ecg = next(iter(val_ds))
test_ecg = test_ecg.numpy()
orig_ecg = orig_ecg.numpy()
# Pass random noise through the model
# test_ecg = np.random.normal(size=(2, frame_size, 1)).astype(np.float32)
# test_ecg = np.zeros((2, frame_size, 1), dtype=np.float32)
recon_ecg = model.predict(test_ecg)
sample_idx = 0
ts = np.arange(0, test_ecg[sample_idx].squeeze().shape[0]) / sampling_rate

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_facecolor("#f7f8fa")
ax.plot(
    ts,
    test_ecg[sample_idx].squeeze(),
    color="#1f77b4",
    lw=2.5,
    label="Original",
)
ax.plot(
    ts,
    recon_ecg[sample_idx].squeeze(),
    color="#ff7f0e",
    lw=2,
    label="Reconstruction @ 32x",
)
# ax.plot(ts, orig_ecg[sample_idx].squeeze(), color="#2ca02c", lw=1.5, label="Actual")
ax.set_title("CompressionKIT: ECG Reconstruction", fontsize=14)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend(frameon=False, loc="lower right")
ax.grid(color="#d7dde5", linestyle="-", linewidth=0.8, alpha=0.5)
for spine in ax.spines.values():
    spine.set_alpha(0.2)
fig.tight_layout()

# %% Load the history from CSV
history_df = pd.read_csv(job_dir / "history.csv")
history_df.head()
# columns = epoch,cos,loss,mse,rvq_bits_per_index_sum,rvq_l1_bits_per_index,rvq_l1_perplexity,rvq_l1_usage,rvq_l2_bits_per_index,rvq_l2_perplexity,rvq_l2_usage,rvq_perplexity_mean,rvq_usage_mean,val_cos,val_loss,val_mse,val_rvq_bits_per_index_sum,val_rvq_l1_bits_per_index,val_rvq_l1_perplexity,val_rvq_l1_usage,val_rvq_l2_bits_per_index,val_rvq_l2_perplexity,val_rvq_l2_usage,val_rvq_perplexity_mean,val_rvq_usage_mean
# plot training/validation loss curves for mse, usage, and perplexity
fig, axs = plt.subplots(3, 1, figsize=(9, 12))
# MSE
axs[0].plot(history_df["epoch"], history_df["loss"], label="Train LOSS")
axs[0].plot(history_df["epoch"], history_df["val_loss"], label="Val LOSS")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("LOSS")
# log scale
axs[0].set_yscale("log")
axs[0].legend()
# Usage
axs[1].plot(history_df["epoch"], history_df["rvq_l1_usage"], label="Train Level 1 Usage")
axs[1].plot(history_df["epoch"], history_df["val_rvq_l1_usage"], label="Val Level 1 Usage")
axs[1].plot(history_df["epoch"], history_df["rvq_l2_usage"], label="Train Level 2 Usage")
axs[1].plot(history_df["epoch"], history_df["val_rvq_l2_usage"], label="Val Level 2 Usage")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Usage")
axs[1].legend()
# Perplexity
axs[2].plot(history_df["epoch"], history_df["rvq_l1_perplexity"], label="Train Level 1 Perplexity")
axs[2].plot(history_df["epoch"], history_df["val_rvq_l1_perplexity"], label="Val Level 1 Perplexity")
axs[2].plot(history_df["epoch"], history_df["rvq_l2_perplexity"], label="Train Level 2 Perplexity")
axs[2].plot(history_df["epoch"], history_df["val_rvq_l2_perplexity"], label="Val Level 2 Perplexity")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Perplexity")
axs[2].legend()
fig.suptitle("Training History Details")
fig.tight_layout()
fig.show()

# %% Export the encoder
# Create rep dataset (numpy of shape (N, 1, frame_size, 1)) from val_ds TF dataset which has shape (batch_size, 1, frame_size, 1)
# The len(val_ds) will be num_batches, so we need to multiply by batch_size to get N
rep_dataset = np.zeros((len(val_ds) * batch_size, 1, frame_size, 1), dtype=np.float32)
for i, (x, _) in enumerate(val_ds):
    rep_dataset[i * batch_size:(i + 1) * batch_size] = x.numpy()

converter = helia.converters.tflite.TfLiteKerasConverter(model=enc)

# Redirect stdout and stderr to devnull since TFLite converter is very verbose
with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        tflite_content = converter.convert(
            test_x=rep_dataset, quantization="INT8", io_type="int8", mode="KERAS", strict=False, verbose=verbose
        )
# %% Save the TFLite model
converter.export(tflite_path=job_dir / "encoder.tflite")

converter.export_header(
    header_path=job_dir / "encoder.h",
    name="encoder",
)

# %%
