# %% import statements
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 3
import contextlib
import tempfile
from pathlib import Path
import keras
import h5py
import numpy as np
import tensorflow as tf
import neuralspot_edge as nse
import matplotlib.pyplot as plt
import logging

from compression_kit.layers import VectorQuantizer, GumbelSoftmaxBottleneck
from compression_kit.layers.residual_vector_quantizer import ResidualVectorQuantizer
from compression_kit.trainers import VQAutoencoder, GSAutoencoder
from compression_kit.callbacks import TemperatureAnneal

# %% Constants

# File paths
datasets_dir = Path("/home/vscode/datasets")
job_dir = Path(tempfile.gettempdir()) / "hk-ecg-compressor"
model_file = job_dir / "model.keras"
val_file = job_dir / "val.pkl"

# Data settings
sampling_rate = 500  # 500 Hz
input_size = 5000  # 10 seconds
frame_size = 4096  # 5 seconds

# Training settings
batch_size = 256  # Batch size for training
buffer_size = 10000  # How many samples are shuffled each epoch
epochs = 250  # Increase this to 100+
steps_per_epoch = 250  # # Steps per epoch (must set since ds has unknown size)
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

# Other settings
seed = 42  # Seed for reproducibility
verbose = 1  # Verbosity level

# Configure logger
logger = logging.getLogger("compression")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
# Remove the source name from the formatter to avoid duplication
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

os.makedirs(job_dir, exist_ok=True)
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

preprocessor = nse.layers.preprocessing.LayerNormalization1D(epsilon=epsilon, name="LayerNormalization")

augmenter = nse.layers.preprocessing.AugmentationPipeline(
    layers=[
        # nse.layers.preprocessing.RandomNoiseDistortion1D(
        #     sample_rate=sampling_rate, amplitude=(0, 1.0), frequency=(0.5, 1.5), name="BaselineWander"
        # ),
        # nse.layers.preprocessing.RandomSineWave(
        #     sample_rate=sampling_rate, amplitude=(0, 0.05), frequency=(45, 50), name="PowerlineNoise"
        # ),
        # nse.layers.preprocessing.AmplitudeWarp(
        #     sample_rate=sampling_rate, amplitude=(0.9, 1.1), frequency=(0.5, 1.5), name="AmplitudeWarp"
        # ),
        # nse.layers.preprocessing.RandomGaussianNoise1D(factor=(0.0001, 0.001), name="GaussianNoise"),
        # nse.layers.preprocessing.RandomBackgroundNoises1D(
        #     noises=noises, amplitude=(0.05, 0.2), num_noises=2, name="RandomBackgroundNoises"
        # ),
        # nse.layers.preprocessing.RandomCutout1D(
        #     factor=(0.01, 0.05), cutouts=2, fill_mode="constant", fill_value=0.0, name="RandomCutout"
        # ),
        nse.layers.preprocessing.RandomCrop1D(duration=frame_size, name="RandomCrop", auto_vectorize=True),
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
            augmenter(preprocessor(x), training=True),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .map(
        lambda x: (x, x),
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
            augmenter(preprocessor(x), training=True),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .map(
        lambda x: (x, x),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .prefetch(tf.data.AUTOTUNE)
)

# Cache the validation dataset
# val_ds = val_ds.take(val_size // batch_size).cache()

#%% Visualize augmented samples
sample_ecg = next(iter(train_ds))[0].numpy()[0]
ts = np.arange(0, sample_ecg.shape[0]) / sampling_rate
fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.plot(ts, sample_ecg, lw=2)
fig.suptitle("Sample Preprocessed + Augmented ECG Signal")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
fig.tight_layout()
fig.show()
# %% Define VQ-VAE model components

# ---------- tiny building blocks ----------
def conv2d_block(x, filters, k_w=7, stride_w=1, name=None):
    x = keras.layers.Conv2D(filters, (1, k_w), strides=(1, stride_w), padding="same",
                      name=None if name is None else f"{name}_conv")(x)
    x = keras.layers.LayerNormalization(axis=-1, epsilon=1e-5,
                      name=None if name is None else f"{name}_ln")(x)
    x = keras.layers.Activation("gelu",
                      name=None if name is None else f"{name}_act")(x)
    return x

def up2d_block(x, filters, k_w=7, name=None):
    x = keras.layers.UpSampling2D(size=(1, 2),
                      name=None if name is None else f"{name}_up")(x)
    x = keras.layers.Conv2D(filters, (1, k_w), padding="same",
                      name=None if name is None else f"{name}_conv")(x)
    x = keras.layers.Activation("gelu",
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

# ---------- Encoder: (B,4096,1) -> (B,1,256,D) ----------
def build_encoder_16x_2d(input_len=4096, in_ch=1, base=32, embedding_dim=16):
    inp = keras.layers.Input(shape=(input_len, in_ch), name="ecg_in_1d")
    x = keras.layers.Reshape((1, input_len, in_ch), name="to_2d")(inp)

    # 4 stages of stride 2 along width: 4096→2048→1024→512→256
    x = conv2d_block(x, base,   k_w=7, stride_w=2, name="enc_s0")
    x = conv2d_block(x, base*2, k_w=7, stride_w=2, name="enc_s1")
    x = conv2d_block(x, base*3, k_w=7, stride_w=2, name="enc_s2")
    x = conv2d_block(x, base*4, k_w=7, stride_w=2, name="enc_s3")

    # project to VQ embedding dim (channels = D)
    x = keras.layers.Conv2D(embedding_dim, (1,1), padding="same", name="to_vq")(x)
    return keras.Model(inp, x, name="Encoder2D_16x")

# ---------- Decoder: (B,1,256,D) -> (B,4096,1) ----------
def build_decoder_16x_2d(output_len=4096, out_ch=1, base=32, embedding_dim=16):
    z = keras.layers.Input(shape=(1, output_len // 16, embedding_dim), name="latent_2d")
    x = z

    # 4 up stages: 256→512→1024→2048→4096
    x = up2d_block(x, base*4, k_w=7, name="dec_s0")
    x = up2d_block(x, base*3, k_w=7, name="dec_s1")
    x = up2d_block(x, base*2, k_w=7, name="dec_s2")
    x = up2d_block(x, base,   k_w=7, name="dec_s3")

    x = keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="head_ln")(x)
    x = keras.layers.Conv2D(out_ch, (1,1), padding="same", name="recon_2d")(x)
    out = keras.layers.Reshape((output_len, out_ch), name="from_2d")(x)
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
    num_levels=2,
    beta=0.25,
):
    enc = build_encoder_16x_2d(input_len, 1, base, embedding_dim)
    dec = build_decoder_16x_2d(input_len, 1, base, embedding_dim)
    rvq = ResidualVectorQuantizer(
        num_levels=num_levels,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        beta=beta,
    )

    inp = keras.layers.Input(shape=(input_len, 1), name="ecg_in")
    z = enc(inp)
    zq = rvq(z)
    out = dec(zq)
    model = keras.Model(inp, out, name="RVQAE_2D_16x")
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

# %% Build VQ-VAE model
enc, vq, dec, model = build_rvqae_2d_16x(
    input_len=frame_size,
    embedding_dim=embedding_dim,
    num_embeddings=latent_width,
    base=32
)
# enc, gs, dec, model = build_gsae_1d_16x(
#     input_len=frame_size,
#     embedding_dim=embedding_dim,
#     num_embeddings=latent_width,
#     base=8,
#     input_is_logits=False,
#     use_wrapper=True,
# )
model.summary()

# %% Compile the model

def get_scheduler():
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=steps_per_epoch * epochs,
    )


optimizer = keras.optimizers.Adam(get_scheduler())
loss = nse.losses.simclr.SimCLRLoss(temperature=temperature)

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
if nse.utils.env_flag("TENSORBOARD"):
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

# %%
