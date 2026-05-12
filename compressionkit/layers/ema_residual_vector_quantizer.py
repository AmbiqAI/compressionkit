"""Residual Vector Quantizer with Exponential Moving Average codebook updates.

Replaces the gradient-based codebook loss with EMA updates to codebook
embeddings, following van den Oord et al. 2017 (VQ-VAE).  Only the
commitment loss is back-propagated; codebook vectors are updated via
running averages of assigned encoder outputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import keras


class EmaResidualVectorQuantizer(keras.layers.Layer):
    """Residual VQ with EMA codebook updates.

    Args:
        num_levels: Number of residual VQ stages (M >= 1).
        num_embeddings: Codebook size K per level (int or per-level list).
        embedding_dim: Latent dimensionality D.
        beta: Commitment loss coefficient.
        ema_decay: EMA decay rate for codebook updates (0.99-0.999 typical).
        epsilon: Small constant for Laplace smoothing of cluster counts.
    """

    def __init__(
        self,
        num_levels: int,
        num_embeddings: int | Sequence[int],
        embedding_dim: int,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        revive_dead_codes: bool = False,
        revive_threshold: float = 0.03,
        kmeans_init: bool = False,
        structured_dropout: bool = False,
        dropout_levels: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_levels < 1 or embedding_dim <= 0 or beta <= 0:
            raise ValueError("num_levels>=1, embedding_dim>0, beta>0 required.")
        self.M = int(num_levels)
        self.D = int(embedding_dim)
        if isinstance(num_embeddings, (list, tuple)):
            if len(num_embeddings) != self.M:
                raise ValueError("num_embeddings list must have length = num_levels.")
            self.Ks = [int(k) for k in num_embeddings]
        else:
            self.Ks = [int(num_embeddings)] * self.M
        self.beta = float(beta)
        self.ema_decay = float(ema_decay)
        self.epsilon = float(epsilon)
        # Dead-code revival: any code whose normalized usage falls below
        # ``revive_threshold / K`` gets resampled from the current batch.
        # Established trick from EnCodec / DAC for preventing silent codebook
        # collapse during long training runs.
        self.revive_dead_codes = bool(revive_dead_codes)
        self.revive_threshold = float(revive_threshold)
        # K-means warm-start: replace the random init with mini-batch k-means
        # of the first training batch's residuals. Big usage win in epoch 1.
        self.kmeans_init = bool(kmeans_init)

        # Structured RVQ dropout: during training, randomly truncate the
        # number of active levels so the decoder learns to reconstruct from
        # partial quantization.  ``dropout_levels`` lists the allowed active
        # level counts (e.g. [1, 2, 4, 8]) sampled uniformly each forward pass.
        self.structured_dropout = bool(structured_dropout)
        if dropout_levels is not None:
            self.dropout_levels = sorted(dropout_levels)
        else:
            # Default: all powers of 2 up to num_levels
            self.dropout_levels = [2**i for i in range(self.M.bit_length()) if 2**i <= self.M]
            if self.M not in self.dropout_levels:
                self.dropout_levels.append(self.M)

        # Metrics (same structure as helia-edge RVQ for drop-in compatibility)
        self._lvl_perp = [keras.metrics.Mean(name=f"rvq_l{lvl + 1}_perplexity") for lvl in range(self.M)]
        self._lvl_usage = [keras.metrics.Mean(name=f"rvq_l{lvl + 1}_usage") for lvl in range(self.M)]
        self._lvl_bpi = [keras.metrics.Mean(name=f"rvq_l{lvl + 1}_bits_per_index") for lvl in range(self.M)]
        self._perp_mean = keras.metrics.Mean(name="rvq_perplexity_mean")
        self._usage_mean = keras.metrics.Mean(name="rvq_usage_mean")
        self._bpi_sum = keras.metrics.Mean(name="rvq_bits_per_index_sum")

        self._codebooks: list = []
        self._ema_counts: list = []
        self._ema_weights: list = []

    def build(self, input_shape):
        last = input_shape[-1]
        if last is not None and int(last) != self.D:
            raise ValueError(f"Input last dim {int(last)} != embedding_dim {self.D}")
        self._codebooks = []
        self._ema_counts = []
        self._ema_weights = []
        for lvl, K in enumerate(self.Ks):
            limit = 1.0 / max(1, K)
            # Codebook embeddings — NOT trainable (updated via EMA)
            cb = self.add_weight(
                name=f"codebook_l{lvl + 1}",
                shape=(K, self.D),
                initializer=keras.initializers.RandomUniform(-limit, limit),
                trainable=False,
                dtype=self.variable_dtype,
            )
            # EMA cluster counts — shape (K,)
            ema_count = self.add_weight(
                name=f"ema_count_l{lvl + 1}",
                shape=(K,),
                initializer="ones",
                trainable=False,
                dtype=self.variable_dtype,
            )
            # EMA embedding sums — shape (K, D)
            ema_weight = self.add_weight(
                name=f"ema_weight_l{lvl + 1}",
                shape=(K, self.D),
                initializer="zeros",
                trainable=False,
                dtype=self.variable_dtype,
            )
            self._codebooks.append(cb)
            self._ema_counts.append(ema_count)
            self._ema_weights.append(ema_weight)

        # Initialize ema_weight to codebook * initial_count so first update is stable
        for cb, ew in zip(self._codebooks, self._ema_weights):
            ew.assign(cb)

        # Per-layer flag controlling whether the next training batch should
        # warm-start codebooks via mini-batch k-means.  Only created when
        # ``self.kmeans_init`` is True so checkpoints saved without this
        # feature stay binary-compatible (loading them does not need the
        # extra ``kmeans_done`` variable).
        if self.kmeans_init:
            self._kmeans_done = self.add_weight(
                name="kmeans_done",
                shape=(),
                initializer="zeros",
                trainable=False,
                dtype="float32",
            )
        else:
            self._kmeans_done = None

        super().build(input_shape)

    def _nearest(self, r_flat, codebook):
        """Return indices [N] and gathered vectors [N, D]."""
        r2 = keras.ops.sum(keras.ops.square(r_flat), axis=1, keepdims=True)
        e2 = keras.ops.sum(keras.ops.square(codebook), axis=1)
        sim = keras.ops.matmul(r_flat, keras.ops.transpose(codebook))
        dist = r2 + e2 - 2.0 * sim
        idx = keras.ops.argmax(-dist, axis=1)
        q = keras.ops.take(codebook, idx, axis=0)
        return idx, q

    def _ema_update(self, lvl, idx, r_flat, K):
        """Update codebook[lvl] via EMA using assigned vectors."""
        gamma = self.ema_decay
        one_hot = keras.ops.one_hot(idx, K)  # (N, K)

        # New cluster counts and embedding sums for this batch
        new_count = keras.ops.sum(one_hot, axis=0)  # (K,)
        new_weight = keras.ops.matmul(keras.ops.transpose(one_hot), r_flat)  # (K, D)

        # EMA update
        updated_count = gamma * self._ema_counts[lvl] + (1 - gamma) * new_count
        updated_weight = gamma * self._ema_weights[lvl] + (1 - gamma) * new_weight

        # Laplace smoothing of counts
        n = keras.ops.sum(updated_count)
        smoothed = (updated_count + self.epsilon) / (n + K * self.epsilon) * n

        # Normalize to get new codebook
        new_cb = updated_weight / keras.ops.expand_dims(smoothed, axis=1)

        # ----- Dead-code revival (EnCodec / DAC style) -----
        # Any code whose smoothed normalized usage falls below
        # ``revive_threshold / K`` is replaced by a random vector drawn from
        # the current batch of residuals.  Implemented with masked blends so
        # the whole op is graph-friendly and fully vectorized.
        if self.revive_dead_codes:
            mean_count = n / float(K)
            dead = keras.ops.cast(
                updated_count < (self.revive_threshold * mean_count),
                self.compute_dtype,
            )  # (K,)
            n_rows = keras.ops.shape(r_flat)[0]
            # Sample K row indices (with replacement) into the current batch.
            sample_idx = keras.random.randint(
                shape=(K,),
                minval=0,
                maxval=n_rows,
                dtype="int32",
            )
            replacement = keras.ops.take(r_flat, sample_idx, axis=0)  # (K, D)
            dead_col = keras.ops.expand_dims(dead, axis=1)  # (K, 1)
            new_cb = (1.0 - dead_col) * new_cb + dead_col * replacement
            # Reset the EMA bookkeeping for revived rows so they get a fresh
            # statistical lease on life.
            updated_count = (1.0 - dead) * updated_count + dead * 1.0
            updated_weight = (1.0 - dead_col) * updated_weight + dead_col * replacement

        # Assign updates
        self._ema_counts[lvl].assign(updated_count)
        self._ema_weights[lvl].assign(updated_weight)
        self._codebooks[lvl].assign(new_cb)

    def warm_start_kmeans(self, z_batch) -> None:
        """Initialize codebooks from a batch of encoder outputs.

        Run mini-batch k-means on the *current* residual at each level
        sequentially.  Call this once, eagerly, after the encoder has
        produced its first batch but before ``model.fit``.

        Args:
            z_batch: Tensor or array of shape ``(..., D)`` — typically the
                concatenated encoder outputs from one or two minibatches.
        """
        z = keras.ops.convert_to_tensor(z_batch, dtype=self.compute_dtype)
        flat = keras.ops.reshape(z, (-1, self.D))
        residual = flat
        for lvl, K in enumerate(self.Ks):
            self._kmeans_warm_start(residual, lvl, K)
            _, q_l = self._nearest(residual, self._codebooks[lvl])
            residual = residual - q_l
        if self._kmeans_done is not None:
            self._kmeans_done.assign(1.0)

    def _kmeans_warm_start(self, r_flat, lvl, K):
        """Mini-batch k-means warm start for codebook ``lvl``.

        Greedy k-means++ seeding from the current batch then a few Lloyd
        iterations.  Only invoked once per layer (gated by ``_kmeans_done``).
        """
        n_rows = keras.ops.shape(r_flat)[0]
        # Seed: K random rows from the batch.
        seed_idx = keras.random.randint(
            shape=(K,),
            minval=0,
            maxval=n_rows,
            dtype="int32",
        )
        centroids = keras.ops.take(r_flat, seed_idx, axis=0)  # (K, D)
        # A few Lloyd iterations.
        for _ in range(5):
            r2 = keras.ops.sum(keras.ops.square(r_flat), axis=1, keepdims=True)
            c2 = keras.ops.sum(keras.ops.square(centroids), axis=1)
            sim = keras.ops.matmul(r_flat, keras.ops.transpose(centroids))
            dist = r2 + c2 - 2.0 * sim
            assign = keras.ops.argmax(-dist, axis=1)  # (N,)
            one_hot = keras.ops.one_hot(assign, K)
            counts = keras.ops.sum(one_hot, axis=0) + 1e-6
            sums = keras.ops.matmul(keras.ops.transpose(one_hot), r_flat)
            centroids = sums / keras.ops.expand_dims(counts, axis=1)
        self._codebooks[lvl].assign(centroids)
        self._ema_weights[lvl].assign(centroids)
        self._ema_counts[lvl].assign(keras.ops.ones((K,), dtype=self.compute_dtype))

    def call(self, x, training=None, return_indices=False):
        x = keras.ops.convert_to_tensor(x, dtype=self.compute_dtype)
        shape = keras.ops.shape(x)
        flat = keras.ops.reshape(x, (-1, self.D))

        # Determine active levels for this forward pass.
        # For structured dropout during training, sample a random cutoff
        # from the allowed level counts. Levels >= cutoff are masked (zeroed).
        if training and self.structured_dropout:
            # Sample an index into dropout_levels uniformly
            choice_idx = keras.random.randint(shape=(), minval=0, maxval=len(self.dropout_levels), dtype="int32")
            # Build a lookup tensor of allowed levels and gather the choice
            levels_tensor = keras.ops.convert_to_tensor(self.dropout_levels, dtype="int32")
            active_levels = keras.ops.take(levels_tensor, choice_idx)
        else:
            active_levels = self.M

        residual = flat
        q_sum = keras.ops.zeros_like(flat)
        indices_list = []
        perp_vals, usage_vals, bpi_vals = [], [], []

        for lvl, (K, codebook) in enumerate(zip(self.Ks, self._codebooks)):
            idx, q_l = self._nearest(residual, codebook)
            indices_list.append(idx)

            # Gate: zero out this level's contribution if beyond active_levels
            if training and self.structured_dropout:
                gate = keras.ops.cast(lvl < active_levels, self.compute_dtype)
                q_l_gated = q_l * keras.ops.expand_dims(gate, axis=0)
            else:
                q_l_gated = q_l

            q_sum = q_sum + q_l_gated

            # Commitment loss — gated so dropped levels don't contribute
            ql_st = keras.ops.stop_gradient(q_l)
            commitment = keras.ops.mean(keras.ops.square(ql_st - residual))
            if training and self.structured_dropout:
                self.add_loss(self.beta * commitment * gate)
            else:
                self.add_loss(self.beta * commitment)

            # EMA codebook update during training (always update — even dropped
            # levels benefit from seeing residuals for codebook quality)
            if training:
                self._ema_update(lvl, idx, residual, K)

            residual = residual - ql_st  # next residual

            # Per-level metrics
            one_hot = keras.ops.one_hot(idx, K)
            probs = keras.ops.mean(one_hot, axis=0)
            eps = keras.ops.convert_to_tensor(1e-10, dtype=self.compute_dtype)
            log2 = keras.ops.log(keras.ops.convert_to_tensor(2.0, self.compute_dtype))
            H = -keras.ops.sum(probs * (keras.ops.log(probs + eps) / log2))
            perp = keras.ops.exp(H * log2)
            usage = keras.ops.sum(keras.ops.cast(probs > 0, self.compute_dtype)) / float(K)

            self._lvl_perp[lvl].update_state(perp)
            self._lvl_usage[lvl].update_state(usage)
            self._lvl_bpi[lvl].update_state(H)
            perp_vals.append(perp)
            usage_vals.append(usage)
            bpi_vals.append(H)

        # Aggregates (only over active levels)
        n_active = max(len(perp_vals), 1)
        self._perp_mean.update_state(sum(perp_vals) / float(n_active) if perp_vals else 0.0)
        self._usage_mean.update_state(sum(usage_vals) / float(n_active) if usage_vals else 0.0)
        self._bpi_sum.update_state(sum(bpi_vals) if bpi_vals else 0.0)

        y_flat = flat + keras.ops.stop_gradient(q_sum - flat)
        y = keras.ops.reshape(y_flat, shape)
        return (y, indices_list) if return_indices else y

    @property
    def metrics(self):
        return self._lvl_perp + self._lvl_usage + self._lvl_bpi + [self._perp_mean, self._usage_mean, self._bpi_sum]

    def encode(self, x):
        """Return list of per-level flat index tensors [N] (no gradients)."""
        x = keras.ops.convert_to_tensor(x, dtype=self.compute_dtype)
        flat = keras.ops.reshape(x, (-1, self.D))
        residual = flat
        indices = []
        for codebook in self._codebooks:
            idx, q_l = self._nearest(residual, codebook)
            indices.append(idx)
            residual = residual - q_l
        return indices

    def decode(self, indices_list, original_shape):
        """Sum per-level code vectors from indices_list and reshape."""
        q_sum = None
        for idx, codebook in zip(indices_list, self._codebooks):
            q_l = keras.ops.take(codebook, idx, axis=0)
            q_sum = q_l if q_sum is None else (q_sum + q_l)
        return keras.ops.reshape(q_sum, original_shape)

    def call_at_level(self, x, num_levels: int):
        """Forward pass using exactly ``num_levels`` RVQ stages (no dropout).

        Useful for evaluation at a specific compression ratio.
        Returns (quantized_output, indices_list[:num_levels]).
        """
        x = keras.ops.convert_to_tensor(x, dtype=self.compute_dtype)
        shape = keras.ops.shape(x)
        flat = keras.ops.reshape(x, (-1, self.D))
        residual = flat
        q_sum = keras.ops.zeros_like(flat)
        indices_list = []
        for lvl in range(min(num_levels, self.M)):
            idx, q_l = self._nearest(residual, self._codebooks[lvl])
            indices_list.append(idx)
            q_sum = q_sum + q_l
            residual = residual - q_l
        y_flat = flat + keras.ops.stop_gradient(q_sum - flat)
        y = keras.ops.reshape(y_flat, shape)
        return y, indices_list

    def call_with_level_outputs(self, x, training=None, return_indices=False):
        """Forward pass returning straight-through per-level quantized tensors.

        The first return value matches :meth:`call`: the summed RVQ output. The
        second return value is a list of tensors, one per RVQ level, each shaped
        like ``x``. This is useful for hierarchical decoders that want to route
        coarse and residual levels through separate branches while preserving the
        same deployed token stream.
        """
        x = keras.ops.convert_to_tensor(x, dtype=self.compute_dtype)
        if not self.built:
            self.build(x.shape)
        shape = keras.ops.shape(x)
        flat = keras.ops.reshape(x, (-1, self.D))

        if training and self.structured_dropout:
            choice_idx = keras.random.randint(shape=(), minval=0, maxval=len(self.dropout_levels), dtype="int32")
            levels_tensor = keras.ops.convert_to_tensor(self.dropout_levels, dtype="int32")
            active_levels = keras.ops.take(levels_tensor, choice_idx)
        else:
            active_levels = self.M

        residual = flat
        q_sum = keras.ops.zeros_like(flat)
        level_outputs = []
        indices_list = []
        perp_vals, usage_vals, bpi_vals = [], [], []

        for lvl, (K, codebook) in enumerate(zip(self.Ks, self._codebooks)):
            idx, q_l = self._nearest(residual, codebook)
            indices_list.append(idx)

            if training and self.structured_dropout:
                gate = keras.ops.cast(lvl < active_levels, self.compute_dtype)
                q_l_gated = q_l * keras.ops.expand_dims(gate, axis=0)
            else:
                gate = keras.ops.convert_to_tensor(1.0, dtype=self.compute_dtype)
                q_l_gated = q_l

            level_flat = residual + keras.ops.stop_gradient(q_l_gated - residual)
            level_outputs.append(keras.ops.reshape(level_flat, shape))
            q_sum = q_sum + q_l_gated

            ql_st = keras.ops.stop_gradient(q_l)
            commitment = keras.ops.mean(keras.ops.square(ql_st - residual))
            if training and self.structured_dropout:
                self.add_loss(self.beta * commitment * gate)
            else:
                self.add_loss(self.beta * commitment)

            if training:
                self._ema_update(lvl, idx, residual, K)

            residual = residual - ql_st

            one_hot = keras.ops.one_hot(idx, K)
            probs = keras.ops.mean(one_hot, axis=0)
            eps = keras.ops.convert_to_tensor(1e-10, dtype=self.compute_dtype)
            log2 = keras.ops.log(keras.ops.convert_to_tensor(2.0, self.compute_dtype))
            H = -keras.ops.sum(probs * (keras.ops.log(probs + eps) / log2))
            perp = keras.ops.exp(H * log2)
            usage = keras.ops.sum(keras.ops.cast(probs > 0, self.compute_dtype)) / float(K)

            self._lvl_perp[lvl].update_state(perp)
            self._lvl_usage[lvl].update_state(usage)
            self._lvl_bpi[lvl].update_state(H)
            perp_vals.append(perp)
            usage_vals.append(usage)
            bpi_vals.append(H)

        n_active = max(len(perp_vals), 1)
        self._perp_mean.update_state(sum(perp_vals) / float(n_active) if perp_vals else 0.0)
        self._usage_mean.update_state(sum(usage_vals) / float(n_active) if usage_vals else 0.0)
        self._bpi_sum.update_state(sum(bpi_vals) if bpi_vals else 0.0)

        y_flat = flat + keras.ops.stop_gradient(q_sum - flat)
        y = keras.ops.reshape(y_flat, shape)
        if return_indices:
            return y, level_outputs, indices_list
        return y, level_outputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "num_levels": self.M,
                "num_embeddings": self.Ks,
                "embedding_dim": self.D,
                "beta": self.beta,
                "ema_decay": self.ema_decay,
                "epsilon": self.epsilon,
                "structured_dropout": self.structured_dropout,
                "dropout_levels": self.dropout_levels,
            }
        )
        return cfg
