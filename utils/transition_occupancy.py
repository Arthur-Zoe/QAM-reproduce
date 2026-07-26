"""Transition-level occupancy classification for online dynamics monitoring."""

from collections.abc import Mapping
import numbers
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.flax_utils import TrainState, nonpytree_field


TRANSITION_FIELDS = (
    "observations",
    "actions",
    "next_observations",
)
DEFAULT_CONFIG = {
    "hidden_dim": 256,
    "num_hidden_layers": 2,
    "learning_rate": 3e-4,
    "clip_grad_norm": 10.0,
}


def _validated_integer(value, name, *, minimum):
    if isinstance(value, bool) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(
            f"{name} must be an integer >= {minimum}; got {value!r}."
        )
    value = int(value)
    if value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}; got {value!r}."
        )
    return value


def should_update_occupancy_detector(
    online_step,
    recent_size,
    enabled,
    start_size,
    update_interval,
):
    """Return whether a detector update burst is due at this online step."""
    online_step = _validated_integer(
        online_step, "online_step", minimum=0
    )
    recent_size = _validated_integer(
        recent_size, "recent_size", minimum=0
    )
    start_size = _validated_integer(
        start_size, "start_size", minimum=1
    )
    update_interval = _validated_integer(
        update_interval, "update_interval", minimum=1
    )
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError(f"enabled must be a boolean; got {enabled!r}.")
    return bool(
        enabled
        and online_step > 0
        and recent_size >= start_size
        and online_step % update_interval == 0
    )


def average_occupancy_metrics(metric_dicts):
    """Return the per-key arithmetic mean across a detector update burst."""
    if not metric_dicts:
        raise ValueError(
            "metric_dicts must be a non-empty sequence of metric mappings."
        )
    if not all(isinstance(metrics, Mapping) for metrics in metric_dicts):
        raise ValueError("Every detector metric collection must be a mapping.")
    expected_keys = set(metric_dicts[0])
    for metrics in metric_dicts[1:]:
        if set(metrics) != expected_keys:
            raise ValueError(
                "Every detector update must return the same metric keys."
            )
    return {
        name: jnp.mean(
            jnp.stack([jnp.asarray(metrics[name]) for metrics in metric_dicts])
        )
        for name in metric_dicts[0]
    }


def pack_transition_features(batch):
    """Flatten and concatenate batched ``(s, a, s')`` transition fields."""
    if not isinstance(batch, Mapping):
        raise ValueError(
            "transition batch must be a mapping; "
            f"got {type(batch).__name__}."
        )
    missing = [field for field in TRANSITION_FIELDS if field not in batch]
    if missing:
        raise ValueError(
            "transition batch is missing required fields: "
            f"{missing}."
        )

    arrays = []
    for field in TRANSITION_FIELDS:
        try:
            array = np.asarray(batch[field])
        except Exception as exc:
            raise ValueError(
                f"transition field {field!r} cannot be converted to an array: "
                f"{exc}"
            ) from exc
        if array.dtype.hasobject:
            raise ValueError(
                f"transition field {field!r} has unsupported object dtype "
                f"{array.dtype}."
            )
        if not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise ValueError(
                f"transition field {field!r} must have a numeric or boolean "
                f"dtype; got {array.dtype}."
            )
        if np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(
                f"transition field {field!r} must have a real-valued dtype; "
                f"got {array.dtype}."
            )
        if (
            np.issubdtype(array.dtype, np.number)
            and not np.all(np.isfinite(array))
        ):
            raise ValueError(
                f"transition field {field!r} must contain only finite "
                "values."
            )
        if array.ndim < 2:
            raise ValueError(
                f"transition field {field!r} must include distinct batch "
                "and feature axes; "
                f"got shape {array.shape}."
            )
        arrays.append(array)
    batch_sizes = [array.shape[0] for array in arrays]
    if len(set(batch_sizes)) != 1:
        raise ValueError(
            "transition fields must have the same batch size; "
            f"got {dict(zip(TRANSITION_FIELDS, batch_sizes))}."
        )
    batch_size = batch_sizes[0]
    if batch_size == 0:
        raise ValueError("transition batch must not be empty.")
    flattened = []
    for field, array in zip(TRANSITION_FIELDS, arrays):
        with np.errstate(over="ignore", invalid="ignore"):
            features = array.reshape(batch_size, -1).astype(
                np.float32, copy=False
            )
        if not np.all(np.isfinite(features)):
            raise ValueError(
                f"transition field {field!r} cannot be represented as "
                "finite float32 features."
            )
        flattened.append(features)
    return np.concatenate(flattened, axis=-1)


def _validated_config(config):
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError(
            "occupancy detector config must be a mapping; "
            f"got {type(config).__name__}."
        )
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(
            "occupancy detector config has unsupported fields: "
            f"{sorted(unknown)}."
        )
    values = {**DEFAULT_CONFIG, **dict(config)}

    for name in ("hidden_dim", "num_hidden_layers"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(
                f"{name} must be a positive integer; got {value!r}."
            )
        value = int(value)
        if value <= 0:
            raise ValueError(
                f"{name} must be a positive integer; got {value!r}."
            )
        values[name] = value

    for name in ("learning_rate", "clip_grad_norm"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(
                f"{name} must be a finite positive number; got {value!r}."
            )
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be a finite positive number; got {value!r}."
            )
        values[name] = value
    return flax.core.FrozenDict(values)


class TransitionDiscriminator(nn.Module):
    """Tanh MLP that returns one uncalibrated logit per transition."""

    hidden_dim: int
    num_hidden_layers: int

    @nn.compact
    def __call__(self, features):
        hidden = features
        for layer_index in range(self.num_hidden_layers):
            hidden = nn.Dense(
                self.hidden_dim,
                name=f"hidden_{layer_index}",
            )(hidden)
            hidden = jnp.tanh(hidden)
        logits = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="logits",
        )(hidden)
        return logits.squeeze(-1)


def _metrics_from_logits(offline_logits, online_logits):
    offline_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            offline_logits, jnp.zeros_like(offline_logits)
        )
    )
    online_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            online_logits, jnp.ones_like(online_logits)
        )
    )
    loss = 0.5 * (offline_loss + online_loss)
    offline_probability = jax.nn.sigmoid(offline_logits)
    online_probability = jax.nn.sigmoid(online_logits)
    offline_accuracy = jnp.mean(offline_logits < 0.0)
    online_accuracy = jnp.mean(online_logits >= 0.0)
    offline_logit_mean = jnp.mean(offline_logits)
    online_logit_mean = jnp.mean(online_logits)
    return {
        "loss": loss,
        "offline_loss": offline_loss,
        "online_loss": online_loss,
        "offline_accuracy": offline_accuracy,
        "online_accuracy": online_accuracy,
        "balanced_accuracy": 0.5
        * (offline_accuracy + online_accuracy),
        "offline_logit_mean": offline_logit_mean,
        "online_logit_mean": online_logit_mean,
        "offline_probability_mean": jnp.mean(offline_probability),
        "online_probability_mean": jnp.mean(online_probability),
        "logit_gap": online_logit_mean - offline_logit_mean,
    }


class TransitionOccupancyDetector(flax.struct.PyTreeNode):
    """Classifier state with offline label 0 and recent-online label 1."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, example_offline_transition, config=None):
        if isinstance(seed, bool) or not isinstance(
            seed, (int, np.integer)
        ):
            raise ValueError(f"seed must be an integer; got {seed!r}.")
        config = _validated_config(config)
        features = pack_transition_features(example_offline_transition)
        features = jnp.asarray(features, dtype=jnp.float32)

        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        model = TransitionDiscriminator(
            hidden_dim=config["hidden_dim"],
            num_hidden_layers=config["num_hidden_layers"],
        )
        params = model.init(init_rng, features)["params"]
        optimizer = optax.chain(
            optax.clip_by_global_norm(config["clip_grad_norm"]),
            optax.adam(config["learning_rate"]),
        )
        network = TrainState.create(
            model_def=model,
            params=params,
            tx=optimizer,
        )
        return cls(rng=rng, network=network, config=config)

    def logits(self, batch):
        features = jnp.asarray(
            pack_transition_features(batch), dtype=jnp.float32
        )
        return self.network(features)

    def online_probability(self, batch):
        return jax.nn.sigmoid(self.logits(batch))

    def log_density_ratio_proxy(self, batch):
        """Return the uncalibrated log occupancy-ratio proxy."""
        return self.logits(batch)

    def next_sampling_generator(self):
        """Return a detector-owned NumPy generator and advance its JAX key."""
        next_rng, sampling_rng = jax.random.split(self.rng)
        entropy = np.asarray(
            jax.random.bits(
                sampling_rng,
                shape=(4,),
                dtype=jnp.uint32,
            ),
            dtype=np.uint32,
        )
        seed_sequence = np.random.SeedSequence(entropy.tolist())
        generator = np.random.default_rng(seed_sequence)
        return self.replace(rng=next_rng), generator

    @jax.jit
    def _evaluate_packed(self, offline_features, online_features):
        offline_logits = self.network(offline_features)
        online_logits = self.network(online_features)
        return _metrics_from_logits(offline_logits, online_logits)

    def evaluate(self, offline_batch, online_batch):
        """Evaluate balanced batches without changing state or RNG."""
        offline_features = pack_transition_features(offline_batch)
        online_features = pack_transition_features(online_batch)
        if offline_features.shape[0] != online_features.shape[0]:
            raise ValueError(
                "offline and online transition batches must have the same "
                f"batch size; got {offline_features.shape[0]} and "
                f"{online_features.shape[0]}."
            )
        return self._evaluate_packed(
            jnp.asarray(offline_features, dtype=jnp.float32),
            jnp.asarray(online_features, dtype=jnp.float32),
        )

    @jax.jit
    def _update_packed(self, offline_features, online_features):
        def loss_fn(params):
            offline_logits = self.network(
                offline_features, params=params
            )
            online_logits = self.network(
                online_features, params=params
            )
            metrics = _metrics_from_logits(
                offline_logits, online_logits
            )
            return metrics["loss"], metrics

        (_, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.network.params)
        network = self.network.apply_gradients(grads=grads)
        return self.replace(network=network), metrics

    def update(self, offline_batch, online_batch):
        offline_features = pack_transition_features(offline_batch)
        online_features = pack_transition_features(online_batch)
        if offline_features.shape[0] != online_features.shape[0]:
            raise ValueError(
                "offline and online transition batches must have the same "
                f"batch size; got {offline_features.shape[0]} and "
                f"{online_features.shape[0]}."
            )
        return self._update_packed(
            jnp.asarray(offline_features, dtype=jnp.float32),
            jnp.asarray(online_features, dtype=jnp.float32),
        )


def sample_occupancy_transition_batches(
    detector,
    offline_dataset,
    recent_buffer,
    batch_size,
):
    """Sample balanced detector batches without using global NumPy state."""
    if not isinstance(detector, TransitionOccupancyDetector):
        raise ValueError(
            "detector must be a TransitionOccupancyDetector; "
            f"got {type(detector).__name__}."
        )
    batch_size = _validated_integer(
        batch_size, "batch_size", minimum=1
    )
    if not hasattr(offline_dataset, "size"):
        raise ValueError("offline_dataset must expose a size attribute.")
    offline_size = _validated_integer(
        offline_dataset.size, "offline_dataset.size", minimum=1
    )

    detector, generator = detector.next_sampling_generator()
    offline_indices = generator.integers(
        0,
        offline_size,
        size=batch_size,
    )
    offline_sample = offline_dataset.sample(
        batch_size,
        idxs=offline_indices,
    )
    online_sample = recent_buffer.sample(batch_size, rng=generator)
    offline_batch = {
        field: offline_sample[field] for field in TRANSITION_FIELDS
    }
    online_batch = {
        field: online_sample[field] for field in TRANSITION_FIELDS
    }
    return detector, offline_batch, online_batch
