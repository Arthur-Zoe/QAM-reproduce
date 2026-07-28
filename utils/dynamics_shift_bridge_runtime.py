"""Training-lifecycle support for shadow-only dynamics-shift correction."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import numbers

import numpy as np

from utils.dynamics_shift_bridge import (
    DynamicsShiftBridge,
    DynamicsShiftBridgeConfig,
)


BRIDGE_MODEL_SEED_OFFSET = 2_000_003
BRIDGE_SAMPLING_SEED_OFFSET = 2_000_033
PRIMITIVE_TRANSITION_FIELDS = (
    "observations",
    "actions",
    "next_observations",
)
EVALUATION_FIELDS = (
    "normalized_mse",
    "raw_mse",
    "prediction_abs_mean",
    "target_abs_mean",
)
CORRECTION_METRIC_FIELDS = (
    "pre_match_mse",
    "post_match_mse",
    "match_improvement",
    "pre_match_mse_raw",
    "post_match_mse_raw",
    "match_improvement_raw",
    "residual_l2_mean",
    "residual_abs_max",
    "action_clip_fraction",
    "residual_clip_fraction",
)
BRIDGE_LOG_FIELDS = (
    "online_step",
    "recent_buffer_size",
    "offline_eval_normalized_mse",
    "offline_eval_raw_mse",
    "online_eval_normalized_mse",
    "online_eval_raw_mse",
    *CORRECTION_METRIC_FIELDS,
    "offline_model_step",
    "online_model_step",
    "bridge_offline_ready",
    "bridge_online_ready",
    "bridge_shadow_ready",
    "correction_applied_to_environment",
)


@dataclass(frozen=True)
class DynamicsShiftBridgeRuntimeConfig:
    """Runtime-only lifecycle configuration for the bridge."""

    enabled: bool = False
    apply_correction: bool = False
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    learning_rate: float = 3e-4
    clip_grad_norm: float = 10.0
    offline_steps: int = 10_000
    offline_batch_size: int = 256
    online_start_size: int = 500
    online_update_interval: int = 500
    online_updates_per_interval: int = 20
    online_batch_size: int = 256
    correction_steps: int = 10
    correction_step_size: float = 0.1
    dynamics_match_weight: float = 1.0
    action_l2_weight: float = 0.01
    max_residual: float = 0.1
    normalization_max_samples: int = 100_000
    normalization_epsilon: float = 1e-6


def _validated_integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer >= {minimum}; got {value!r}."
        )
    value = int(value)
    if value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}; got {value!r}."
        )
    return value


def _validated_real(value, name, *, positive):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(
            f"{name} must be a finite {qualifier} number; got {value!r}."
        )
    value = float(value)
    invalid_range = value <= 0.0 if positive else value < 0.0
    if not np.isfinite(value) or invalid_range:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(
            f"{name} must be a finite {qualifier} number; got {value!r}."
        )
    return value


def validate_bridge_runtime_config(
    config,
    *,
    recent_dynamics_capacity,
    online_save_interval,
    log_interval,
):
    """Validate all bridge flags and runtime-only dependencies."""
    if not isinstance(config, DynamicsShiftBridgeRuntimeConfig):
        raise ValueError(
            "config must be a DynamicsShiftBridgeRuntimeConfig; "
            f"got {type(config).__name__}."
        )
    if not isinstance(config.enabled, (bool, np.bool_)):
        raise ValueError(
            f"enabled must be boolean; got {config.enabled!r}."
        )
    if not isinstance(config.apply_correction, (bool, np.bool_)):
        raise ValueError(
            "apply_correction must be boolean; "
            f"got {config.apply_correction!r}."
        )
    if config.apply_correction:
        raise NotImplementedError(
            "dynamics_bridge_apply_correction=True is not supported in "
            "shadow mode; real environment action execution will be enabled "
            "in a later PR."
        )

    for name, minimum in (
        ("hidden_dim", 1),
        ("num_hidden_layers", 1),
        ("offline_steps", 0),
        ("offline_batch_size", 1),
        ("online_start_size", 1),
        ("online_update_interval", 1),
        ("online_updates_per_interval", 1),
        ("online_batch_size", 1),
        ("correction_steps", 0),
        ("normalization_max_samples", 1),
    ):
        _validated_integer(getattr(config, name), name, minimum)
    for name in (
        "learning_rate",
        "clip_grad_norm",
        "normalization_epsilon",
    ):
        _validated_real(getattr(config, name), name, positive=True)
    for name in (
        "correction_step_size",
        "dynamics_match_weight",
        "action_l2_weight",
        "max_residual",
    ):
        _validated_real(getattr(config, name), name, positive=False)

    recent_dynamics_capacity = _validated_integer(
        recent_dynamics_capacity, "recent_dynamics_capacity", 0
    )
    online_save_interval = _validated_integer(
        online_save_interval, "online_save_interval", 0
    )
    log_interval = _validated_integer(log_interval, "log_interval", 1)
    if config.enabled:
        if recent_dynamics_capacity == 0:
            raise ValueError(
                "dynamics_bridge requires recent_dynamics_capacity > 0."
            )
        if recent_dynamics_capacity < config.online_start_size:
            raise ValueError(
                "recent_dynamics_capacity must be at least "
                "bridge_online_start_size when dynamics_bridge is enabled; "
                f"got {recent_dynamics_capacity} < "
                f"{config.online_start_size}."
            )
        if online_save_interval > 0:
            raise NotImplementedError(
                "Online checkpoint saving is unavailable while "
                "dynamics_bridge is enabled because checkpoint v4 does not "
                "yet include Bridge state."
            )
    return config


def validate_bridge_checkpoint_resume(enabled, load_stage):
    """Reject an online resume that cannot restore Bridge state."""
    if enabled and load_stage == "online":
        raise NotImplementedError(
            "Online checkpoint resume is unavailable while dynamics_bridge "
            "is enabled because checkpoint v4 does not yet include Bridge "
            "state."
        )


def call_preserving_global_numpy_rng(function, *args, **kwargs):
    """Call an evaluator without advancing NumPy's process-global RNG."""
    state = copy.deepcopy(np.random.get_state())
    try:
        return function(*args, **kwargs)
    finally:
        np.random.set_state(state)


def _normalized_seed(seed, offset, *, bits, name):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {seed!r}.")
    return (int(seed) + offset) % (2**bits)


def _finite_float32_array(
    value,
    name,
    *,
    copy_array=True,
    check_finite=True,
):
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} cannot be converted to an array: {exc}") from exc
    if (
        array.dtype.hasobject
        or np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(
            f"{name} must have a real non-boolean numeric dtype; "
            f"got {array.dtype}."
        )
    if check_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float32, copy=copy_array)
    if check_finite and not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} cannot be represented as finite float32 values."
        )
    return array


def extract_primitive_transitions(dataset, expected_action_shape):
    """Validate raw single-step transition views before sequence sampling."""
    if not isinstance(dataset, Mapping):
        raise ValueError(
            "offline dataset must be a mapping; "
            f"got {type(dataset).__name__}."
        )
    missing = [
        name for name in PRIMITIVE_TRANSITION_FIELDS if name not in dataset
    ]
    if missing:
        raise ValueError(
            f"offline dataset is missing primitive fields: {missing}."
        )
    if not isinstance(expected_action_shape, tuple) or not expected_action_shape:
        raise ValueError(
            "expected_action_shape must be a non-empty tuple; "
            f"got {expected_action_shape!r}."
        )
    transitions = {
        name: _finite_float32_array(
            dataset[name],
            name,
            copy_array=False,
            check_finite=True,
        )
        for name in PRIMITIVE_TRANSITION_FIELDS
    }
    observations = transitions["observations"]
    actions = transitions["actions"]
    next_observations = transitions["next_observations"]
    if observations.ndim < 2:
        raise ValueError(
            "offline observations must have a leading transition axis; "
            f"got shape {observations.shape}."
        )
    if next_observations.shape != observations.shape:
        raise ValueError(
            "offline next_observations must exactly match observations; "
            f"got {next_observations.shape} and {observations.shape}."
        )
    expected_action_rank = len(expected_action_shape) + 1
    if (
        actions.ndim != expected_action_rank
        or actions.shape[1:] != expected_action_shape
    ):
        raise ValueError(
            "offline actions must be primitive actions with shape "
            f"(N, {expected_action_shape}); got {actions.shape}. "
            "Flattened chunks and horizon action sequences are rejected."
        )
    sizes = {
        name: value.shape[0] for name, value in transitions.items()
    }
    if len(set(sizes.values())) != 1:
        raise ValueError(
            "primitive transition fields must have the same size; "
            f"got {sizes}."
        )
    if observations.shape[0] < 2:
        raise ValueError(
            "at least two primitive transitions are required for separate "
            "offline training and held-out evaluation."
        )
    return transitions


def deterministic_uniform_indices(size, max_samples):
    """Return deterministic, evenly spaced indices without using an RNG."""
    size = _validated_integer(size, "size", 1)
    max_samples = _validated_integer(max_samples, "max_samples", 1)
    if size <= max_samples:
        return np.arange(size, dtype=np.int64)
    return np.linspace(
        0, size - 1, num=max_samples, dtype=np.int64
    )


def compute_bridge_normalization(transitions, max_samples):
    """Compute fixed finite float32 statistics from primitive transitions."""
    if not isinstance(transitions, Mapping):
        raise ValueError("transitions must be a mapping.")
    indices = deterministic_uniform_indices(
        len(transitions["observations"]), max_samples
    )
    observations = _finite_float32_array(
        transitions["observations"][indices], "sampled observations"
    )
    actions = _finite_float32_array(
        transitions["actions"][indices], "sampled actions"
    )
    next_observations = _finite_float32_array(
        transitions["next_observations"][indices],
        "sampled next_observations",
    )
    deltas = next_observations - observations

    def mean_std(array, prefix):
        mean = np.mean(array, axis=0, dtype=np.float64).astype(np.float32)
        std = np.std(array, axis=0, dtype=np.float64).astype(np.float32)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError(
                f"non-finite {prefix} normalization statistics."
            )
        return mean, std

    observation_mean, observation_std = mean_std(
        observations, "observation"
    )
    action_mean, action_std = mean_std(actions, "action")
    delta_mean, delta_std = mean_std(deltas, "delta")
    return {
        "observation_mean": observation_mean,
        "observation_std": observation_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "sample_count": int(indices.size),
    }


def should_update_bridge_online(
    *,
    online_step,
    recent_size,
    start_size,
    update_interval,
):
    """Return whether a scheduled online update burst is due."""
    online_step = _validated_integer(online_step, "online_step", 1)
    recent_size = _validated_integer(recent_size, "recent_size", 0)
    start_size = _validated_integer(start_size, "start_size", 1)
    update_interval = _validated_integer(
        update_interval, "update_interval", 1
    )
    return (
        recent_size >= start_size
        and online_step % update_interval == 0
    )


def _transition_subset(transitions, indices):
    return {
        name: np.asarray(transitions[name][indices], dtype=np.float32)
        for name in PRIMITIVE_TRANSITION_FIELDS
    }


def _finite_metric_dict(metrics, expected_fields):
    converted = {}
    for name in expected_fields:
        if name not in metrics:
            raise ValueError(f"metrics are missing required field {name!r}.")
        value = np.asarray(metrics[name], dtype=np.float32)
        if value.shape != () or not np.isfinite(value):
            raise ValueError(
                f"metric {name!r} must be a finite scalar; got {value!r}."
            )
        converted[name] = np.float32(value)
    return converted


class DynamicsShiftBridgeRuntime:
    """Mutable Python lifecycle around the immutable JAX Bridge state."""

    def __init__(
        self,
        *,
        config,
        bridge,
        sampling_rng,
        offline_eval,
        normalization_sample_count,
    ):
        self.config = config
        self.bridge = bridge
        self.sampling_rng = sampling_rng
        self.offline_eval = offline_eval
        self.online_eval = {
            name: np.float32(0.0) for name in EVALUATION_FIELDS
        }
        self.normalization_sample_count = normalization_sample_count
        self.bridge_offline_ready = True
        self.bridge_online_ready = False
        self.online_update_bursts = 0
        self.last_base_action = None
        self.last_corrected_action = None
        self.last_executed_action = None
        self._correction_sums = {
            name: np.float64(0.0) for name in CORRECTION_METRIC_FIELDS
        }
        self._correction_count = 0
        self._last_logged_online_step = 0

    @property
    def bridge_shadow_ready(self):
        return self.bridge_offline_ready and self.bridge_online_ready

    @classmethod
    def create(
        cls,
        *,
        config,
        dataset,
        expected_action_shape,
        action_low,
        action_high,
        seed,
    ):
        """Extract, normalize, pretrain, evaluate, and synchronize."""
        if not config.enabled:
            raise ValueError(
                "DynamicsShiftBridgeRuntime.create requires enabled=True."
            )
        transitions = extract_primitive_transitions(
            dataset, expected_action_shape
        )
        normalization = compute_bridge_normalization(
            transitions, config.normalization_max_samples
        )
        model_seed = _normalized_seed(
            seed,
            BRIDGE_MODEL_SEED_OFFSET,
            bits=32,
            name="Bridge model seed",
        )
        sampling_seed = _normalized_seed(
            seed,
            BRIDGE_SAMPLING_SEED_OFFSET,
            bits=64,
            name="Bridge sampling seed",
        )
        sampling_rng = np.random.default_rng(sampling_seed)
        bridge = DynamicsShiftBridge.create(
            seed=model_seed,
            example_observations=transitions["observations"][:1],
            example_actions=transitions["actions"][:1],
            config=DynamicsShiftBridgeConfig(
                hidden_dim=config.hidden_dim,
                num_hidden_layers=config.num_hidden_layers,
                learning_rate=config.learning_rate,
                clip_grad_norm=config.clip_grad_norm,
                correction_steps=config.correction_steps,
                correction_step_size=config.correction_step_size,
                dynamics_match_weight=config.dynamics_match_weight,
                action_l2_weight=config.action_l2_weight,
                max_residual=config.max_residual,
                action_low=action_low,
                action_high=action_high,
                observation_mean=normalization["observation_mean"],
                observation_std=normalization["observation_std"],
                action_mean=normalization["action_mean"],
                action_std=normalization["action_std"],
                delta_mean=normalization["delta_mean"],
                delta_std=normalization["delta_std"],
                normalization_epsilon=config.normalization_epsilon,
            ),
        )

        size = transitions["observations"].shape[0]
        heldout_size = min(
            config.offline_batch_size,
            max(1, size // 10),
        )
        training_size = size - heldout_size
        heldout_indices = np.arange(
            size - heldout_size, size, dtype=np.int64
        )
        for _ in range(config.offline_steps):
            sampled_indices = sampling_rng.integers(
                0,
                training_size,
                size=config.offline_batch_size,
            )
            batch = _transition_subset(transitions, sampled_indices)
            bridge, _ = bridge.update_offline(batch)
        offline_eval = _finite_metric_dict(
            bridge.evaluate_offline(
                _transition_subset(transitions, heldout_indices)
            ),
            EVALUATION_FIELDS,
        )
        bridge = bridge.synchronize_online_from_offline()
        return cls(
            config=config,
            bridge=bridge,
            sampling_rng=sampling_rng,
            offline_eval=offline_eval,
            normalization_sample_count=normalization["sample_count"],
        )

    def sampling_rng_state(self):
        """Return a copy suitable for future checkpoint integration."""
        return copy.deepcopy(self.sampling_rng.bit_generator.state)

    def maybe_update_online(self, online_step, recent_buffer):
        """Run one scheduled online update burst after buffer insertion."""
        if not should_update_bridge_online(
            online_step=online_step,
            recent_size=recent_buffer.size,
            start_size=self.config.online_start_size,
            update_interval=self.config.online_update_interval,
        ):
            return False
        for _ in range(self.config.online_updates_per_interval):
            # RecentDynamicsBuffer.sample() explicitly samples with
            # replacement, so batch_size may exceed the current size.
            sampled = recent_buffer.sample(
                self.config.online_batch_size,
                rng=self.sampling_rng,
            )
            batch = {
                name: sampled[name] for name in PRIMITIVE_TRANSITION_FIELDS
            }
            self.bridge, _ = self.bridge.update_online(batch)
        evaluation_batch = recent_buffer.sample(
            self.config.online_batch_size,
            rng=self.sampling_rng,
        )
        self.online_eval = _finite_metric_dict(
            self.bridge.evaluate_online(
                {
                    name: evaluation_batch[name]
                    for name in PRIMITIVE_TRANSITION_FIELDS
                }
            ),
            EVALUATION_FIELDS,
        )
        self.online_update_bursts += 1
        self.bridge_online_ready = True
        return True

    def shadow_correct(self, observation, base_action):
        """Compute but never select a corrected primitive action."""
        base_action = _finite_float32_array(base_action, "base_action")
        if not self.bridge_shadow_ready:
            corrected_action = base_action.copy()
            metrics = {
                name: np.float32(0.0)
                for name in CORRECTION_METRIC_FIELDS
            }
        else:
            corrected_action, raw_metrics = self.bridge.correct_actions(
                observation, base_action
            )
            metrics = _finite_metric_dict(
                raw_metrics, CORRECTION_METRIC_FIELDS
            )
            for name in (
                "action_clip_fraction",
                "residual_clip_fraction",
            ):
                if not 0.0 <= float(metrics[name]) <= 1.0:
                    raise ValueError(
                        f"metric {name!r} must be in [0, 1]; "
                        f"got {metrics[name]!r}."
                    )
            for name, value in metrics.items():
                self._correction_sums[name] += float(value)
            self._correction_count += 1
        corrected_action = np.asarray(
            corrected_action, dtype=np.float32
        )
        self.last_base_action = base_action.copy()
        self.last_corrected_action = corrected_action.copy()
        return corrected_action, metrics

    def environment_action(self, base_action, corrected_action):
        """Return the base action and enforce the shadow-only contract."""
        if self.config.apply_correction:
            raise RuntimeError(
                "Corrected environment actions are disabled in this PR."
            )
        base_action_array = np.asarray(base_action)
        _finite_float32_array(base_action_array, "base_action")
        corrected_action = _finite_float32_array(
            corrected_action, "corrected_action"
        )
        if corrected_action.shape != base_action_array.shape:
            raise ValueError(
                "corrected_action shape must match base_action; "
                f"got {corrected_action.shape} and "
                f"{base_action_array.shape}."
            )
        self.last_executed_action = base_action_array.copy()
        return self.last_executed_action.copy()

    def log_row(self, online_step, recent_buffer_size, *, reset=True):
        """Build one finite aggregate row for dynamics_shift_bridge.csv."""
        online_step = _validated_integer(
            online_step, "online_step", 1
        )
        recent_buffer_size = _validated_integer(
            recent_buffer_size, "recent_buffer_size", 0
        )
        if online_step <= self._last_logged_online_step:
            raise ValueError(
                "Bridge log online_step must be strictly increasing; "
                f"got {online_step} after "
                f"{self._last_logged_online_step}."
            )
        if self._correction_count:
            correction_metrics = {
                name: np.float32(
                    value / self._correction_count
                )
                for name, value in self._correction_sums.items()
            }
        else:
            correction_metrics = {
                name: np.float32(0.0)
                for name in CORRECTION_METRIC_FIELDS
            }
        row = {
            "online_step": int(online_step),
            "recent_buffer_size": int(recent_buffer_size),
            "offline_eval_normalized_mse": self.offline_eval[
                "normalized_mse"
            ],
            "offline_eval_raw_mse": self.offline_eval["raw_mse"],
            "online_eval_normalized_mse": self.online_eval[
                "normalized_mse"
            ],
            "online_eval_raw_mse": self.online_eval["raw_mse"],
            **correction_metrics,
            "offline_model_step": int(self.bridge.offline_model.step),
            "online_model_step": int(self.bridge.online_model.step),
            "bridge_offline_ready": int(self.bridge_offline_ready),
            "bridge_online_ready": int(self.bridge_online_ready),
            "bridge_shadow_ready": int(self.bridge_shadow_ready),
            "correction_applied_to_environment": 0,
        }
        if tuple(row) != BRIDGE_LOG_FIELDS:
            raise RuntimeError(
                "Bridge log row schema drifted from BRIDGE_LOG_FIELDS."
            )
        for name, value in row.items():
            array = np.asarray(value)
            if array.shape != () or not np.isfinite(array):
                raise ValueError(
                    f"Bridge log field {name!r} is not finite scalar: "
                    f"{value!r}."
                )
        if reset:
            self._correction_sums = {
                name: np.float64(0.0)
                for name in CORRECTION_METRIC_FIELDS
            }
            self._correction_count = 0
        self._last_logged_online_step = online_step
        return row


def shadow_step_environment(
    runtime,
    env,
    observation,
    base_action,
):
    """Evaluate one primitive correction, then execute the base action."""
    corrected_action, correction_metrics = runtime.shadow_correct(
        observation, base_action
    )
    executed_action = runtime.environment_action(
        base_action, corrected_action
    )
    if not np.array_equal(executed_action, np.asarray(base_action)):
        raise RuntimeError(
            "Shadow mode attempted to change the environment action."
        )
    environment_result = env.step(executed_action)
    return (
        environment_result,
        executed_action,
        corrected_action,
        correction_metrics,
    )
