"""Training-lifecycle support for gated dynamics-shift correction."""

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
EXECUTION_METRIC_FIELDS = (
    "online_model_uncertainty",
    "shift_excess",
    "relative_match_improvement",
    "gate_readiness",
    "gate_model_confident",
    "gate_shift_detected",
    "gate_correction_quality",
    "gate_open",
    "ramp_scale",
    "candidate_residual_l2",
    "executed_residual_l2",
    "executed_residual_abs_max",
    "correction_requested",
    "correction_applied_to_environment",
)
EXECUTION_RATE_FIELDS = (
    "gate_open_fraction",
    "correction_requested_fraction",
    "correction_applied_fraction",
)
EVALUATION_EXECUTION_METRIC_FIELDS = (
    "evaluation_bridge_ready_fraction",
    "evaluation_bridge_gate_open_fraction",
    "evaluation_bridge_requested_fraction",
    "evaluation_bridge_applied_fraction",
    "evaluation_bridge_executed_residual_l2",
    "evaluation_bridge_executed_residual_abs_max",
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
    *EXECUTION_METRIC_FIELDS,
    *EXECUTION_RATE_FIELDS,
    "bridge_applied_steps",
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
    gate_max_online_eval_mse: float = 0.10
    gate_uncertainty_multiplier: float = 1.0
    gate_min_shift_excess: float = 0.005
    gate_min_relative_improvement: float = 0.20
    apply_ramp_steps: int = 1_000
    apply_residual_scale: float = 1.0
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
    if config.apply_correction and not config.enabled:
        raise ValueError(
            "dynamics_bridge_apply_correction=True requires "
            "dynamics_bridge=True."
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
        ("apply_ramp_steps", 0),
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
        "gate_max_online_eval_mse",
        "gate_uncertainty_multiplier",
        "gate_min_shift_excess",
    ):
        _validated_real(getattr(config, name), name, positive=False)
    gate_min_relative_improvement = _validated_real(
        config.gate_min_relative_improvement,
        "gate_min_relative_improvement",
        positive=False,
    )
    if gate_min_relative_improvement > 1.0:
        raise ValueError(
            "gate_min_relative_improvement must be at most 1; "
            f"got {gate_min_relative_improvement!r}."
        )
    apply_residual_scale = _validated_real(
        config.apply_residual_scale,
        "apply_residual_scale",
        positive=True,
    )
    if apply_residual_scale > 1.0:
        raise ValueError(
            "apply_residual_scale must be at most 1 so executed residuals "
            f"remain within Bridge bounds; got {apply_residual_scale!r}."
        )

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
        self._online_heldout_count = 0
        self.bridge_applied_steps = 0
        self.last_base_action = None
        self.last_corrected_action = None
        self.last_executed_action = None
        self.last_execution_metrics = {
            name: np.float32(0.0)
            for name in EXECUTION_METRIC_FIELDS
        }
        self._correction_sums = {
            name: np.float64(0.0) for name in CORRECTION_METRIC_FIELDS
        }
        self._correction_count = 0
        self._execution_sums = {
            name: np.float64(0.0) for name in EXECUTION_METRIC_FIELDS
        }
        self._execution_count = 0
        self._last_logged_online_step = 0
        self._is_evaluation_snapshot = False

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

    def make_evaluation_snapshot(self):
        """Create an episode-local correction runtime with isolated state."""
        # The Bridge and its TrainStates are immutable Flax pytrees, so they
        # can be shared safely. Copy only the mutable Python runtime state;
        # duplicating the full model and optimizer trees per episode would be
        # both unnecessary and expensive.
        snapshot = copy.copy(self)
        snapshot.sampling_rng = copy.deepcopy(self.sampling_rng)
        snapshot.offline_eval = dict(self.offline_eval)
        snapshot.online_eval = dict(self.online_eval)
        snapshot._correction_sums = {
            name: np.float64(0.0)
            for name in CORRECTION_METRIC_FIELDS
        }
        snapshot._correction_count = 0
        snapshot._execution_sums = {
            name: np.float64(0.0)
            for name in EXECUTION_METRIC_FIELDS
        }
        snapshot._execution_count = 0
        snapshot.last_base_action = None
        snapshot.last_corrected_action = None
        snapshot.last_executed_action = None
        snapshot.last_execution_metrics = {
            name: np.float32(0.0)
            for name in EXECUTION_METRIC_FIELDS
        }
        snapshot._is_evaluation_snapshot = True
        return snapshot

    def maybe_update_online(self, online_step, recent_buffer):
        """Refresh held-out uncertainty and run a due online update burst."""
        update_due = should_update_bridge_online(
            online_step=online_step,
            recent_size=recent_buffer.size,
            start_size=self.config.online_start_size,
            update_interval=self.config.online_update_interval,
        )
        if not update_due and not self.bridge_online_ready:
            return False
        recent_transitions = recent_buffer.ordered_data()
        recent_size = recent_transitions["observations"].shape[0]
        if not update_due:
            # One new transition has been inserted since the previous call.
            # It and the tail held out at the last burst have never trained
            # the current online model. Refresh uncertainty on that rolling
            # held-out tail so execution gates do not use a burst-old MSE.
            self._online_heldout_count = min(
                recent_size,
                self._online_heldout_count + 1,
            )
            evaluation_size = min(
                self.config.online_batch_size,
                self._online_heldout_count,
            )
            evaluation_indices = np.arange(
                recent_size - evaluation_size,
                recent_size,
                dtype=np.int64,
            )
            self.online_eval = _finite_metric_dict(
                self.bridge.evaluate_online(
                    _transition_subset(
                        recent_transitions, evaluation_indices
                    )
                ),
                EVALUATION_FIELDS,
            )
            return False
        if recent_size == 1:
            # A disjoint split is impossible. Preserve the documented
            # start_size=1 configuration with one explicitly shared sample.
            training_size = 1
            heldout_indices = np.asarray([0], dtype=np.int64)
        else:
            heldout_size = min(
                self.config.online_batch_size,
                max(1, recent_size // 10),
                recent_size - 1,
            )
            training_size = recent_size - heldout_size
            heldout_indices = np.arange(
                training_size, recent_size, dtype=np.int64
            )
        self._online_heldout_count = int(heldout_indices.size)
        for _ in range(self.config.online_updates_per_interval):
            # Sample with replacement from the training prefix. The newest
            # held-out tail is used only for post-burst model evaluation.
            training_indices = self.sampling_rng.integers(
                0,
                training_size,
                size=self.config.online_batch_size,
            )
            batch = _transition_subset(
                recent_transitions, training_indices
            )
            self.bridge, _ = self.bridge.update_online(batch)
        self.online_eval = _finite_metric_dict(
            self.bridge.evaluate_online(
                _transition_subset(
                    recent_transitions, heldout_indices
                )
            ),
            EVALUATION_FIELDS,
        )
        self.online_update_bursts += 1
        self.bridge_online_ready = True
        return True

    def shadow_correct(self, observation, base_action):
        """Compute one bounded candidate primitive action."""
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

    def environment_action(
        self,
        base_action,
        corrected_action,
        correction_metrics,
    ):
        """Apply gates and a residual ramp to select the environment action."""
        base_action_array = np.asarray(base_action)
        base_action_float32 = _finite_float32_array(
            base_action_array, "base_action"
        )
        corrected_action = _finite_float32_array(
            corrected_action, "corrected_action"
        )
        if corrected_action.shape != base_action_array.shape:
            raise ValueError(
                "corrected_action shape must match base_action; "
                f"got {corrected_action.shape} and "
                f"{base_action_array.shape}."
            )
        correction_metrics = _finite_metric_dict(
            correction_metrics, CORRECTION_METRIC_FIELDS
        )
        online_uncertainty = np.float32(
            self.online_eval["normalized_mse"]
        )
        pre_match_mse = np.float32(
            correction_metrics["pre_match_mse"]
        )
        match_improvement = np.float32(
            correction_metrics["match_improvement"]
        )
        shift_excess = np.float32(
            pre_match_mse
            - self.config.gate_uncertainty_multiplier
            * online_uncertainty
        )
        relative_match_improvement = np.float32(
            match_improvement
            / max(
                float(pre_match_mse),
                float(np.finfo(np.float32).eps),
            )
        )
        gate_readiness = int(self.bridge_shadow_ready)
        gate_model_confident = int(
            online_uncertainty
            <= self.config.gate_max_online_eval_mse
        )
        gate_shift_detected = int(
            shift_excess >= self.config.gate_min_shift_excess
        )
        gate_correction_quality = int(
            relative_match_improvement
            >= self.config.gate_min_relative_improvement
            and match_improvement > 0.0
        )
        gate_open = int(
            gate_readiness
            and gate_model_confident
            and gate_shift_detected
            and gate_correction_quality
        )

        candidate_residual = (
            corrected_action - base_action_float32
        )
        candidate_residual_l2 = np.float32(
            np.linalg.norm(candidate_residual.reshape(-1), ord=2)
        )
        correction_requested = int(
            self.config.apply_correction and gate_open
        )
        ramp_scale = np.float32(0.0)
        if correction_requested:
            if not np.issubdtype(
                base_action_array.dtype, np.floating
            ):
                raise ValueError(
                    "executed corrected actions require a floating-point "
                    f"base action dtype; got {base_action_array.dtype}."
                )
            self.bridge_applied_steps += 1
            if self.config.apply_ramp_steps == 0:
                ramp_scale = np.float32(1.0)
            else:
                ramp_scale = np.float32(
                    min(
                        1.0,
                        self.bridge_applied_steps
                        / self.config.apply_ramp_steps,
                    )
                )
            scaled_residual = (
                ramp_scale
                * self.config.apply_residual_scale
                * candidate_residual
            )
            scaled_residual = np.clip(
                scaled_residual,
                -np.asarray(self.bridge.max_residual),
                np.asarray(self.bridge.max_residual),
            )
            executed_float32 = np.clip(
                base_action_float32 + scaled_residual,
                np.asarray(self.bridge.action_low),
                np.asarray(self.bridge.action_high),
            )
            executed_action = np.asarray(
                executed_float32,
                dtype=base_action_array.dtype,
            )
        else:
            executed_action = base_action_array.copy()

        executed_residual = (
            np.asarray(executed_action, dtype=np.float32)
            - base_action_float32
        )
        executed_residual_l2 = np.float32(
            np.linalg.norm(executed_residual.reshape(-1), ord=2)
        )
        executed_residual_abs_max = np.float32(
            np.max(np.abs(executed_residual))
        )
        correction_applied = int(
            not np.array_equal(executed_action, base_action_array)
        )
        execution_metrics = {
            "online_model_uncertainty": online_uncertainty,
            "shift_excess": shift_excess,
            "relative_match_improvement": (
                relative_match_improvement
            ),
            "gate_readiness": np.float32(gate_readiness),
            "gate_model_confident": np.float32(
                gate_model_confident
            ),
            "gate_shift_detected": np.float32(
                gate_shift_detected
            ),
            "gate_correction_quality": np.float32(
                gate_correction_quality
            ),
            "gate_open": np.float32(gate_open),
            "ramp_scale": ramp_scale,
            "candidate_residual_l2": candidate_residual_l2,
            "executed_residual_l2": executed_residual_l2,
            "executed_residual_abs_max": (
                executed_residual_abs_max
            ),
            "correction_requested": np.float32(
                correction_requested
            ),
            "correction_applied_to_environment": np.float32(
                correction_applied
            ),
        }
        execution_metrics = _finite_metric_dict(
            execution_metrics, EXECUTION_METRIC_FIELDS
        )
        for name in (
            "gate_readiness",
            "gate_model_confident",
            "gate_shift_detected",
            "gate_correction_quality",
            "gate_open",
            "ramp_scale",
            "correction_requested",
            "correction_applied_to_environment",
        ):
            if not 0.0 <= float(execution_metrics[name]) <= 1.0:
                raise ValueError(
                    f"execution metric {name!r} must be in [0, 1]."
                )
        for name, value in execution_metrics.items():
            self._execution_sums[name] += float(value)
        self._execution_count += 1
        self.last_execution_metrics = execution_metrics
        self.last_executed_action = executed_action.copy()
        return self.last_executed_action.copy()

    def _select_primitive_action(self, observation, base_action):
        corrected_action, correction_metrics = self.shadow_correct(
            observation, base_action
        )
        executed_action = self.environment_action(
            base_action,
            corrected_action,
            correction_metrics,
        )
        if (
            not self.config.apply_correction
            and not np.array_equal(
                executed_action, np.asarray(base_action)
            )
        ):
            raise RuntimeError(
                "Shadow mode attempted to change the environment action."
            )
        return executed_action, corrected_action, correction_metrics

    def evaluate_action(self, observation, proposed_action):
        """Transform one primitive action on an evaluation-only snapshot."""
        if not self._is_evaluation_snapshot:
            raise RuntimeError(
                "evaluate_action requires make_evaluation_snapshot()."
            )
        executed_action, _, _ = self._select_primitive_action(
            observation, proposed_action
        )
        metrics = self.last_execution_metrics
        diagnostics = {
            "evaluation_bridge_ready_fraction": np.float32(
                metrics["gate_readiness"]
            ),
            "evaluation_bridge_gate_open_fraction": np.float32(
                metrics["gate_open"]
            ),
            "evaluation_bridge_requested_fraction": np.float32(
                metrics["correction_requested"]
            ),
            "evaluation_bridge_applied_fraction": np.float32(
                metrics["correction_applied_to_environment"]
            ),
            "evaluation_bridge_executed_residual_l2": np.float32(
                metrics["executed_residual_l2"]
            ),
            "evaluation_bridge_executed_residual_abs_max": np.float32(
                metrics["executed_residual_abs_max"]
            ),
        }
        if tuple(diagnostics) != EVALUATION_EXECUTION_METRIC_FIELDS:
            raise RuntimeError(
                "evaluation Bridge diagnostics schema drifted."
            )
        return executed_action, diagnostics

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
        if self._execution_count:
            execution_metrics = {
                name: np.float32(
                    value / self._execution_count
                )
                for name, value in self._execution_sums.items()
            }
            for name in (
                "gate_readiness",
                "gate_model_confident",
                "gate_shift_detected",
                "gate_correction_quality",
                "gate_open",
                "ramp_scale",
                "correction_requested",
                "correction_applied_to_environment",
            ):
                execution_metrics[name] = self.last_execution_metrics[
                    name
                ]
            execution_rates = {
                "gate_open_fraction": np.float32(
                    self._execution_sums["gate_open"]
                    / self._execution_count
                ),
                "correction_requested_fraction": np.float32(
                    self._execution_sums["correction_requested"]
                    / self._execution_count
                ),
                "correction_applied_fraction": np.float32(
                    self._execution_sums[
                        "correction_applied_to_environment"
                    ]
                    / self._execution_count
                ),
            }
        else:
            execution_metrics = {
                name: np.float32(0.0)
                for name in EXECUTION_METRIC_FIELDS
            }
            execution_rates = {
                name: np.float32(0.0)
                for name in EXECUTION_RATE_FIELDS
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
            **execution_metrics,
            **execution_rates,
            "bridge_applied_steps": int(
                self.bridge_applied_steps
            ),
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
            self._execution_sums = {
                name: np.float64(0.0)
                for name in EXECUTION_METRIC_FIELDS
            }
            self._execution_count = 0
        self._last_logged_online_step = online_step
        return row


def bridge_step_environment(
    runtime,
    env,
    observation,
    base_action,
):
    """Correct, gate, ramp, and execute one primitive action."""
    (
        executed_action,
        corrected_action,
        correction_metrics,
    ) = runtime._select_primitive_action(observation, base_action)
    environment_result = env.step(executed_action)
    return (
        environment_result,
        executed_action,
        corrected_action,
        correction_metrics,
    )


def shadow_step_environment(
    runtime,
    env,
    observation,
    base_action,
):
    """Compatibility wrapper for Bridge shadow-mode callers."""
    if runtime.config.apply_correction:
        raise ValueError(
            "shadow_step_environment requires apply_correction=False; "
            "use bridge_step_environment for gated execution."
        )
    return bridge_step_environment(
        runtime,
        env,
        observation,
        base_action,
    )
