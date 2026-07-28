"""Dynamics-model bridge for bounded primitive-action corrections.

This module is intentionally independent from QAM action execution.  It learns
offline and recent-online observation-delta models and exposes their raw-space
predictions.  Action correction is added through the same public bridge state.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import numbers
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.flax_utils import TrainState, nonpytree_field


@dataclass(frozen=True)
class DynamicsShiftBridgeConfig:
    """Construction, training, normalization, and correction settings."""

    hidden_dim: int = 256
    num_hidden_layers: int = 2
    learning_rate: float = 3e-4
    clip_grad_norm: float = 10.0
    correction_steps: int = 10
    correction_step_size: float = 0.1
    dynamics_match_weight: float = 1.0
    action_l2_weight: float = 0.01
    max_residual: Any = 0.3
    action_low: Any = -1.0
    action_high: Any = 1.0
    observation_mean: Any = None
    observation_std: Any = None
    action_mean: Any = None
    action_std: Any = None
    delta_mean: Any = None
    delta_std: Any = None
    normalization_epsilon: float = 1e-6


class DynamicsModel(nn.Module):
    """Tanh MLP that predicts a normalized flattened observation delta."""

    observation_dim: int
    hidden_dim: int = 256
    num_hidden_layers: int = 2

    @nn.compact
    def __call__(self, observations, actions):
        observations = jnp.asarray(observations, dtype=jnp.float32)
        actions = jnp.asarray(actions, dtype=jnp.float32)
        if observations.ndim < 2 or actions.ndim < 2:
            raise ValueError(
                "DynamicsModel inputs must include batch and feature axes."
            )
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                "DynamicsModel observations and actions must have the same "
                f"batch size; got {observations.shape[0]} and "
                f"{actions.shape[0]}."
            )
        observations = observations.reshape(observations.shape[0], -1)
        actions = actions.reshape(actions.shape[0], -1)
        hidden = jnp.concatenate((observations, actions), axis=-1)
        for layer_index in range(self.num_hidden_layers):
            hidden = nn.Dense(
                self.hidden_dim,
                name=f"hidden_{layer_index}",
            )(hidden)
            hidden = jnp.tanh(hidden)
        return nn.Dense(
            self.observation_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="delta",
        )(hidden)


def _validated_config(config):
    if config is None:
        config = DynamicsShiftBridgeConfig()
    elif isinstance(config, Mapping):
        try:
            config = DynamicsShiftBridgeConfig(**dict(config))
        except TypeError as exc:
            raise ValueError(f"invalid dynamics bridge config: {exc}") from exc
    if not isinstance(config, DynamicsShiftBridgeConfig):
        raise ValueError(
            "config must be a DynamicsShiftBridgeConfig or mapping; "
            f"got {type(config).__name__}."
        )

    integer_fields = {
        "hidden_dim": (config.hidden_dim, 1),
        "num_hidden_layers": (config.num_hidden_layers, 1),
        "correction_steps": (config.correction_steps, 0),
    }
    validated = {}
    for name, (value, minimum) in integer_fields.items():
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
        validated[name] = value

    positive_fields = (
        "learning_rate",
        "clip_grad_norm",
        "normalization_epsilon",
    )
    nonnegative_fields = (
        "correction_step_size",
        "dynamics_match_weight",
        "action_l2_weight",
    )
    for name in positive_fields + nonnegative_fields:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            qualifier = "positive" if name in positive_fields else "non-negative"
            raise ValueError(
                f"{name} must be a finite {qualifier} number; got {value!r}."
            )
        value = float(value)
        if (
            not np.isfinite(value)
            or (name in positive_fields and value <= 0.0)
            or (name in nonnegative_fields and value < 0.0)
        ):
            qualifier = "positive" if name in positive_fields else "non-negative"
            raise ValueError(
                f"{name} must be a finite {qualifier} number; got {value!r}."
            )
        validated[name] = value
    return config, flax.core.FrozenDict(validated)


def _finite_float32_array(value, name):
    if isinstance(value, jax.core.Tracer):
        dtype = np.dtype(value.dtype)
        if (
            dtype.hasobject
            or not np.issubdtype(dtype, np.number)
            or np.issubdtype(dtype, np.complexfloating)
        ):
            raise ValueError(
                f"{name} must have a real numeric dtype; got {dtype}."
            )
        return jnp.asarray(value, dtype=jnp.float32)
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} cannot be converted to an array: {exc}") from exc
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must have a real numeric dtype; got {array.dtype}.")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be real-valued; got {array.dtype}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} cannot be represented as finite float32 values.")
    return array


def _metadata_array(value, shape, default, name):
    if value is None:
        value = default
    array = _finite_float32_array(value, name)
    if array.ndim == 0:
        array = np.full(shape, array.item(), dtype=np.float32)
    elif array.shape != shape:
        raise ValueError(
            f"{name} must be scalar or have shape {shape}; got {array.shape}."
        )
    return jnp.asarray(array.reshape(-1), dtype=jnp.float32)


def _example_batch(array, name):
    array = _finite_float32_array(array, name)
    if array.ndim < 2:
        raise ValueError(
            f"{name} must include batch and feature axes; got shape {array.shape}."
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} batch must not be empty.")
    return array


class DynamicsShiftBridge(flax.struct.PyTreeNode):
    """Offline/online dynamics state and bounded action-correction interface."""

    offline_model: TrainState
    online_model: TrainState
    observation_mean: Any
    observation_scale: Any
    action_mean: Any
    action_scale: Any
    delta_mean: Any
    delta_scale: Any
    action_low: Any
    action_high: Any
    max_residual: Any
    observation_shape: Any = nonpytree_field()
    action_shape: Any = nonpytree_field()
    config: Any = nonpytree_field()

    @classmethod
    def create(
        cls,
        seed,
        example_observations,
        example_actions,
        config=None,
    ):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an integer; got {seed!r}.")
        config_object, scalar_config = _validated_config(config)
        observations = _example_batch(
            example_observations, "example_observations"
        )
        actions = _example_batch(example_actions, "example_actions")
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                "example observations and actions must have the same batch "
                f"size; got {observations.shape[0]} and {actions.shape[0]}."
            )

        observation_shape = observations.shape[1:]
        action_shape = actions.shape[1:]
        observation_dim = int(np.prod(observation_shape))
        action_dim = int(np.prod(action_shape))
        epsilon = scalar_config["normalization_epsilon"]

        observation_mean = _metadata_array(
            config_object.observation_mean,
            observation_shape,
            0.0,
            "observation_mean",
        )
        observation_std = _metadata_array(
            config_object.observation_std,
            observation_shape,
            1.0,
            "observation_std",
        )
        action_mean = _metadata_array(
            config_object.action_mean,
            action_shape,
            0.0,
            "action_mean",
        )
        action_std = _metadata_array(
            config_object.action_std,
            action_shape,
            1.0,
            "action_std",
        )
        delta_mean = _metadata_array(
            config_object.delta_mean,
            observation_shape,
            0.0,
            "delta_mean",
        )
        delta_std = _metadata_array(
            config_object.delta_std,
            observation_shape,
            1.0,
            "delta_std",
        )
        if np.any(np.asarray(observation_std) < 0.0):
            raise ValueError("observation_std must be non-negative.")
        if np.any(np.asarray(action_std) < 0.0):
            raise ValueError("action_std must be non-negative.")
        if np.any(np.asarray(delta_std) < 0.0):
            raise ValueError("delta_std must be non-negative.")
        observation_scale = jnp.maximum(
            observation_std, epsilon
        ).astype(jnp.float32)
        action_scale = jnp.maximum(
            action_std, epsilon
        ).astype(jnp.float32)
        delta_scale = jnp.maximum(
            delta_std, epsilon
        ).astype(jnp.float32)

        action_low = _metadata_array(
            config_object.action_low,
            action_shape,
            -1.0,
            "action_low",
        )
        action_high = _metadata_array(
            config_object.action_high,
            action_shape,
            1.0,
            "action_high",
        )
        max_residual = _metadata_array(
            config_object.max_residual,
            action_shape,
            0.3,
            "max_residual",
        )
        if np.any(np.asarray(action_low) >= np.asarray(action_high)):
            raise ValueError("action_low must be strictly less than action_high.")
        if np.any(np.asarray(max_residual) < 0.0):
            raise ValueError("max_residual must be non-negative.")

        model = DynamicsModel(
            observation_dim=observation_dim,
            hidden_dim=scalar_config["hidden_dim"],
            num_hidden_layers=scalar_config["num_hidden_layers"],
        )
        normalized_observations = jnp.zeros(
            (observations.shape[0], observation_dim), dtype=jnp.float32
        )
        normalized_actions = jnp.zeros(
            (actions.shape[0], action_dim), dtype=jnp.float32
        )
        rng = jax.random.PRNGKey(int(seed))
        offline_rng, online_rng = jax.random.split(rng)
        offline_params = model.init(
            offline_rng, normalized_observations, normalized_actions
        )["params"]
        online_params = model.init(
            online_rng, normalized_observations, normalized_actions
        )["params"]
        optimizer = optax.chain(
            optax.clip_by_global_norm(scalar_config["clip_grad_norm"]),
            optax.adam(scalar_config["learning_rate"]),
        )
        return cls(
            offline_model=TrainState.create(
                model_def=model, params=offline_params, tx=optimizer
            ),
            online_model=TrainState.create(
                model_def=model, params=online_params, tx=optimizer
            ),
            observation_mean=observation_mean,
            observation_scale=observation_scale,
            action_mean=action_mean,
            action_scale=action_scale,
            delta_mean=delta_mean,
            delta_scale=delta_scale,
            action_low=action_low,
            action_high=action_high,
            max_residual=max_residual,
            observation_shape=tuple(observation_shape),
            action_shape=tuple(action_shape),
            config=scalar_config,
        )

    def _prepare_prediction_inputs(self, observations, actions):
        observations = _finite_float32_array(observations, "observations")
        actions = _finite_float32_array(actions, "actions")
        observation_was_single = observations.shape == self.observation_shape
        action_was_single = actions.shape == self.action_shape
        if observation_was_single != action_was_single:
            raise ValueError(
                "observations and actions must both be single examples or "
                "both be batches."
            )
        if observation_was_single:
            observations = observations[None]
            actions = actions[None]
        else:
            expected_observation_rank = len(self.observation_shape) + 1
            expected_action_rank = len(self.action_shape) + 1
            if (
                observations.ndim != expected_observation_rank
                or observations.shape[1:] != self.observation_shape
            ):
                raise ValueError(
                    "observations must have shape "
                    f"{self.observation_shape} or (batch, "
                    f"{', '.join(map(str, self.observation_shape))}); "
                    f"got {observations.shape}."
                )
            if (
                actions.ndim != expected_action_rank
                or actions.shape[1:] != self.action_shape
            ):
                raise ValueError(
                    "actions must have shape "
                    f"{self.action_shape} or a matching batch shape; "
                    f"got {actions.shape}."
                )
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                "observations and actions must have the same batch size; "
                f"got {observations.shape[0]} and {actions.shape[0]}."
            )
        if observations.shape[0] == 0:
            raise ValueError("prediction batch must not be empty.")
        observation_batch = jnp.asarray(
            observations.reshape(observations.shape[0], -1),
            dtype=jnp.float32,
        )
        action_batch = jnp.asarray(
            actions.reshape(actions.shape[0], -1), dtype=jnp.float32
        )
        return observation_batch, action_batch, observation_was_single

    def _predict(self, model, observations, actions):
        observations, actions, was_single = self._prepare_prediction_inputs(
            observations, actions
        )
        normalized_observations = (
            observations - self.observation_mean
        ) / self.observation_scale
        normalized_actions = (
            actions - self.action_mean
        ) / self.action_scale
        normalized_delta = model(
            normalized_observations, normalized_actions
        )
        raw_delta = (
            normalized_delta * self.delta_scale + self.delta_mean
        ).astype(jnp.float32)
        output_shape = (raw_delta.shape[0],) + self.observation_shape
        raw_delta = raw_delta.reshape(output_shape)
        return raw_delta[0] if was_single else raw_delta

    def predict_offline(self, observations, actions):
        """Predict raw observation deltas with the frozen offline model."""
        return self._predict(self.offline_model, observations, actions)

    def predict_online(self, observations, actions):
        """Predict raw observation deltas with the adaptable online model."""
        return self._predict(self.online_model, observations, actions)

    def _prepare_transition_batch(self, batch):
        if not isinstance(batch, Mapping):
            raise ValueError(
                "transition batch must be a mapping; "
                f"got {type(batch).__name__}."
            )
        required = ("observations", "actions", "next_observations")
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError(
                f"transition batch is missing required fields: {missing}."
            )
        observations = _finite_float32_array(
            batch["observations"], "observations"
        )
        actions = _finite_float32_array(batch["actions"], "actions")
        next_observations = _finite_float32_array(
            batch["next_observations"], "next_observations"
        )
        expected_observation_rank = len(self.observation_shape) + 1
        expected_action_rank = len(self.action_shape) + 1
        if (
            observations.ndim != expected_observation_rank
            or observations.shape[1:] != self.observation_shape
        ):
            raise ValueError(
                "observations must have batched shape (batch, "
                f"{self.observation_shape}); got {observations.shape}."
            )
        if next_observations.shape != observations.shape:
            raise ValueError(
                "next_observations must exactly match observations shape; "
                f"got {next_observations.shape} and {observations.shape}."
            )
        if (
            actions.ndim != expected_action_rank
            or actions.shape[1:] != self.action_shape
        ):
            raise ValueError(
                "actions must have batched shape (batch, "
                f"{self.action_shape}); got {actions.shape}."
            )
        batch_sizes = {
            "observations": observations.shape[0],
            "actions": actions.shape[0],
            "next_observations": next_observations.shape[0],
        }
        if len(set(batch_sizes.values())) != 1:
            raise ValueError(
                "transition fields must have the same batch size; "
                f"got {batch_sizes}."
            )
        if observations.shape[0] == 0:
            raise ValueError("transition batch must not be empty.")

        observations = jnp.asarray(
            observations.reshape(observations.shape[0], -1),
            dtype=jnp.float32,
        )
        actions = jnp.asarray(
            actions.reshape(actions.shape[0], -1), dtype=jnp.float32
        )
        next_observations = jnp.asarray(
            next_observations.reshape(next_observations.shape[0], -1),
            dtype=jnp.float32,
        )
        raw_delta = (next_observations - observations).astype(jnp.float32)
        normalized_observations = (
            observations - self.observation_mean
        ) / self.observation_scale
        normalized_actions = (
            actions - self.action_mean
        ) / self.action_scale
        normalized_delta = (
            raw_delta - self.delta_mean
        ) / self.delta_scale
        return (
            normalized_observations.astype(jnp.float32),
            normalized_actions.astype(jnp.float32),
            normalized_delta.astype(jnp.float32),
        )

    def _prediction_metrics(
        self,
        network,
        params,
        observations,
        actions,
        target_delta,
    ):
        prediction = network(
            observations, actions, params=params
        ).astype(jnp.float32)
        normalized_prediction_mse = jnp.mean(
            jnp.square(prediction - target_delta)
        )
        raw_prediction = prediction * self.delta_scale + self.delta_mean
        raw_target = target_delta * self.delta_scale + self.delta_mean
        raw_prediction_mse = jnp.mean(
            jnp.square(raw_prediction - raw_target)
        )
        return {
            "loss": normalized_prediction_mse,
            "normalized_prediction_mse": normalized_prediction_mse,
            "raw_prediction_mse": raw_prediction_mse,
            "prediction_abs_mean": jnp.mean(jnp.abs(raw_prediction)),
            "target_abs_mean": jnp.mean(jnp.abs(raw_target)),
        }

    @jax.jit
    def _update_offline_packed(
        self, observations, actions, target_delta
    ):
        def loss_fn(params):
            metrics = self._prediction_metrics(
                self.offline_model,
                params,
                observations,
                actions,
                target_delta,
            )
            return metrics["loss"], metrics

        (_, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.offline_model.params)
        offline_model = self.offline_model.apply_gradients(grads=grads)
        return self.replace(offline_model=offline_model), metrics

    def update_offline(self, batch):
        """Update only the offline model on normalized observation deltas."""
        prepared = self._prepare_transition_batch(batch)
        return self._update_offline_packed(*prepared)

    def synchronize_online_from_offline(self):
        """Copy offline parameters and reset the online optimizer state."""
        online_params = jax.tree_util.tree_map(
            lambda value: value + jnp.zeros_like(value),
            self.offline_model.params,
        )
        online_model = TrainState.create(
            model_def=self.online_model.model_def,
            params=online_params,
            tx=self.online_model.tx,
        )
        return self.replace(online_model=online_model)

    @jax.jit
    def _update_online_packed(
        self, observations, actions, target_delta
    ):
        def loss_fn(params):
            metrics = self._prediction_metrics(
                self.online_model,
                params,
                observations,
                actions,
                target_delta,
            )
            return metrics["loss"], metrics

        (_, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.online_model.params)
        online_model = self.online_model.apply_gradients(grads=grads)
        return self.replace(online_model=online_model), metrics

    def update_online(self, batch):
        """Update only the online model on normalized observation deltas."""
        prepared = self._prepare_transition_batch(batch)
        return self._update_online_packed(*prepared)

    @jax.jit
    def _evaluate_packed(
        self, model, observations, actions, target_delta
    ):
        metrics = self._prediction_metrics(
            model,
            model.params,
            observations,
            actions,
            target_delta,
        )
        return {
            "normalized_mse": metrics["normalized_prediction_mse"],
            "raw_mse": metrics["raw_prediction_mse"],
            "prediction_abs_mean": metrics["prediction_abs_mean"],
            "target_abs_mean": metrics["target_abs_mean"],
        }

    def evaluate_offline(self, batch):
        """Evaluate the offline model without updating any bridge state."""
        prepared = self._prepare_transition_batch(batch)
        return self._evaluate_packed(self.offline_model, *prepared)

    def evaluate_online(self, batch):
        """Evaluate the online model without updating any bridge state."""
        prepared = self._prepare_transition_batch(batch)
        return self._evaluate_packed(self.online_model, *prepared)

    def _prepare_correction_inputs(self, observations, base_actions):
        observations = jnp.asarray(
            _finite_float32_array(observations, "observations"),
            dtype=jnp.float32,
        )
        base_actions = jnp.asarray(
            _finite_float32_array(base_actions, "base_actions"),
            dtype=jnp.float32,
        )
        observation_was_single = (
            tuple(observations.shape) == self.observation_shape
        )
        action_was_single = tuple(base_actions.shape) == self.action_shape
        if observation_was_single != action_was_single:
            raise ValueError(
                "observations and base_actions must both be single examples "
                "or both be batches."
            )
        if observation_was_single:
            observations = observations[None]
            base_actions = base_actions[None]
        else:
            if (
                observations.ndim != len(self.observation_shape) + 1
                or tuple(observations.shape[1:])
                != self.observation_shape
            ):
                raise ValueError(
                    "observations must have shape "
                    f"{self.observation_shape} or a matching batch shape; "
                    f"got {observations.shape}."
                )
            if (
                base_actions.ndim != len(self.action_shape) + 1
                or tuple(base_actions.shape[1:]) != self.action_shape
            ):
                raise ValueError(
                    "base_actions must have shape "
                    f"{self.action_shape} or a matching batch shape; "
                    f"got {base_actions.shape}."
                )
        if observations.shape[0] != base_actions.shape[0]:
            raise ValueError(
                "observations and base_actions must have the same batch "
                f"size; got {observations.shape[0]} and "
                f"{base_actions.shape[0]}."
            )
        if observations.shape[0] == 0:
            raise ValueError("correction batch must not be empty.")

        observations = observations.reshape(observations.shape[0], -1)
        base_actions = base_actions.reshape(base_actions.shape[0], -1)
        if not isinstance(base_actions, jax.core.Tracer):
            base_actions_host = np.asarray(base_actions)
            action_low_host = np.asarray(self.action_low)
            action_high_host = np.asarray(self.action_high)
            if np.any(base_actions_host < action_low_host) or np.any(
                base_actions_host > action_high_host
            ):
                raise ValueError(
                    "base_actions must already lie within action bounds so "
                    "zero-step correction can remain an exact no-op."
                )
        return observations, base_actions, observation_was_single

    def _normalized_model_prediction(
        self, network, observations, actions
    ):
        normalized_observations = (
            observations - self.observation_mean
        ) / self.observation_scale
        normalized_actions = (
            actions - self.action_mean
        ) / self.action_scale
        return network(
            normalized_observations.astype(jnp.float32),
            normalized_actions.astype(jnp.float32),
        ).astype(jnp.float32)

    def _match_metrics(
        self, target_delta, pre_prediction, post_prediction
    ):
        pre_match_mse = jnp.mean(
            jnp.square(pre_prediction - target_delta)
        )
        post_match_mse = jnp.mean(
            jnp.square(post_prediction - target_delta)
        )
        raw_target = target_delta * self.delta_scale + self.delta_mean
        raw_pre_prediction = (
            pre_prediction * self.delta_scale + self.delta_mean
        )
        raw_post_prediction = (
            post_prediction * self.delta_scale + self.delta_mean
        )
        pre_match_mse_raw = jnp.mean(
            jnp.square(raw_pre_prediction - raw_target)
        )
        post_match_mse_raw = jnp.mean(
            jnp.square(raw_post_prediction - raw_target)
        )
        return {
            "pre_match_mse": pre_match_mse,
            "post_match_mse": post_match_mse,
            "match_improvement": pre_match_mse - post_match_mse,
            "pre_match_mse_normalized": pre_match_mse,
            "post_match_mse_normalized": post_match_mse,
            "match_improvement_normalized": (
                pre_match_mse - post_match_mse
            ),
            "pre_match_mse_raw": pre_match_mse_raw,
            "post_match_mse_raw": post_match_mse_raw,
            "match_improvement_raw": (
                pre_match_mse_raw - post_match_mse_raw
            ),
        }

    @jax.jit
    def _correct_actions_packed(self, observations, base_actions):
        target_delta = jax.lax.stop_gradient(
            self._normalized_model_prediction(
                self.offline_model, observations, base_actions
            )
        )
        pre_prediction = self._normalized_model_prediction(
            self.online_model, observations, base_actions
        )

        def objective(actions):
            prediction = self._normalized_model_prediction(
                self.online_model, observations, actions
            )
            match_per_example = jnp.mean(
                jnp.square(prediction - target_delta), axis=-1
            )
            action_l2_per_example = jnp.mean(
                jnp.square(actions - base_actions), axis=-1
            )
            return jnp.sum(
                self.config["dynamics_match_weight"]
                * match_per_example
                + self.config["action_l2_weight"]
                * action_l2_per_example
            )

        def correction_step(_, carry):
            actions, action_clip_sum, residual_clip_sum = carry
            gradient = jax.grad(objective)(actions)
            gradient = jnp.nan_to_num(
                gradient, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(jnp.float32)
            proposed_actions = (
                actions
                - self.config["correction_step_size"] * gradient
            )
            proposed_residual = proposed_actions - base_actions
            clipped_residual = jnp.clip(
                proposed_residual,
                -self.max_residual,
                self.max_residual,
            )
            residual_clip_fraction = jnp.mean(
                proposed_residual != clipped_residual
            )
            residual_bounded_actions = base_actions + clipped_residual
            clipped_actions = jnp.clip(
                residual_bounded_actions,
                self.action_low,
                self.action_high,
            )
            action_clip_fraction = jnp.mean(
                residual_bounded_actions != clipped_actions
            )
            return (
                clipped_actions.astype(jnp.float32),
                action_clip_sum + action_clip_fraction,
                residual_clip_sum + residual_clip_fraction,
            )

        corrected_actions, action_clip_sum, residual_clip_sum = (
            jax.lax.fori_loop(
                0,
                self.config["correction_steps"],
                correction_step,
                (
                    base_actions,
                    jnp.asarray(0.0, dtype=jnp.float32),
                    jnp.asarray(0.0, dtype=jnp.float32),
                ),
            )
        )
        post_prediction = self._normalized_model_prediction(
            self.online_model, observations, corrected_actions
        )
        exact_noop = jnp.asarray(
            self.config["correction_steps"] == 0
            or self.config["correction_step_size"] == 0.0
            or self.config["dynamics_match_weight"] == 0.0,
            dtype=jnp.bool_,
        ) | jnp.all(self.max_residual == 0.0)
        post_prediction = jnp.where(
            exact_noop, pre_prediction, post_prediction
        )
        residual = corrected_actions - base_actions
        step_denominator = jnp.asarray(
            max(self.config["correction_steps"], 1),
            dtype=jnp.float32,
        )
        match_metrics = self._match_metrics(
            target_delta, pre_prediction, post_prediction
        )
        for suffix in ("", "_normalized", "_raw"):
            pre_name = f"pre_match_mse{suffix}"
            post_name = f"post_match_mse{suffix}"
            improvement_name = f"match_improvement{suffix}"
            match_metrics[post_name] = jnp.where(
                exact_noop,
                match_metrics[pre_name],
                match_metrics[post_name],
            )
            match_metrics[improvement_name] = jnp.where(
                exact_noop,
                jnp.asarray(0.0, dtype=jnp.float32),
                match_metrics[improvement_name],
            )
        metrics = {
            **match_metrics,
            "residual_l2_mean": jnp.mean(
                jnp.sqrt(jnp.sum(jnp.square(residual), axis=-1))
            ),
            "residual_abs_max": jnp.max(jnp.abs(residual)),
            "action_clip_fraction": action_clip_sum / step_denominator,
            "residual_clip_fraction": (
                residual_clip_sum / step_denominator
            ),
        }
        return corrected_actions.astype(jnp.float32), metrics

    def correct_actions(self, observations, base_actions):
        """Return bounded actions matching offline deltas under online dynamics.

        ``pre_match_mse`` and ``post_match_mse`` are measured in normalized
        delta space.  Their ``*_raw`` counterparts are measured in raw
        observation-delta units.  Each clip fraction is the fraction of scalar
        action components clipped per step, averaged over correction steps.
        """
        observations, base_actions, was_single = (
            self._prepare_correction_inputs(
                observations, base_actions
            )
        )
        corrected_actions, metrics = self._correct_actions_packed(
            observations, base_actions
        )
        corrected_actions = corrected_actions.reshape(
            (corrected_actions.shape[0],) + self.action_shape
        )
        if was_single:
            corrected_actions = corrected_actions[0]
        return corrected_actions, metrics
