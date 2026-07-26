"""Episode-boundary checkpointing for online training."""

import errno
from collections.abc import Mapping
import numbers
import os
import pickle
import random
import tempfile

import flax
import jax
import jax.numpy as jnp
import numpy as np

from utils.recent_dynamics_buffer import RecentDynamicsBuffer
from utils.transition_occupancy import TransitionOccupancyDetector


FORMAT_VERSION = 3
RECENT_DYNAMICS_FORMAT_VERSION = 2
LEGACY_FORMAT_VERSION = 1
CHECKPOINT_FILENAME = "online_checkpoint.pkl"
PROGRESS_FILENAME = "progress.tk"
OCCUPANCY_CONFIG_FIELDS = (
    "hidden_dim",
    "num_hidden_layers",
    "learning_rate",
    "clip_grad_norm",
    "batch_size",
    "start_size",
    "update_interval",
    "updates_per_interval",
)


class OnlineCheckpointError(RuntimeError):
    """Raised when an online checkpoint cannot be safely saved or restored."""


def _validate_recent_dynamics_capacity(capacity, field_name):
    if isinstance(capacity, bool) or not isinstance(
        capacity, (int, np.integer)
    ):
        raise OnlineCheckpointError(
            f"{field_name} must be a non-negative integer; got {capacity!r}."
        )
    capacity = int(capacity)
    if capacity < 0:
        raise OnlineCheckpointError(
            f"{field_name} must be a non-negative integer; got {capacity!r}."
        )
    return capacity


def should_save_online_checkpoint(
    online_step,
    online_save_interval,
    last_saved_online_step,
    done,
    action_queue,
):
    """Return whether an interval is due at a safe episode boundary."""
    if online_save_interval < 0:
        raise ValueError(
            "online_save_interval must be a non-negative integer; "
            f"got {online_save_interval!r}."
        )
    if online_save_interval == 0 or not done or len(action_queue) != 0:
        return False
    next_due_step = (
        last_saved_online_step // online_save_interval + 1
    ) * online_save_interval
    return online_step >= next_due_step


def online_start_step(checkpoint):
    """Return the first online step that has not already been completed."""
    online_step = checkpoint.get("online_step")
    if not isinstance(online_step, int) or online_step < 0:
        raise OnlineCheckpointError(
            f"Checkpoint online_step must be a non-negative integer; got {online_step!r}."
        )
    return online_step + 1


def read_progress(save_dir):
    """Read progress.tk, returning ``(None, None)`` when it does not exist."""
    progress_path = os.path.join(save_dir, PROGRESS_FILENAME)
    if not os.path.exists(progress_path):
        return None, None
    try:
        with open(progress_path, "r") as file:
            progress = file.read().strip()
        parts = progress.split(",")
        if len(parts) != 2:
            raise ValueError("expected '<stage>,<step>'")
        stage, step_text = parts
        if stage not in ("offline", "online"):
            raise ValueError(f"unknown stage {stage!r}")
        step = int(step_text)
        if step < 0:
            raise ValueError(f"step must be non-negative; got {step}")
    except (OSError, ValueError) as exc:
        raise OnlineCheckpointError(
            f"Invalid progress file {progress_path}: {exc}"
        ) from exc
    return stage, step


def _fsync(file):
    try:
        os.fsync(file.fileno())
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise


def _atomic_pickle_dump(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=os.path.dirname(path),
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            _fsync(file)
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=os.path.dirname(path),
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as file:
            file.write(text)
            file.flush()
            _fsync(file)
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _validate_replay_bounds(pointer, size, max_size):
    if not isinstance(max_size, int) or max_size <= 0:
        raise OnlineCheckpointError(
            f"Replay Buffer max_size must be a positive integer; got {max_size!r}."
        )
    if not isinstance(pointer, int) or not 0 <= pointer < max_size:
        raise OnlineCheckpointError(
            f"Replay Buffer pointer {pointer!r} is outside [0, {max_size})."
        )
    if not isinstance(size, int) or not 0 <= size <= max_size:
        raise OnlineCheckpointError(
            f"Replay Buffer size {size!r} is outside [0, {max_size}]."
        )


def _validate_replay_layout(
    *,
    pointer,
    size,
    max_size,
    data_start,
    data_count,
    online_step,
    balanced_sampling,
    initial_replay_size,
):
    """Validate Replay Buffer metadata before serializing or restoring data."""
    _validate_replay_bounds(pointer, size, max_size)
    if not isinstance(online_step, int) or online_step < 0:
        raise OnlineCheckpointError(
            f"Replay Buffer online_step must be a non-negative integer; "
            f"got {online_step!r}."
        )
    if not isinstance(data_count, int) or data_count < 0:
        raise OnlineCheckpointError(
            f"Replay Buffer data_count must be a non-negative integer; "
            f"got {data_count!r}."
        )
    if data_count != online_step:
        raise OnlineCheckpointError(
            f"Replay Buffer data_count {data_count} does not match "
            f"online_step {online_step}."
        )

    expected_start = 0 if balanced_sampling else initial_replay_size
    if data_start != expected_start:
        raise OnlineCheckpointError(
            f"Replay Buffer data_start {data_start!r} does not match "
            f"expected data_start {expected_start!r}."
        )

    data_end = data_start + data_count
    if data_end > max_size:
        raise OnlineCheckpointError(
            "Replay Buffer data slice is outside the buffer: "
            f"data_start={data_start}, data_count={data_count}, "
            f"max_size={max_size}."
        )

    expected_pointer = data_end % max_size
    if pointer != expected_pointer:
        raise OnlineCheckpointError(
            f"Replay Buffer pointer {pointer} does not match expected pointer "
            f"{expected_pointer} for data_start={data_start}, "
            f"data_count={data_count}, and max_size={max_size}."
        )

    if data_end < max_size:
        allowed_sizes = {data_end}
    else:
        # ReplayBuffer.add_transition updates size from the wrapped pointer. On
        # the insertion that exactly fills the buffer, pointer becomes zero and
        # the current implementation may retain max_size - 1. Accept only that
        # compatibility value or a conventional fully populated max_size.
        allowed_sizes = {max_size - 1, max_size}
    if size not in allowed_sizes:
        raise OnlineCheckpointError(
            f"Replay Buffer size {size} is inconsistent with data_start="
            f"{data_start}, data_count={data_count}, and max_size={max_size}; "
            f"expected one of {sorted(allowed_sizes)}."
        )


def _serialize_replay_buffer(
    replay_buffer,
    online_step,
    balanced_sampling,
    initial_replay_size,
):
    pointer = int(replay_buffer.pointer)
    size = int(replay_buffer.size)
    max_size = int(replay_buffer.max_size)
    online_step = int(online_step)
    data_start = 0 if balanced_sampling else initial_replay_size
    data_count = online_step
    _validate_replay_layout(
        pointer=pointer,
        size=size,
        max_size=max_size,
        data_start=data_start,
        data_count=data_count,
        online_step=online_step,
        balanced_sampling=balanced_sampling,
        initial_replay_size=initial_replay_size,
    )

    online_data = {
        key: np.array(value[data_start : data_start + data_count], copy=True)
        for key, value in replay_buffer.items()
    }
    return {
        "pointer": pointer,
        "size": size,
        "max_size": max_size,
        "data_start": data_start,
        "data_count": data_count,
        "online_data": online_data,
    }


def _validate_recent_state_without_template(state, capacity, online_step):
    """Validate recent state without allocating another capacity-sized copy."""
    if not isinstance(state, dict):
        raise OnlineCheckpointError(
            "Checkpoint recent_dynamics_buffer must be a dictionary when "
            f"recent_dynamics_capacity={capacity}; got "
            f"{type(state).__name__}."
        )
    data = state.get("data")
    if not isinstance(data, dict) or not data:
        raise OnlineCheckpointError(
            "Checkpoint recent_dynamics_buffer data must be a non-empty "
            f"dictionary; got {type(data).__name__}."
        )

    template_data = {}
    for key, value in data.items():
        try:
            saved = np.asarray(value)
        except Exception as exc:
            raise OnlineCheckpointError(
                f"Checkpoint recent_dynamics_buffer field {key!r} cannot "
                f"be converted to a NumPy array: {exc}"
            ) from exc
        if saved.ndim == 0 or saved.shape[0] != capacity:
            raise OnlineCheckpointError(
                f"Checkpoint recent_dynamics_buffer field {key!r} shape "
                f"{saved.shape} must start with capacity {capacity}."
            )
        if saved.dtype.hasobject:
            raise OnlineCheckpointError(
                f"Checkpoint recent_dynamics_buffer field {key!r} has "
                f"unsupported object dtype {saved.dtype}."
            )
        single_item = np.empty((1, *saved.shape[1:]), dtype=saved.dtype)
        template_data[key] = np.broadcast_to(single_item, saved.shape)

    validation_buffer = RecentDynamicsBuffer(
        data=template_data, capacity=capacity
    )
    try:
        validation_buffer.validate_state_dict(state)
    except (TypeError, ValueError) as exc:
        raise OnlineCheckpointError(
            f"Invalid recent_dynamics_buffer state: {exc}"
        ) from exc
    saved_total_added = int(state["total_added"])
    if saved_total_added != online_step:
        raise OnlineCheckpointError(
            "Checkpoint recent_dynamics_buffer total_added "
            f"{saved_total_added} does not match online_step "
            f"{online_step}."
        )


def _serialize_recent_dynamics_buffer(
    recent_dynamics_buffer,
    recent_dynamics_capacity,
    online_step,
):
    capacity = _validate_recent_dynamics_capacity(
        recent_dynamics_capacity, "recent_dynamics_capacity"
    )
    if capacity == 0:
        if recent_dynamics_buffer is not None:
            raise OnlineCheckpointError(
                "recent_dynamics_buffer must be None when "
                "recent_dynamics_capacity=0."
            )
        return None

    if not isinstance(recent_dynamics_buffer, RecentDynamicsBuffer):
        raise OnlineCheckpointError(
            "recent_dynamics_buffer must be a RecentDynamicsBuffer when "
            f"recent_dynamics_capacity={capacity}; got "
            f"{type(recent_dynamics_buffer).__name__}."
        )
    if recent_dynamics_buffer.capacity != capacity:
        raise OnlineCheckpointError(
            "recent_dynamics_buffer capacity "
            f"{recent_dynamics_buffer.capacity} does not match configured "
            f"recent_dynamics_capacity {capacity}."
        )
    state = recent_dynamics_buffer.state_dict()
    _validate_recent_state_without_template(state, capacity, online_step)
    return state


def _normalize_occupancy_detector_config(config, field_name):
    if not isinstance(config, Mapping):
        raise OnlineCheckpointError(
            f"{field_name} must be a mapping; "
            f"got {type(config).__name__}."
        )
    actual_fields = set(config)
    expected_fields = set(OCCUPANCY_CONFIG_FIELDS)
    if actual_fields != expected_fields:
        raise OnlineCheckpointError(
            f"{field_name} fields must be exactly "
            f"{sorted(expected_fields)}; got {sorted(actual_fields)}."
        )

    normalized = {}
    for name in (
        "hidden_dim",
        "num_hidden_layers",
        "batch_size",
        "start_size",
        "update_interval",
        "updates_per_interval",
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(
            value, (int, np.integer)
        ):
            raise OnlineCheckpointError(
                f"{field_name} {name} must be a positive integer; "
                f"got {value!r}."
            )
        value = int(value)
        if value <= 0:
            raise OnlineCheckpointError(
                f"{field_name} {name} must be a positive integer; "
                f"got {value!r}."
            )
        normalized[name] = value

    for name in ("learning_rate", "clip_grad_norm"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise OnlineCheckpointError(
                f"{field_name} {name} must be a finite positive number; "
                f"got {value!r}."
            )
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise OnlineCheckpointError(
                f"{field_name} {name} must be a finite positive number; "
                f"got {value!r}."
            )
        normalized[name] = value
    return {
        name: normalized[name] for name in OCCUPANCY_CONFIG_FIELDS
    }


def _validate_live_occupancy_detector_config(
    detector, normalized_config, field_name
):
    for name in (
        "hidden_dim",
        "num_hidden_layers",
        "learning_rate",
        "clip_grad_norm",
    ):
        if detector.config[name] != normalized_config[name]:
            raise OnlineCheckpointError(
                f"{field_name} detector config does not match metadata for "
                f"{name}: detector={detector.config[name]!r}, "
                f"metadata={normalized_config[name]!r}."
            )


def _serialize_occupancy_detector(detector, config):
    if detector is None:
        if config is not None:
            raise OnlineCheckpointError(
                "occupancy_detector_config must be None when "
                "occupancy_detector is disabled."
            )
        return False, None, None
    if not isinstance(detector, TransitionOccupancyDetector):
        raise OnlineCheckpointError(
            "occupancy_detector must be a TransitionOccupancyDetector or "
            f"None; got {type(detector).__name__}."
        )
    normalized_config = _normalize_occupancy_detector_config(
        config, "occupancy_detector_config"
    )
    _validate_live_occupancy_detector_config(
        detector,
        normalized_config,
        "occupancy_detector",
    )
    return (
        True,
        normalized_config,
        flax.serialization.to_state_dict(detector),
    )


def save_online_checkpoint(
    save_dir,
    agent,
    replay_buffer,
    online_rng,
    online_step,
    offline_steps,
    balanced_sampling,
    initial_replay_size,
    action_dim,
    horizon_length,
    env_name,
    done,
    action_queue,
    recent_dynamics_buffer=None,
    recent_dynamics_capacity=0,
    occupancy_detector=None,
    occupancy_detector_config=None,
):
    """Atomically save a complete online checkpoint, then update progress.tk."""
    if not done:
        raise OnlineCheckpointError(
            "Online checkpoint requires done=True at an episode boundary."
        )
    if len(action_queue) != 0:
        raise OnlineCheckpointError(
            "Online checkpoint requires an empty action_queue; "
            f"got {len(action_queue)} queued actions."
        )

    (
        occupancy_detector_enabled,
        normalized_occupancy_config,
        occupancy_detector_state,
    ) = _serialize_occupancy_detector(
        occupancy_detector, occupancy_detector_config
    )

    checkpoint = {
        "format_version": FORMAT_VERSION,
        "stage": "online",
        "online_step": int(online_step),
        "global_step": int(offline_steps + online_step),
        "offline_steps": int(offline_steps),
        "agent": flax.serialization.to_state_dict(agent),
        "replay_buffer": _serialize_replay_buffer(
            replay_buffer,
            online_step=online_step,
            balanced_sampling=balanced_sampling,
            initial_replay_size=initial_replay_size,
        ),
        "online_rng": np.array(online_rng, copy=True),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "balanced_sampling": bool(balanced_sampling),
        "initial_replay_size": int(initial_replay_size),
        "action_dim": int(action_dim),
        "horizon_length": int(horizon_length),
        "env_name": str(env_name),
        "recent_dynamics_capacity": _validate_recent_dynamics_capacity(
            recent_dynamics_capacity, "recent_dynamics_capacity"
        ),
        "recent_dynamics_buffer": _serialize_recent_dynamics_buffer(
            recent_dynamics_buffer,
            recent_dynamics_capacity=recent_dynamics_capacity,
            online_step=int(online_step),
        ),
        "occupancy_detector_enabled": occupancy_detector_enabled,
        "occupancy_detector_config": normalized_occupancy_config,
        "occupancy_detector": occupancy_detector_state,
    }

    checkpoint_path = os.path.join(save_dir, CHECKPOINT_FILENAME)
    _atomic_pickle_dump(checkpoint, checkpoint_path)
    _atomic_write_text(
        f"online,{online_step}",
        os.path.join(save_dir, PROGRESS_FILENAME),
    )
    return checkpoint_path


def _load_checkpoint_file(save_dir):
    checkpoint_path = os.path.join(save_dir, CHECKPOINT_FILENAME)
    if not os.path.exists(checkpoint_path):
        raise OnlineCheckpointError(
            f"Online progress requires checkpoint file {checkpoint_path}, but it is missing."
        )
    try:
        with open(checkpoint_path, "rb") as file:
            checkpoint = pickle.load(file)
    except Exception as exc:
        raise OnlineCheckpointError(
            f"Failed to load online checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise OnlineCheckpointError(
            f"Online checkpoint {checkpoint_path} must contain a dictionary."
        )
    return checkpoint


def _validate_flax_state_layout(template, state, description):
    if not isinstance(state, dict):
        raise OnlineCheckpointError(
            f"{description} must be a dictionary; "
            f"got {type(state).__name__}."
        )
    template_state = flax.serialization.to_state_dict(template)
    try:
        template_items, template_tree = (
            jax.tree_util.tree_flatten_with_path(template_state)
        )
        saved_items, saved_tree = jax.tree_util.tree_flatten_with_path(
            state
        )
    except Exception as exc:
        raise OnlineCheckpointError(
            f"Invalid {description} tree structure: {exc}"
        ) from exc
    if saved_tree != template_tree:
        raise OnlineCheckpointError(
            f"{description} tree structure does not match the current "
            "template."
        )

    for (path, template_leaf), (_, saved_leaf) in zip(
        template_items, saved_items
    ):
        path_text = jax.tree_util.keystr(path)
        try:
            template_array = np.asarray(template_leaf)
            saved_array = np.asarray(saved_leaf)
        except Exception as exc:
            raise OnlineCheckpointError(
                f"{description}{path_text} cannot be converted to an array: "
                f"{exc}"
            ) from exc
        if saved_array.shape != template_array.shape:
            raise OnlineCheckpointError(
                f"{description}{path_text} shape mismatch: "
                f"checkpoint={saved_array.shape}, "
                f"expected={template_array.shape}."
            )
        scalar_integer_compatibility = (
            saved_array.shape == ()
            and template_array.shape == ()
            and np.issubdtype(saved_array.dtype, np.integer)
            and np.issubdtype(template_array.dtype, np.integer)
        )
        if (
            saved_array.dtype != template_array.dtype
            and not scalar_integer_compatibility
        ):
            raise OnlineCheckpointError(
                f"{description}{path_text} dtype mismatch: "
                f"checkpoint={saved_array.dtype}, "
                f"expected={template_array.dtype}."
            )
        if (
            np.issubdtype(saved_array.dtype, np.number)
            and not np.all(np.isfinite(saved_array))
        ):
            raise OnlineCheckpointError(
                f"{description}{path_text} contains non-finite values."
            )


def _validate_occupancy_detector_restore(
    checkpoint,
    *,
    format_version,
    occupancy_detector,
    expected_config,
):
    current_enabled = occupancy_detector is not None
    if current_enabled and not isinstance(
        occupancy_detector, TransitionOccupancyDetector
    ):
        raise OnlineCheckpointError(
            "occupancy_detector template must be a "
            f"TransitionOccupancyDetector; got "
            f"{type(occupancy_detector).__name__}."
        )
    if not current_enabled and expected_config is not None:
        raise OnlineCheckpointError(
            "expected_occupancy_detector_config must be None when the "
            "current occupancy detector is disabled."
        )

    if format_version in (
        LEGACY_FORMAT_VERSION,
        RECENT_DYNAMICS_FORMAT_VERSION,
    ):
        if current_enabled:
            raise OnlineCheckpointError(
                f"Online checkpoint format_version {format_version} does "
                "not contain occupancy detector state and cannot be "
                "restored when the current occupancy detector is enabled."
            )
        return

    required_fields = {
        "occupancy_detector_enabled",
        "occupancy_detector_config",
        "occupancy_detector",
    }
    missing = required_fields - set(checkpoint)
    if missing:
        raise OnlineCheckpointError(
            "Online checkpoint is missing version 3 occupancy detector "
            f"fields: {sorted(missing)}."
        )
    saved_enabled = checkpoint["occupancy_detector_enabled"]
    if not isinstance(saved_enabled, bool):
        raise OnlineCheckpointError(
            "Checkpoint occupancy_detector_enabled must be a boolean; "
            f"got {saved_enabled!r}."
        )
    if saved_enabled != current_enabled:
        raise OnlineCheckpointError(
            "Checkpoint occupancy_detector_enabled "
            f"{saved_enabled} does not match current enabled state "
            f"{current_enabled}."
        )

    saved_config = checkpoint["occupancy_detector_config"]
    saved_state = checkpoint["occupancy_detector"]
    if not saved_enabled:
        if saved_config is not None:
            raise OnlineCheckpointError(
                "Checkpoint occupancy_detector_config must be None when "
                "the detector is disabled."
            )
        if saved_state is not None:
            raise OnlineCheckpointError(
                "Checkpoint occupancy_detector must be None when the "
                "detector is disabled."
            )
        return

    normalized_saved_config = _normalize_occupancy_detector_config(
        saved_config, "Checkpoint occupancy_detector_config"
    )
    normalized_expected_config = _normalize_occupancy_detector_config(
        expected_config, "expected_occupancy_detector_config"
    )
    if normalized_saved_config != normalized_expected_config:
        raise OnlineCheckpointError(
            "Checkpoint occupancy_detector_config does not match current "
            f"configuration: checkpoint={normalized_saved_config}, "
            f"expected={normalized_expected_config}."
        )

    _validate_live_occupancy_detector_config(
        occupancy_detector,
        normalized_expected_config,
        "occupancy_detector template",
    )
    _validate_flax_state_layout(
        occupancy_detector,
        saved_state,
        "Checkpoint occupancy_detector",
    )


def _restore_occupancy_detector(occupancy_detector, checkpoint):
    if occupancy_detector is None:
        return None
    try:
        restored = flax.serialization.from_state_dict(
            occupancy_detector, checkpoint["occupancy_detector"]
        )
    except Exception as exc:
        raise OnlineCheckpointError(
            "Failed to restore occupancy detector state from online "
            f"checkpoint: {exc}"
        ) from exc
    step = np.asarray(restored.network.step)
    if (
        step.shape != ()
        or not np.issubdtype(step.dtype, np.integer)
        or int(step) < 1
    ):
        raise OnlineCheckpointError(
            "Restored occupancy detector TrainState step must be a positive "
            f"integer scalar; got {restored.network.step!r}."
        )
    return restored


def _validate_checkpoint_metadata(
    checkpoint,
    expected_env_name,
    expected_horizon_length,
    expected_balanced_sampling,
    expected_initial_replay_size,
    expected_action_dim,
    expected_offline_steps,
    expected_recent_dynamics_capacity,
    expected_online_step,
):
    base_required_fields = {
        "format_version",
        "stage",
        "online_step",
        "global_step",
        "offline_steps",
        "agent",
        "replay_buffer",
        "online_rng",
        "numpy_rng_state",
        "python_rng_state",
        "balanced_sampling",
        "initial_replay_size",
        "action_dim",
        "horizon_length",
        "env_name",
    }
    missing = base_required_fields - set(checkpoint)
    if missing:
        raise OnlineCheckpointError(
            f"Online checkpoint is missing required fields: {sorted(missing)}."
        )
    format_version = checkpoint["format_version"]
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, (int, np.integer))
        or int(format_version)
        not in (
            LEGACY_FORMAT_VERSION,
            RECENT_DYNAMICS_FORMAT_VERSION,
            FORMAT_VERSION,
        )
    ):
        raise OnlineCheckpointError(
            "Unsupported online checkpoint format_version "
            f"{format_version!r}; supported versions are "
            f"{LEGACY_FORMAT_VERSION}, "
            f"{RECENT_DYNAMICS_FORMAT_VERSION}, and {FORMAT_VERSION}."
        )
    format_version = int(format_version)
    if checkpoint["stage"] != "online":
        raise OnlineCheckpointError(
            "Online checkpoint stage must be 'online'; "
            f"got {checkpoint['stage']!r}."
        )
    if checkpoint["env_name"] != expected_env_name:
        raise OnlineCheckpointError(
            f"Checkpoint env_name {checkpoint['env_name']!r} does not match "
            f"{expected_env_name!r}."
        )
    if checkpoint["horizon_length"] != expected_horizon_length:
        raise OnlineCheckpointError(
            "Checkpoint horizon_length "
            f"{checkpoint['horizon_length']!r} does not match "
            f"{expected_horizon_length!r}."
        )
    if checkpoint["balanced_sampling"] != expected_balanced_sampling:
        raise OnlineCheckpointError(
            "Checkpoint balanced_sampling "
            f"{checkpoint['balanced_sampling']!r} does not match "
            f"{expected_balanced_sampling!r}."
        )
    if checkpoint["initial_replay_size"] != expected_initial_replay_size:
        raise OnlineCheckpointError(
            "Checkpoint initial_replay_size "
            f"{checkpoint['initial_replay_size']!r} does not match "
            f"{expected_initial_replay_size!r}."
        )
    if checkpoint["action_dim"] != expected_action_dim:
        raise OnlineCheckpointError(
            f"Checkpoint action_dim {checkpoint['action_dim']!r} does not "
            f"match {expected_action_dim!r}."
        )
    if checkpoint["offline_steps"] != expected_offline_steps:
        raise OnlineCheckpointError(
            f"Checkpoint offline_steps {checkpoint['offline_steps']!r} "
            f"does not match {expected_offline_steps!r}."
        )
    online_step = checkpoint["online_step"]
    global_step = checkpoint["global_step"]
    if not isinstance(online_step, int) or online_step < 0:
        raise OnlineCheckpointError(
            "Checkpoint online_step must be a non-negative integer; "
            f"got {online_step!r}."
        )
    if expected_online_step is not None:
        if (
            isinstance(expected_online_step, bool)
            or not isinstance(expected_online_step, (int, np.integer))
            or int(expected_online_step) < 0
        ):
            raise OnlineCheckpointError(
                "expected_online_step must be a non-negative integer or "
                f"None; got {expected_online_step!r}."
            )
        expected_online_step = int(expected_online_step)
        if online_step != expected_online_step:
            raise OnlineCheckpointError(
                f"progress online step {expected_online_step} does not "
                f"match checkpoint online step {online_step}."
            )
    if global_step != expected_offline_steps + online_step:
        raise OnlineCheckpointError(
            f"Checkpoint global_step {global_step!r} is inconsistent with "
            f"offline_steps={expected_offline_steps} and "
            f"online_step={online_step}."
        )

    expected_recent_dynamics_capacity = _validate_recent_dynamics_capacity(
        expected_recent_dynamics_capacity,
        "expected_recent_dynamics_capacity",
    )
    if format_version == LEGACY_FORMAT_VERSION:
        if expected_recent_dynamics_capacity > 0:
            raise OnlineCheckpointError(
                "Online checkpoint format_version 1 does not contain a "
                "recent_dynamics_buffer and cannot be safely restored when "
                f"expected_recent_dynamics_capacity="
                f"{expected_recent_dynamics_capacity}."
            )
    else:
        recent_fields = {
            "recent_dynamics_capacity",
            "recent_dynamics_buffer",
        }
        recent_missing = recent_fields - set(checkpoint)
        if recent_missing:
            raise OnlineCheckpointError(
                "Online checkpoint is missing recent dynamics "
                f"fields: {sorted(recent_missing)}."
            )
        saved_recent_capacity = _validate_recent_dynamics_capacity(
            checkpoint["recent_dynamics_capacity"],
            "Checkpoint recent_dynamics_capacity",
        )
        if saved_recent_capacity != expected_recent_dynamics_capacity:
            raise OnlineCheckpointError(
                "Checkpoint recent_dynamics_capacity "
                f"{saved_recent_capacity} does not match expected "
                f"recent_dynamics_capacity "
                f"{expected_recent_dynamics_capacity}."
            )
        recent_state = checkpoint["recent_dynamics_buffer"]
        if saved_recent_capacity == 0:
            if recent_state is not None:
                raise OnlineCheckpointError(
                    "Checkpoint recent_dynamics_buffer must be None when "
                    "recent_dynamics_capacity=0."
                )
        else:
            _validate_recent_state_without_template(
                recent_state,
                capacity=saved_recent_capacity,
                online_step=online_step,
            )
    return format_version


def load_online_checkpoint(
    save_dir,
    agent,
    replay_buffer,
    recent_dynamics_buffer,
    expected_env_name,
    expected_horizon_length,
    expected_balanced_sampling,
    expected_initial_replay_size,
    expected_action_dim,
    expected_offline_steps,
    expected_recent_dynamics_capacity=0,
    expected_online_step=None,
    occupancy_detector=None,
    expected_occupancy_detector_config=None,
):
    """Validate and atomically restore all online training state."""
    checkpoint = _load_checkpoint_file(save_dir)
    format_version = _validate_checkpoint_metadata(
        checkpoint,
        expected_env_name=expected_env_name,
        expected_horizon_length=expected_horizon_length,
        expected_balanced_sampling=expected_balanced_sampling,
        expected_initial_replay_size=expected_initial_replay_size,
        expected_action_dim=expected_action_dim,
        expected_offline_steps=expected_offline_steps,
        expected_recent_dynamics_capacity=expected_recent_dynamics_capacity,
        expected_online_step=expected_online_step,
    )

    # Preflight every saved tree, mutable buffer target, and RNG before
    # creating restored objects or changing any live state.
    _validate_flax_state_layout(
        agent, checkpoint["agent"], "Checkpoint Agent"
    )
    _validated_replay_buffer_restore(
        replay_buffer,
        checkpoint,
        expected_balanced_sampling,
        expected_initial_replay_size,
    )
    _validated_recent_dynamics_buffer_restore(
        recent_dynamics_buffer,
        checkpoint,
        expected_recent_dynamics_capacity,
    )
    _validate_occupancy_detector_restore(
        checkpoint,
        format_version=format_version,
        occupancy_detector=occupancy_detector,
        expected_config=expected_occupancy_detector_config,
    )
    _validated_rng_states(checkpoint)

    try:
        restored_agent = flax.serialization.from_state_dict(
            agent, checkpoint["agent"]
        )
    except Exception as exc:
        raise OnlineCheckpointError(
            f"Failed to restore Agent state from online checkpoint: {exc}"
        ) from exc
    restored_occupancy_detector = _restore_occupancy_detector(
        occupancy_detector, checkpoint
    )
    restore_replay_buffer(
        replay_buffer,
        checkpoint,
        expected_balanced_sampling,
        expected_initial_replay_size,
    )
    restore_recent_dynamics_buffer(
        recent_dynamics_buffer,
        checkpoint,
        expected_recent_dynamics_capacity,
    )
    online_rng = restore_rng_states(checkpoint)
    return (
        restored_agent,
        restored_occupancy_detector,
        online_rng,
        checkpoint,
    )


def _validated_recent_dynamics_buffer_restore(
    recent_dynamics_buffer,
    checkpoint,
    expected_capacity,
):
    """Validate recent restore compatibility and return its saved state."""
    expected_capacity = _validate_recent_dynamics_capacity(
        expected_capacity, "expected_recent_dynamics_capacity"
    )
    format_version = checkpoint.get("format_version")
    if isinstance(format_version, bool):
        raise OnlineCheckpointError(
            "Unsupported online checkpoint format_version "
            f"{format_version!r}; supported versions are "
            f"{LEGACY_FORMAT_VERSION}, "
            f"{RECENT_DYNAMICS_FORMAT_VERSION}, and {FORMAT_VERSION}."
        )
    if isinstance(format_version, np.integer):
        format_version = int(format_version)
    if format_version == LEGACY_FORMAT_VERSION:
        if expected_capacity > 0:
            raise OnlineCheckpointError(
                "Online checkpoint format_version 1 does not contain a "
                "recent_dynamics_buffer and cannot be safely restored when "
                f"expected_recent_dynamics_capacity={expected_capacity}."
            )
        if recent_dynamics_buffer is not None:
            raise OnlineCheckpointError(
                "recent_dynamics_buffer template must be None when "
                "expected_recent_dynamics_capacity=0."
            )
        return None
    if format_version not in (
        RECENT_DYNAMICS_FORMAT_VERSION,
        FORMAT_VERSION,
    ):
        raise OnlineCheckpointError(
            "Unsupported online checkpoint format_version "
            f"{format_version!r}; supported versions are "
            f"{LEGACY_FORMAT_VERSION}, "
            f"{RECENT_DYNAMICS_FORMAT_VERSION}, and {FORMAT_VERSION}."
        )

    if "recent_dynamics_capacity" not in checkpoint:
        raise OnlineCheckpointError(
            "Online checkpoint is missing recent_dynamics_capacity."
        )
    if "recent_dynamics_buffer" not in checkpoint:
        raise OnlineCheckpointError(
            "Online checkpoint is missing recent_dynamics_buffer."
        )
    saved_capacity = _validate_recent_dynamics_capacity(
        checkpoint["recent_dynamics_capacity"],
        "Checkpoint recent_dynamics_capacity",
    )
    if saved_capacity != expected_capacity:
        raise OnlineCheckpointError(
            f"Checkpoint recent_dynamics_capacity {saved_capacity} does not "
            f"match expected recent_dynamics_capacity {expected_capacity}."
        )

    state = checkpoint["recent_dynamics_buffer"]
    if expected_capacity == 0:
        if recent_dynamics_buffer is not None:
            raise OnlineCheckpointError(
                "recent_dynamics_buffer template must be None when "
                "expected_recent_dynamics_capacity=0."
            )
        if state is not None:
            raise OnlineCheckpointError(
                "Checkpoint recent_dynamics_buffer must be None when "
                "recent_dynamics_capacity=0."
            )
        return None

    if not isinstance(recent_dynamics_buffer, RecentDynamicsBuffer):
        raise OnlineCheckpointError(
            "recent_dynamics_buffer template must be a "
            f"RecentDynamicsBuffer; got "
            f"{type(recent_dynamics_buffer).__name__}."
        )
    if recent_dynamics_buffer.capacity != expected_capacity:
        raise OnlineCheckpointError(
            "RecentDynamicsBuffer template capacity "
            f"{recent_dynamics_buffer.capacity} does not match expected "
            f"capacity {expected_capacity}."
        )
    _validate_recent_state_without_template(
        state,
        capacity=expected_capacity,
        online_step=checkpoint["online_step"],
    )
    try:
        recent_dynamics_buffer.validate_state_dict(state)
    except (TypeError, ValueError) as exc:
        raise OnlineCheckpointError(
            f"Invalid recent_dynamics_buffer restore state: {exc}"
        ) from exc
    return state


def validate_recent_dynamics_buffer_restore(
    recent_dynamics_buffer,
    checkpoint,
    expected_capacity,
):
    """Validate a recent-buffer restore without changing its template."""
    _validated_recent_dynamics_buffer_restore(
        recent_dynamics_buffer,
        checkpoint,
        expected_capacity,
    )


def restore_recent_dynamics_buffer(
    recent_dynamics_buffer,
    checkpoint,
    expected_capacity,
):
    """Restore a recent transition buffer after strict compatibility checks."""
    state = _validated_recent_dynamics_buffer_restore(
        recent_dynamics_buffer,
        checkpoint,
        expected_capacity,
    )
    if state is None:
        return None
    try:
        recent_dynamics_buffer.load_state_dict(state)
    except (TypeError, ValueError) as exc:
        raise OnlineCheckpointError(
            f"Failed to restore recent_dynamics_buffer: {exc}"
        ) from exc
    return recent_dynamics_buffer


def _validated_replay_buffer_restore(
    replay_buffer,
    checkpoint,
    balanced_sampling,
    initial_replay_size,
):
    """Validate ReplayBuffer restore data without modifying the target."""
    replay_state = checkpoint.get("replay_buffer")
    required_fields = {
        "pointer",
        "size",
        "max_size",
        "data_start",
        "data_count",
        "online_data",
    }
    if not isinstance(replay_state, dict):
        raise OnlineCheckpointError("Checkpoint Replay Buffer state must be a dictionary.")
    missing = required_fields - set(replay_state)
    if missing:
        raise OnlineCheckpointError(
            f"Checkpoint Replay Buffer is missing fields: {sorted(missing)}."
        )

    pointer = replay_state["pointer"]
    size = replay_state["size"]
    max_size = replay_state["max_size"]
    if max_size != replay_buffer.max_size:
        raise OnlineCheckpointError(
            f"Checkpoint Replay Buffer max_size {max_size} does not match "
            f"current max_size {replay_buffer.max_size}."
        )

    data_start = replay_state["data_start"]
    data_count = replay_state["data_count"]
    _validate_replay_layout(
        pointer=pointer,
        size=size,
        max_size=max_size,
        data_start=data_start,
        data_count=data_count,
        online_step=checkpoint["online_step"],
        balanced_sampling=balanced_sampling,
        initial_replay_size=initial_replay_size,
    )

    online_data = replay_state["online_data"]
    if not isinstance(online_data, dict):
        raise OnlineCheckpointError(
            "Checkpoint Replay Buffer online_data must be a dictionary."
        )
    checkpoint_keys = set(online_data)
    current_keys = set(replay_buffer.keys())
    if checkpoint_keys != current_keys:
        raise OnlineCheckpointError(
            "Replay Buffer key mismatch: "
            f"checkpoint={sorted(checkpoint_keys)}, current={sorted(current_keys)}."
        )

    validated_online_data = {}
    for key in sorted(current_keys):
        saved = np.asarray(online_data[key])
        current = replay_buffer[key]
        expected_shape = (data_count, *current.shape[1:])
        if saved.shape != expected_shape:
            raise OnlineCheckpointError(
                f"Replay Buffer field {key!r} shape mismatch: "
                f"checkpoint={saved.shape}, expected={expected_shape}."
            )
        if saved.dtype != current.dtype:
            raise OnlineCheckpointError(
                f"Replay Buffer field {key!r} dtype mismatch: "
                f"checkpoint={saved.dtype}, expected={current.dtype}."
            )
        validated_online_data[key] = saved

    return {
        "pointer": pointer,
        "size": size,
        "data_start": data_start,
        "data_count": data_count,
        "online_data": validated_online_data,
    }


def restore_replay_buffer(
    replay_buffer,
    checkpoint,
    balanced_sampling,
    initial_replay_size,
):
    """Restore only online Replay Buffer data into a freshly rebuilt buffer."""
    validated = _validated_replay_buffer_restore(
        replay_buffer,
        checkpoint,
        balanced_sampling,
        initial_replay_size,
    )
    data_start = validated["data_start"]
    data_count = validated["data_count"]
    for key, saved in validated["online_data"].items():
        replay_buffer[key][data_start : data_start + data_count] = saved

    replay_buffer.pointer = validated["pointer"]
    replay_buffer.size = validated["size"]
    return replay_buffer


def _validated_rng_states(checkpoint):
    """Validate all saved RNG states without changing global generators."""
    try:
        saved_online_rng = np.asarray(checkpoint["online_rng"])
        if (
            saved_online_rng.shape != (2,)
            or saved_online_rng.dtype != np.dtype(np.uint32)
        ):
            raise ValueError(
                "online_rng must have shape (2,) and dtype uint32; "
                f"got shape {saved_online_rng.shape} and "
                f"dtype {saved_online_rng.dtype}"
            )
        online_rng = jnp.asarray(saved_online_rng)
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(checkpoint["numpy_rng_state"])
        python_probe = random.Random()
        python_probe.setstate(checkpoint["python_rng_state"])
    except Exception as exc:
        raise OnlineCheckpointError(
            f"Invalid checkpoint random state: {exc}"
        ) from exc
    return online_rng


def validate_online_checkpoint_restore(
    replay_buffer,
    recent_dynamics_buffer,
    checkpoint,
    balanced_sampling,
    initial_replay_size,
    expected_recent_dynamics_capacity,
):
    """Preflight every mutable online restore target before writing any."""
    _validated_replay_buffer_restore(
        replay_buffer,
        checkpoint,
        balanced_sampling,
        initial_replay_size,
    )
    _validated_recent_dynamics_buffer_restore(
        recent_dynamics_buffer,
        checkpoint,
        expected_recent_dynamics_capacity,
    )
    _validated_rng_states(checkpoint)


def restore_rng_states(checkpoint):
    """Atomically restore JAX, NumPy, and Python RNG states."""
    online_rng = _validated_rng_states(checkpoint)
    previous_numpy_state = np.random.get_state()
    previous_python_state = random.getstate()
    try:
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
    except Exception as exc:
        np.random.set_state(previous_numpy_state)
        random.setstate(previous_python_state)
        raise OnlineCheckpointError(
            f"Failed to restore checkpoint random states: {exc}"
        ) from exc
    return online_rng
