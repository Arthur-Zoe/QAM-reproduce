"""Episode-boundary checkpointing for online training."""

import errno
import os
import pickle
import random
import tempfile

import flax
import jax.numpy as jnp
import numpy as np


FORMAT_VERSION = 1
CHECKPOINT_FILENAME = "online_checkpoint.pkl"
PROGRESS_FILENAME = "progress.tk"


class OnlineCheckpointError(RuntimeError):
    """Raised when an online checkpoint cannot be safely saved or restored."""


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


def _validate_checkpoint_metadata(
    checkpoint,
    expected_env_name,
    expected_horizon_length,
    expected_balanced_sampling,
    expected_initial_replay_size,
    expected_action_dim,
    expected_offline_steps,
):
    required_fields = {
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
    missing = required_fields - set(checkpoint)
    if missing:
        raise OnlineCheckpointError(
            f"Online checkpoint is missing required fields: {sorted(missing)}."
        )
    if checkpoint["format_version"] != FORMAT_VERSION:
        raise OnlineCheckpointError(
            "Unsupported online checkpoint format_version "
            f"{checkpoint['format_version']!r}; expected {FORMAT_VERSION}."
        )
    if checkpoint["stage"] != "online":
        raise OnlineCheckpointError(
            f"Online checkpoint stage must be 'online'; got {checkpoint['stage']!r}."
        )
    if checkpoint["env_name"] != expected_env_name:
        raise OnlineCheckpointError(
            f"Checkpoint env_name {checkpoint['env_name']!r} does not match "
            f"{expected_env_name!r}."
        )
    if checkpoint["horizon_length"] != expected_horizon_length:
        raise OnlineCheckpointError(
            f"Checkpoint horizon_length {checkpoint['horizon_length']!r} does not "
            f"match {expected_horizon_length!r}."
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
            f"Checkpoint action_dim {checkpoint['action_dim']!r} does not match "
            f"{expected_action_dim!r}."
        )
    if checkpoint["offline_steps"] != expected_offline_steps:
        raise OnlineCheckpointError(
            f"Checkpoint offline_steps {checkpoint['offline_steps']!r} does not "
            f"match {expected_offline_steps!r}."
        )
    online_step = checkpoint["online_step"]
    global_step = checkpoint["global_step"]
    if not isinstance(online_step, int) or online_step < 0:
        raise OnlineCheckpointError(
            f"Checkpoint online_step must be a non-negative integer; got {online_step!r}."
        )
    if global_step != expected_offline_steps + online_step:
        raise OnlineCheckpointError(
            f"Checkpoint global_step {global_step!r} is inconsistent with "
            f"offline_steps={expected_offline_steps} and online_step={online_step}."
        )


def load_online_checkpoint(
    save_dir,
    agent,
    expected_env_name,
    expected_horizon_length,
    expected_balanced_sampling,
    expected_initial_replay_size,
    expected_action_dim,
    expected_offline_steps,
):
    """Load metadata and restore the Agent into the supplied template."""
    checkpoint = _load_checkpoint_file(save_dir)
    _validate_checkpoint_metadata(
        checkpoint,
        expected_env_name=expected_env_name,
        expected_horizon_length=expected_horizon_length,
        expected_balanced_sampling=expected_balanced_sampling,
        expected_initial_replay_size=expected_initial_replay_size,
        expected_action_dim=expected_action_dim,
        expected_offline_steps=expected_offline_steps,
    )
    try:
        restored_agent = flax.serialization.from_state_dict(
            agent, checkpoint["agent"]
        )
    except Exception as exc:
        raise OnlineCheckpointError(
            f"Failed to restore Agent state from online checkpoint: {exc}"
        ) from exc
    return restored_agent, checkpoint


def restore_replay_buffer(
    replay_buffer,
    checkpoint,
    balanced_sampling,
    initial_replay_size,
):
    """Restore only online Replay Buffer data into a freshly rebuilt buffer."""
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

    for key, saved in validated_online_data.items():
        replay_buffer[key][data_start : data_start + data_count] = saved

    replay_buffer.pointer = pointer
    replay_buffer.size = size
    return replay_buffer


def restore_rng_states(checkpoint):
    """Restore JAX, NumPy, and Python RNG states from a validated checkpoint."""
    try:
        online_rng = jnp.asarray(checkpoint["online_rng"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
    except Exception as exc:
        raise OnlineCheckpointError(
            f"Failed to restore checkpoint random states: {exc}"
        ) from exc
    return online_rng
