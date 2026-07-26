"""A fixed-capacity ring buffer for recent online transitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


FORMAT_VERSION = 1
ONLINE_TRANSITION_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "masks",
    "next_observations",
)


def _validate_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return value


def _as_supported_array(value: Any, field_description: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(
            f"{field_description} cannot be converted to a NumPy array: {exc}"
        ) from exc
    if array.dtype.hasobject:
        raise ValueError(
            f"{field_description} has unsupported object dtype "
            f"{array.dtype}; numeric or boolean arrays are required."
        )
    return array


def create_recent_transition_template(replay_buffer):
    """Create the online single-transition layout from ReplayBuffer storage."""
    if not isinstance(replay_buffer, Mapping):
        raise ValueError(
            "replay_buffer must be a mapping of physical storage arrays; "
            f"got {type(replay_buffer).__name__}."
        )
    missing = set(ONLINE_TRANSITION_FIELDS) - set(replay_buffer)
    if missing:
        raise ValueError(
            "ReplayBuffer is missing online transition fields: "
            f"{sorted(missing)}."
        )

    template = {}
    for key in ONLINE_TRANSITION_FIELDS:
        storage = _as_supported_array(
            replay_buffer[key], f"ReplayBuffer field {key!r}"
        )
        if storage.ndim == 0:
            raise ValueError(
                f"ReplayBuffer field {key!r} must include a storage axis; "
                f"got shape {storage.shape}."
            )
        template[key] = np.zeros(storage.shape[1:], dtype=storage.dtype)
    return template


class RecentDynamicsBuffer:
    """Transition-level ring buffer that preserves strict field layouts."""

    def __init__(self, data: dict[str, np.ndarray], capacity: int):
        self.data = data
        self.capacity = capacity
        self.size = 0
        self.write_index = 0
        self.total_added = 0

    @classmethod
    def create(cls, example_transition, capacity):
        """Create an empty buffer using a transition as the field template."""
        capacity = _validate_positive_integer(capacity, "capacity")
        if not isinstance(example_transition, dict) or not example_transition:
            raise ValueError(
                "example_transition must be a non-empty dictionary; "
                f"got {type(example_transition).__name__}."
            )

        data = {}
        for key, value in example_transition.items():
            example = _as_supported_array(
                value, f"example_transition field {key!r}"
            )
            data[key] = np.zeros(
                (capacity, *example.shape),
                dtype=example.dtype,
            )
        return cls(data=data, capacity=capacity)

    def _validated_transition(self, transition):
        if not isinstance(transition, dict):
            raise ValueError(
                "transition must be a dictionary; "
                f"got {type(transition).__name__}."
            )
        expected_keys = set(self.data)
        actual_keys = set(transition)
        if actual_keys != expected_keys:
            raise ValueError(
                "transition key mismatch: "
                f"got {sorted(actual_keys)}, expected {sorted(expected_keys)}."
            )

        validated = {}
        for key in sorted(expected_keys):
            value = _as_supported_array(
                transition[key], f"transition field {key!r}"
            )
            expected_shape = self.data[key].shape[1:]
            expected_dtype = self.data[key].dtype
            if value.shape != expected_shape:
                raise ValueError(
                    f"transition field {key!r} shape mismatch: "
                    f"got {value.shape}, expected {expected_shape}."
                )
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"transition field {key!r} dtype mismatch: "
                    f"got {value.dtype}, expected {expected_dtype}."
                )
            validated[key] = value
        return validated

    def validate_transition(self, transition):
        """Validate one transition without changing buffer state."""
        self._validated_transition(transition)

    def prepare_transition(self, transition):
        """Explicitly cast an online transition to this buffer's dtypes."""
        if not isinstance(transition, dict):
            raise ValueError(
                "transition must be a dictionary; "
                f"got {type(transition).__name__}."
            )
        expected_keys = set(self.data)
        actual_keys = set(transition)
        if actual_keys != expected_keys:
            raise ValueError(
                "transition key mismatch: "
                f"got {sorted(actual_keys)}, expected {sorted(expected_keys)}."
            )

        prepared = {}
        for key in sorted(expected_keys):
            expected_dtype = self.data[key].dtype
            try:
                prepared[key] = np.asarray(
                    transition[key], dtype=expected_dtype
                )
            except Exception as exc:
                raise ValueError(
                    f"transition field {key!r} cannot be converted to "
                    f"dtype {expected_dtype}: {exc}"
                ) from exc
        self._validated_transition(prepared)
        return prepared

    def add_transition(self, transition):
        """Append one transition after validating every field."""
        validated = self._validated_transition(transition)
        for key, value in validated.items():
            self.data[key][self.write_index] = value

        self.write_index = (self.write_index + 1) % self.capacity
        self.total_added += 1
        self.size = min(self.total_added, self.capacity)

    def ordered_data(self):
        """Return safe copies ordered from oldest to newest."""
        if self.size == 0:
            return {
                key: value[:0].copy()
                for key, value in self.data.items()
            }

        oldest_index = (self.write_index - self.size) % self.capacity
        indices = (
            oldest_index + np.arange(self.size, dtype=np.int64)
        ) % self.capacity
        return {
            key: value[indices].copy()
            for key, value in self.data.items()
        }

    def sample(self, batch_size, rng=None):
        """Sample individual valid transitions with replacement."""
        batch_size = _validate_positive_integer(batch_size, "batch_size")
        if self.size == 0:
            raise ValueError("Cannot sample from an empty RecentDynamicsBuffer.")

        if rng is None:
            logical_indices = np.random.randint(
                0, self.size, size=batch_size
            )
        elif isinstance(rng, np.random.Generator):
            logical_indices = rng.integers(0, self.size, size=batch_size)
        elif isinstance(rng, np.random.RandomState):
            logical_indices = rng.randint(0, self.size, size=batch_size)
        else:
            raise TypeError(
                "rng must be a numpy.random.Generator or "
                f"numpy.random.RandomState; got {type(rng).__name__}."
            )

        oldest_index = (self.write_index - self.size) % self.capacity
        physical_indices = (oldest_index + logical_indices) % self.capacity
        return {
            key: value[physical_indices].copy()
            for key, value in self.data.items()
        }

    def state_dict(self):
        """Return a pickle-safe representation of the full physical layout."""
        return {
            "format_version": FORMAT_VERSION,
            "capacity": self.capacity,
            "size": self.size,
            "write_index": self.write_index,
            "total_added": self.total_added,
            "data": {
                key: value.copy()
                for key, value in self.data.items()
            },
        }

    def _validated_state_dict(self, state, copy_data):
        if not isinstance(state, dict):
            raise ValueError(
                "RecentDynamicsBuffer state must be a dictionary; "
                f"got {type(state).__name__}."
            )
        required_fields = {
            "format_version",
            "capacity",
            "size",
            "write_index",
            "total_added",
            "data",
        }
        missing = required_fields - set(state)
        if missing:
            raise ValueError(
                "RecentDynamicsBuffer state is missing required fields: "
                f"{sorted(missing)}."
            )
        format_version = state["format_version"]
        if (
            isinstance(format_version, bool)
            or not isinstance(format_version, (int, np.integer))
            or int(format_version) != FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported RecentDynamicsBuffer format_version "
                f"{format_version!r}; expected {FORMAT_VERSION}."
            )

        saved_capacity = state["capacity"]
        if (
            isinstance(saved_capacity, bool)
            or not isinstance(saved_capacity, (int, np.integer))
        ):
            raise ValueError(
                "RecentDynamicsBuffer state capacity must be a positive "
                f"integer; got {saved_capacity!r}."
            )
        saved_capacity = int(saved_capacity)
        if saved_capacity != self.capacity:
            raise ValueError(
                "RecentDynamicsBuffer capacity mismatch: "
                f"checkpoint={saved_capacity}, expected={self.capacity}."
            )

        size = state["size"]
        write_index = state["write_index"]
        total_added = state["total_added"]
        for name, value in (
            ("size", size),
            ("write_index", write_index),
            ("total_added", total_added),
        ):
            if isinstance(value, bool) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(
                    f"RecentDynamicsBuffer {name} must be an integer; "
                    f"got {value!r}."
                )
        size = int(size)
        write_index = int(write_index)
        total_added = int(total_added)

        if not 0 <= size <= self.capacity:
            raise ValueError(
                f"RecentDynamicsBuffer size {size} is outside "
                f"[0, {self.capacity}]."
            )
        if not 0 <= write_index < self.capacity:
            raise ValueError(
                f"RecentDynamicsBuffer write_index {write_index} is outside "
                f"[0, {self.capacity})."
            )
        if total_added < size:
            raise ValueError(
                f"RecentDynamicsBuffer total_added {total_added} must be at "
                f"least size {size}."
            )
        expected_write_index = total_added % self.capacity
        if write_index != expected_write_index:
            raise ValueError(
                f"RecentDynamicsBuffer write_index {write_index} does not "
                f"match total_added % capacity ({expected_write_index})."
            )
        expected_size = min(total_added, self.capacity)
        if size != expected_size:
            raise ValueError(
                f"RecentDynamicsBuffer size {size} does not match "
                f"min(total_added, capacity) ({expected_size})."
            )

        saved_data = state["data"]
        if not isinstance(saved_data, dict):
            raise ValueError(
                "RecentDynamicsBuffer state data must be a dictionary; "
                f"got {type(saved_data).__name__}."
            )
        expected_keys = set(self.data)
        saved_keys = set(saved_data)
        if saved_keys != expected_keys:
            raise ValueError(
                "RecentDynamicsBuffer state key mismatch: "
                f"checkpoint={sorted(saved_keys)}, "
                f"expected={sorted(expected_keys)}."
            )

        validated_data = {}
        for key in sorted(expected_keys):
            saved = _as_supported_array(
                saved_data[key], f"RecentDynamicsBuffer field {key!r}"
            )
            expected_shape = self.data[key].shape
            expected_dtype = self.data[key].dtype
            if saved.shape != expected_shape:
                raise ValueError(
                    f"RecentDynamicsBuffer field {key!r} shape mismatch: "
                    f"checkpoint={saved.shape}, expected={expected_shape}."
                )
            if saved.dtype != expected_dtype:
                raise ValueError(
                    f"RecentDynamicsBuffer field {key!r} dtype mismatch: "
                    f"checkpoint={saved.dtype}, expected={expected_dtype}."
                )
            validated_data[key] = saved.copy() if copy_data else saved
        return validated_data, size, write_index, total_added

    def validate_state_dict(self, state):
        """Validate a saved state without changing this buffer."""
        self._validated_state_dict(state, copy_data=False)

    def load_state_dict(self, state):
        """Strictly validate and then restore a saved physical layout."""
        validated_data, size, write_index, total_added = (
            self._validated_state_dict(state, copy_data=True)
        )

        # No state is changed until all metadata and arrays pass validation.
        self.data = validated_data
        self.size = size
        self.write_index = write_index
        self.total_added = total_added
        return self

    def clear(self):
        """Logically discard all retained transitions."""
        self.size = 0
        self.write_index = 0
        self.total_added = 0
