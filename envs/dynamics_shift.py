"""Controllable action-side dynamics shifts for Gym-compatible environments."""

from collections import deque
import numbers

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - compatibility for repositories using old Gym.
    import gym


ACTION_GAIN_KEY = "dynamics_shift/action_gain"
ACTION_DELAY_KEY = "dynamics_shift/action_delay"
POLICY_ACTION_KEY = "dynamics_shift/policy_action"
EXECUTED_ACTION_KEY = "dynamics_shift/executed_action"


def _validate_gain(gain):
    if isinstance(gain, bool) or not isinstance(gain, numbers.Real):
        raise TypeError(
            f"action_gain must be a real number greater than 0; got {gain!r}."
        )
    gain = float(gain)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError(
            f"action_gain must be finite and greater than 0; got {gain!r}."
        )
    return gain


def _validate_delay(delay_steps):
    if isinstance(delay_steps, bool) or not isinstance(delay_steps, numbers.Integral):
        raise TypeError(
            "action_delay must be a non-negative integer; "
            f"got {delay_steps!r}."
        )
    delay_steps = int(delay_steps)
    if delay_steps < 0:
        raise ValueError(
            "action_delay must be a non-negative integer; "
            f"got {delay_steps!r}."
        )
    return delay_steps


def _validate_action_space(env):
    action_space = env.action_space
    required_attributes = ("low", "high", "shape", "dtype")
    missing = [name for name in required_attributes if not hasattr(action_space, name)]
    if missing:
        raise TypeError(
            "Dynamics-shift wrappers require a continuous action space with "
            f"{', '.join(required_attributes)}; missing {missing!r}."
        )
    return action_space


def _replace_step_info(step_result, updates, preserve_executed_action=False):
    """Add shift metadata without changing four- or five-value step semantics."""
    if not isinstance(step_result, tuple) or len(step_result) not in (4, 5):
        raise TypeError(
            "Wrapped environment step() must return a 4-value Gym tuple or "
            f"5-value Gymnasium tuple; got {step_result!r}."
        )

    values = list(step_result)
    info = dict(values[-1])
    if (
        preserve_executed_action
        and ACTION_DELAY_KEY in info
        and EXECUTED_ACTION_KEY in info
    ):
        executed_action = np.array(info[EXECUTED_ACTION_KEY], copy=True)
        info.update(updates)
        info[EXECUTED_ACTION_KEY] = executed_action
    else:
        info.update(updates)
    values[-1] = info
    return tuple(values)


class ActionGainWrapper(gym.Wrapper):
    """Scale policy actions before passing them to the wrapped environment."""

    def __init__(self, env, gain):
        super().__init__(env)
        self.gain = _validate_gain(gain)
        self._shift_action_space = _validate_action_space(env)

    def step(self, action):
        policy_action = np.array(action, copy=True)
        executed_action = np.asarray(
            np.clip(
                self.gain * policy_action,
                self._shift_action_space.low,
                self._shift_action_space.high,
            ),
            dtype=self._shift_action_space.dtype,
        ).copy()

        step_result = self.env.step(executed_action)
        return _replace_step_info(
            step_result,
            {
                ACTION_GAIN_KEY: self.gain,
                POLICY_ACTION_KEY: policy_action,
                EXECUTED_ACTION_KEY: executed_action.copy(),
            },
            preserve_executed_action=True,
        )


class ActionDelayWrapper(gym.Wrapper):
    """Execute the action proposed ``delay_steps`` policy steps earlier."""

    def __init__(self, env, delay_steps):
        super().__init__(env)
        self.delay_steps = _validate_delay(delay_steps)
        self._shift_action_space = _validate_action_space(env)
        self._action_queue = deque()
        self._reset_action_queue()

    def _zero_action(self):
        zero_action = np.zeros(
            self._shift_action_space.shape,
            dtype=self._shift_action_space.dtype,
        )
        return np.asarray(
            np.clip(
                zero_action,
                self._shift_action_space.low,
                self._shift_action_space.high,
            ),
            dtype=self._shift_action_space.dtype,
        ).copy()

    def _reset_action_queue(self):
        self._action_queue.clear()
        for _ in range(self.delay_steps):
            self._action_queue.append(self._zero_action())

    def reset(self, *args, **kwargs):
        self._reset_action_queue()
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        policy_action = np.array(action, copy=True)
        candidate_action = np.asarray(
            np.clip(
                policy_action,
                self._shift_action_space.low,
                self._shift_action_space.high,
            ),
            dtype=self._shift_action_space.dtype,
        ).copy()

        if self.delay_steps == 0:
            executed_action = candidate_action
        else:
            executed_action = self._action_queue.popleft()
            self._action_queue.append(candidate_action)

        step_result = self.env.step(executed_action.copy())
        return _replace_step_info(
            step_result,
            {
                ACTION_DELAY_KEY: self.delay_steps,
                POLICY_ACTION_KEY: policy_action,
                EXECUTED_ACTION_KEY: executed_action.copy(),
            },
        )


def apply_dynamics_shift(env, action_gain=1.0, action_delay=0):
    """Apply gain followed by delay, returning ``env`` unchanged for defaults."""
    action_gain = _validate_gain(action_gain)
    action_delay = _validate_delay(action_delay)

    if action_gain == 1.0 and action_delay == 0:
        return env

    # Wrapper calls run outside-in. Construct delay first so gain is applied first.
    shifted_env = env
    if action_delay != 0:
        shifted_env = ActionDelayWrapper(shifted_env, action_delay)
    if action_gain != 1.0:
        shifted_env = ActionGainWrapper(shifted_env, action_gain)
    return shifted_env
