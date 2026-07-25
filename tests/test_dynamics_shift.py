import unittest

import gymnasium as gym
import numpy as np

from envs.dynamics_shift import (
    ACTION_DELAY_KEY,
    ACTION_GAIN_KEY,
    EXECUTED_ACTION_KEY,
    POLICY_ACTION_KEY,
    ActionDelayWrapper,
    ActionGainWrapper,
    apply_dynamics_shift,
)


class DummyEnv(gym.Env):
    def __init__(self, old_step_api=False):
        self.action_space = gym.spaces.Box(
            low=np.array([-2.0, -0.5], dtype=np.float32),
            high=np.array([2.0, 0.75], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )
        self.old_step_api = old_step_api
        self.executed_actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.executed_actions.append(np.array(action, copy=True))
        result = (
            np.zeros(1, dtype=np.float32),
            1.0,
            False,
            {"source": "dummy"},
        )
        if self.old_step_api:
            return result
        return result[:2] + (False,) + result[2:]


class DynamicsShiftTest(unittest.TestCase):
    def assertActionEqual(self, actual, expected):
        np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float32))

    def test_gain_one_leaves_action_unchanged(self):
        env = DummyEnv()
        wrapped = ActionGainWrapper(env, 1.0)
        wrapped.step(np.array([1.25, 0.25], dtype=np.float32))
        self.assertActionEqual(env.executed_actions[-1], [1.25, 0.25])

    def test_gain_scales_action(self):
        env = DummyEnv()
        wrapped = ActionGainWrapper(env, 0.5)
        wrapped.step(np.array([1.5, 0.5], dtype=np.float32))
        self.assertActionEqual(env.executed_actions[-1], [0.75, 0.25])

    def test_gain_clips_to_action_space(self):
        env = DummyEnv()
        wrapped = ActionGainWrapper(env, 2.0)
        wrapped.step(np.array([1.5, -0.5], dtype=np.float32))
        self.assertActionEqual(env.executed_actions[-1], [2.0, -0.5])

    def test_gain_does_not_modify_callers_action(self):
        env = DummyEnv()
        wrapped = ActionGainWrapper(env, 2.0)
        action = np.array([1.5, -0.5], dtype=np.float32)
        original = action.copy()
        wrapped.step(action)
        np.testing.assert_array_equal(action, original)

    def test_delay_zero_executes_current_action(self):
        env = DummyEnv()
        wrapped = ActionDelayWrapper(env, 0)
        wrapped.step(np.array([1.0, 0.25], dtype=np.float32))
        self.assertActionEqual(env.executed_actions[-1], [1.0, 0.25])

    def test_delay_one_executes_zero_then_previous_action(self):
        env = DummyEnv()
        wrapped = ActionDelayWrapper(env, 1)
        wrapped.reset()
        wrapped.step(np.array([1.0, 0.25], dtype=np.float32))
        wrapped.step(np.array([-1.0, 0.5], dtype=np.float32))
        self.assertActionEqual(env.executed_actions[-2], [0.0, 0.0])
        self.assertActionEqual(env.executed_actions[-1], [1.0, 0.25])

    def test_delay_two_sequence(self):
        env = DummyEnv()
        wrapped = ActionDelayWrapper(env, 2)
        wrapped.reset()
        actions = ([1.0, 0.1], [2.0, 0.2], [-1.0, 0.3], [0.5, 0.4])
        for action in actions:
            wrapped.step(np.array(action, dtype=np.float32))
        expected = ([0.0, 0.0], [0.0, 0.0], actions[0], actions[1])
        for actual, target in zip(env.executed_actions, expected):
            self.assertActionEqual(actual, target)

    def test_reset_clears_delay_queue_with_action_space_dtype(self):
        env = DummyEnv()
        wrapped = ActionDelayWrapper(env, 1)
        wrapped.reset()
        wrapped.step(np.array([1.0, 0.25], dtype=np.float32))
        wrapped.reset()
        wrapped.step(np.array([-1.0, 0.5], dtype=np.float32))
        executed = env.executed_actions[-1]
        self.assertActionEqual(executed, [0.0, 0.0])
        self.assertEqual(executed.shape, env.action_space.shape)
        self.assertEqual(executed.dtype, env.action_space.dtype)

    def test_delay_warm_start_is_clipped_to_action_bounds(self):
        env = DummyEnv()
        env.action_space = gym.spaces.Box(
            low=np.array([0.2, 0.5], dtype=np.float32),
            high=np.array([1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        wrapped = ActionDelayWrapper(env, 1)
        wrapped.reset()
        _, _, _, _, info = wrapped.step(
            np.array([0.8, 1.5], dtype=np.float32)
        )

        executed = env.executed_actions[-1]
        expected = np.clip(
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype),
            env.action_space.low,
            env.action_space.high,
        )
        self.assertActionEqual(executed, [0.2, 0.5])
        np.testing.assert_array_equal(executed, expected)
        self.assertEqual(executed.shape, env.action_space.shape)
        self.assertEqual(executed.dtype, env.action_space.dtype)
        np.testing.assert_array_equal(info[EXECUTED_ACTION_KEY], executed)

    def test_gain_is_applied_before_delay(self):
        env = DummyEnv()
        wrapped = apply_dynamics_shift(env, action_gain=2.0, action_delay=1)
        wrapped.reset()
        _, _, _, _, first_info = wrapped.step(
            np.array([1.5, 0.5], dtype=np.float32)
        )
        _, _, _, _, second_info = wrapped.step(
            np.array([-0.25, 0.1], dtype=np.float32)
        )
        self.assertActionEqual(env.executed_actions[0], [0.0, 0.0])
        self.assertActionEqual(env.executed_actions[1], [2.0, 0.75])
        self.assertActionEqual(first_info[EXECUTED_ACTION_KEY], [0.0, 0.0])
        self.assertActionEqual(second_info[EXECUTED_ACTION_KEY], [2.0, 0.75])

    def test_default_returns_original_env(self):
        env = DummyEnv()
        self.assertIs(apply_dynamics_shift(env), env)

    def test_invalid_gain_and_delay_raise_clear_errors(self):
        env = DummyEnv()
        for gain in (0.0, -1.0, np.inf, "bad", True):
            with self.subTest(gain=gain):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), rf"action_gain.*{gain!r}"
                ):
                    apply_dynamics_shift(env, action_gain=gain)
        for delay in (-1, 1.5, "bad", True):
            with self.subTest(delay=delay):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), rf"action_delay.*{delay!r}"
                ):
                    apply_dynamics_shift(env, action_delay=delay)

    def test_info_records_actions_and_shift_parameters(self):
        env = DummyEnv()
        gain_wrapper = ActionGainWrapper(env, 0.5)
        _, _, _, _, gain_info = gain_wrapper.step(
            np.array([1.0, 0.5], dtype=np.float32)
        )
        self.assertEqual(gain_info[ACTION_GAIN_KEY], 0.5)
        self.assertActionEqual(gain_info[POLICY_ACTION_KEY], [1.0, 0.5])
        self.assertActionEqual(gain_info[EXECUTED_ACTION_KEY], [0.5, 0.25])

        delay_wrapper = ActionDelayWrapper(DummyEnv(), 1)
        _, _, _, _, delay_info = delay_wrapper.step(
            np.array([1.0, 0.5], dtype=np.float32)
        )
        self.assertEqual(delay_info[ACTION_DELAY_KEY], 1)
        self.assertActionEqual(delay_info[POLICY_ACTION_KEY], [1.0, 0.5])
        self.assertActionEqual(delay_info[EXECUTED_ACTION_KEY], [0.0, 0.0])

    def test_old_gym_four_value_step_api_is_preserved(self):
        env = DummyEnv(old_step_api=True)
        result = ActionDelayWrapper(env, 0).step(
            np.array([1.0, 0.25], dtype=np.float32)
        )
        self.assertEqual(len(result), 4)
        self.assertEqual(result[-1][ACTION_DELAY_KEY], 0)


if __name__ == "__main__":
    unittest.main()
