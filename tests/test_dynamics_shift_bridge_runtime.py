import copy
from dataclasses import replace
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import flax
import jax
import numpy as np

from evaluation import evaluate
from log_utils import CsvLogger
from utils.datasets import ReplayBuffer
from utils.dynamics_shift_bridge import DynamicsShiftBridge
from utils.dynamics_shift_bridge_runtime import (
    BRIDGE_LOG_FIELDS,
    BRIDGE_MODEL_SEED_OFFSET,
    CORRECTION_METRIC_FIELDS,
    EXECUTION_METRIC_FIELDS,
    DynamicsShiftBridgeRuntime,
    DynamicsShiftBridgeRuntimeConfig,
    bridge_step_environment,
    call_preserving_global_numpy_rng,
    compute_bridge_normalization,
    deterministic_uniform_indices,
    extract_primitive_transitions,
    shadow_step_environment,
    should_update_bridge_online,
    validate_bridge_checkpoint_resume,
    validate_bridge_runtime_config,
)
from utils.recent_dynamics_buffer import RecentDynamicsBuffer


class FakeEnvironment:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return (
            np.ones(3, dtype=np.float32),
            1.0,
            False,
            False,
            {},
        )


class EvaluationAgent:
    def sample_actions(self, observations, rng):
        del observations, rng
        return np.asarray([0.1, -0.1], dtype=np.float32)


class EvaluationEnvironment:
    def __init__(self, episode_length=1):
        self.episode_length = episode_length
        self.actions = []
        self.reset_seeds = []
        self.step_count = 0
        self.episode_return = 0.0

    def reset(self, **kwargs):
        self.reset_seeds.append(kwargs.get("seed"))
        self.step_count = 0
        self.episode_return = 0.0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        action = np.asarray(action).copy()
        self.actions.append(action)
        self.step_count += 1
        reward = float(np.sum(action))
        self.episode_return += reward
        done = self.step_count >= self.episode_length
        return (
            np.full(3, self.step_count, dtype=np.float32),
            reward,
            done,
            False,
            {
                "success": float(done),
                "episode": {
                    "return": self.episode_return,
                    "length": self.step_count,
                },
            },
        )


class DynamicsShiftBridgeRuntimeTest(unittest.TestCase):
    def primitive_dataset(self, size=32):
        generator = np.random.default_rng(123)
        observations = generator.uniform(
            -0.5, 0.5, (size, 3)
        ).astype(np.float32)
        actions = generator.uniform(
            -0.6, 0.6, (size, 2)
        ).astype(np.float32)
        matrix = np.asarray(
            [[0.7, -0.2], [0.1, 0.5], [-0.3, 0.4]],
            dtype=np.float32,
        )
        return {
            "observations": observations,
            "actions": actions,
            "next_observations": observations + actions @ matrix.T,
            "rewards": np.zeros(size, dtype=np.float32),
            "terminals": np.zeros(size, dtype=np.float32),
            "masks": np.ones(size, dtype=np.float32),
        }

    def enabled_config(self, **overrides):
        config = DynamicsShiftBridgeRuntimeConfig(
            enabled=True,
            hidden_dim=8,
            num_hidden_layers=2,
            learning_rate=3e-3,
            clip_grad_norm=5.0,
            offline_steps=4,
            offline_batch_size=8,
            online_start_size=4,
            online_update_interval=2,
            online_updates_per_interval=2,
            online_batch_size=4,
            correction_steps=3,
            correction_step_size=0.1,
            max_residual=0.1,
            normalization_max_samples=16,
        )
        return replace(config, **overrides)

    def execution_config(self, **overrides):
        values = dict(
            apply_correction=True,
            gate_max_online_eval_mse=0.10,
            gate_uncertainty_multiplier=1.0,
            gate_min_shift_excess=0.005,
            gate_min_relative_improvement=0.20,
            apply_ramp_steps=4,
            apply_residual_scale=1.0,
        )
        values.update(overrides)
        return self.enabled_config(**values)

    def make_runtime(self, **overrides):
        return DynamicsShiftBridgeRuntime.create(
            config=self.enabled_config(**overrides),
            dataset=self.primitive_dataset(),
            expected_action_shape=(2,),
            action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
            action_high=np.asarray([1.0, 1.0], dtype=np.float32),
            seed=17,
        )

    def make_execution_runtime(self, **overrides):
        runtime = DynamicsShiftBridgeRuntime.create(
            config=self.execution_config(**overrides),
            dataset=self.primitive_dataset(),
            expected_action_shape=(2,),
            action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
            action_high=np.asarray([1.0, 1.0], dtype=np.float32),
            seed=17,
        )
        runtime.bridge_online_ready = True
        runtime.online_eval["normalized_mse"] = np.float32(0.02)
        return runtime

    def correction_metrics(self, *, pre=0.10, post=0.05):
        metrics = {
            name: np.float32(0.0)
            for name in CORRECTION_METRIC_FIELDS
        }
        metrics.update(
            pre_match_mse=np.float32(pre),
            post_match_mse=np.float32(post),
            match_improvement=np.float32(pre - post),
        )
        return metrics

    def make_recent_buffer(self, count=0, capacity=8):
        dataset = self.primitive_dataset(size=max(count, 1))
        example = {
            "observations": np.zeros(3, dtype=np.float32),
            "actions": np.zeros(2, dtype=np.float32),
            "rewards": np.float32(0.0),
            "terminals": np.float32(0.0),
            "masks": np.float32(1.0),
            "next_observations": np.zeros(3, dtype=np.float32),
        }
        buffer = RecentDynamicsBuffer.create(example, capacity=capacity)
        for index in range(count):
            buffer.add_transition(
                {
                    "observations": dataset["observations"][index],
                    "actions": dataset["actions"][index],
                    "rewards": np.float32(0.0),
                    "terminals": np.float32(0.0),
                    "masks": np.float32(1.0),
                    "next_observations": dataset[
                        "next_observations"
                    ][index],
                }
            )
        return buffer

    def assert_trees_equal(self, first, second):
        first_leaves = jax.tree_util.tree_leaves(first)
        second_leaves = jax.tree_util.tree_leaves(second)
        self.assertEqual(len(first_leaves), len(second_leaves))
        for first_leaf, second_leaf in zip(first_leaves, second_leaves):
            np.testing.assert_array_equal(first_leaf, second_leaf)

    def assert_numpy_rng_state_equal(self, first, second):
        self.assertEqual(len(first), len(second))
        for first_item, second_item in zip(first, second):
            if isinstance(first_item, np.ndarray):
                np.testing.assert_array_equal(first_item, second_item)
            else:
                self.assertEqual(first_item, second_item)

    def assert_nested_equal(self, first, second):
        if isinstance(first, dict):
            self.assertIsInstance(second, dict)
            self.assertEqual(tuple(first), tuple(second))
            for name in first:
                self.assert_nested_equal(first[name], second[name])
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        else:
            self.assertEqual(first, second)

    def assert_runtime_equal(self, first, second):
        self.assertEqual(first.config, second.config)
        self.assert_trees_equal(first.bridge, second.bridge)
        self.assert_nested_equal(
            first.sampling_rng_state(),
            second.sampling_rng_state(),
        )
        for name in (
            "offline_eval",
            "online_eval",
            "last_execution_metrics",
            "_correction_sums",
            "_execution_sums",
        ):
            self.assert_nested_equal(
                getattr(first, name), getattr(second, name)
            )
        for name in (
            "normalization_sample_count",
            "bridge_offline_ready",
            "bridge_online_ready",
            "online_update_bursts",
            "_online_heldout_count",
            "bridge_applied_steps",
            "_correction_count",
            "_execution_count",
            "_last_logged_online_step",
            "_is_evaluation_snapshot",
        ):
            self.assertEqual(getattr(first, name), getattr(second, name))
        for name in (
            "last_base_action",
            "last_corrected_action",
            "last_executed_action",
        ):
            first_value = getattr(first, name)
            second_value = getattr(second, name)
            if first_value is None or second_value is None:
                self.assertIs(first_value, second_value)
            else:
                np.testing.assert_array_equal(first_value, second_value)

    def test_bridge_defaults_to_disabled_shadow_mode(self):
        config = DynamicsShiftBridgeRuntimeConfig()

        self.assertFalse(config.enabled)
        self.assertFalse(config.apply_correction)

    def test_disabled_validation_does_not_advance_numpy_rng(self):
        config = DynamicsShiftBridgeRuntimeConfig()
        np.random.seed(91)
        state_before = copy.deepcopy(np.random.get_state())

        validate_bridge_runtime_config(
            config,
            recent_dynamics_capacity=0,
            online_save_interval=10,
            log_interval=100,
        )

        state_after = np.random.get_state()
        for before, after in zip(state_before, state_after):
            if isinstance(before, np.ndarray):
                np.testing.assert_array_equal(before, after)
            else:
                self.assertEqual(before, after)

    def test_all_flag_ranges_and_dependencies_are_validated(self):
        invalid_fields = {
            "hidden_dim": 0,
            "num_hidden_layers": 0,
            "learning_rate": np.nan,
            "clip_grad_norm": 0.0,
            "offline_steps": -1,
            "offline_batch_size": 0,
            "online_start_size": 0,
            "online_update_interval": 0,
            "online_updates_per_interval": 0,
            "online_batch_size": 0,
            "correction_steps": -1,
            "correction_step_size": -0.1,
            "dynamics_match_weight": np.inf,
            "action_l2_weight": -0.1,
            "max_residual": -0.1,
            "gate_max_online_eval_mse": np.nan,
            "gate_uncertainty_multiplier": np.inf,
            "gate_min_shift_excess": -0.1,
            "gate_min_relative_improvement": np.nan,
            "apply_ramp_steps": -1,
            "apply_residual_scale": 0.0,
            "normalization_max_samples": 0,
        }
        for name, value in invalid_fields.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_bridge_runtime_config(
                        self.enabled_config(**{name: value}),
                        recent_dynamics_capacity=8,
                        online_save_interval=0,
                        log_interval=2,
                    )

        for name, value in (
            ("gate_min_relative_improvement", 1.1),
            ("apply_residual_scale", 1.1),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_bridge_runtime_config(
                        self.execution_config(**{name: value}),
                        recent_dynamics_capacity=8,
                        online_save_interval=0,
                        log_interval=2,
                    )

        for capacity in (0, 3):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(
                    ValueError, "recent_dynamics_capacity"
                ):
                    validate_bridge_runtime_config(
                        self.enabled_config(),
                        recent_dynamics_capacity=capacity,
                        online_save_interval=0,
                        log_interval=2,
                    )

    def test_apply_correction_requires_bridge_enabled(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_bridge_runtime_config(
                DynamicsShiftBridgeRuntimeConfig(
                    enabled=False,
                    apply_correction=True,
                ),
                recent_dynamics_capacity=0,
                online_save_interval=0,
                log_interval=2,
            )
        validate_bridge_runtime_config(
            self.execution_config(),
            recent_dynamics_capacity=8,
            online_save_interval=0,
            log_interval=2,
        )

    def test_bridge_checkpoint_save_and_resume_are_rejected(self):
        with self.assertRaisesRegex(
            NotImplementedError, "checkpoint v4"
        ):
            validate_bridge_runtime_config(
                self.enabled_config(),
                recent_dynamics_capacity=8,
                online_save_interval=10,
                log_interval=2,
            )
        with self.assertRaisesRegex(
            NotImplementedError, "checkpoint v4"
        ):
            validate_bridge_runtime_config(
                self.execution_config(),
                recent_dynamics_capacity=8,
                online_save_interval=10,
                log_interval=2,
            )
        with self.assertRaisesRegex(
            NotImplementedError, "checkpoint v4"
        ):
            validate_bridge_checkpoint_resume(True, "online")
        validate_bridge_checkpoint_resume(False, "online")
        validate_bridge_checkpoint_resume(True, "offline")

    def test_primitive_transition_extraction_uses_raw_single_steps(self):
        dataset = self.primitive_dataset()
        original = {
            name: value.copy() for name, value in dataset.items()
        }

        transitions = extract_primitive_transitions(dataset, (2,))

        self.assertEqual(
            set(transitions),
            {"observations", "actions", "next_observations"},
        )
        self.assertEqual(transitions["actions"].shape, (32, 2))
        np.testing.assert_array_equal(
            transitions["next_observations"]
            - transitions["observations"],
            dataset["next_observations"] - dataset["observations"],
        )
        for name, value in original.items():
            np.testing.assert_array_equal(dataset[name], value)

    def test_invalid_primitive_transition_data_is_rejected(self):
        cases = {}
        empty = self.primitive_dataset(size=0)
        cases["at least two"] = empty
        mismatched_observation = self.primitive_dataset()
        mismatched_observation["next_observations"] = (
            mismatched_observation["next_observations"][:, :2]
        )
        cases["exactly match"] = mismatched_observation
        mismatched_length = self.primitive_dataset()
        mismatched_length["actions"] = mismatched_length["actions"][:-1]
        cases["same size"] = mismatched_length
        nonfinite = self.primitive_dataset()
        nonfinite["observations"][-1, -1] = np.nan
        cases["finite"] = nonfinite

        for message, dataset in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    extract_primitive_transitions(dataset, (2,))

    def test_action_chunks_are_rejected_as_primitive_actions(self):
        dataset = self.primitive_dataset()
        sequence_chunk = {
            **dataset,
            "actions": np.zeros((32, 5, 2), dtype=np.float32),
        }
        flattened_chunk = {
            **dataset,
            "actions": np.zeros((32, 10), dtype=np.float32),
        }

        for invalid in (sequence_chunk, flattened_chunk):
            with self.subTest(shape=invalid["actions"].shape):
                with self.assertRaisesRegex(
                    ValueError, "primitive actions"
                ):
                    extract_primitive_transitions(invalid, (2,))

    def test_runtime_offline_batches_remain_primitive_transitions(self):
        dataset = self.primitive_dataset()
        captured = []
        original_update = DynamicsShiftBridge.update_offline

        def capture_update(bridge, batch):
            captured.append(
                {name: np.asarray(value).copy() for name, value in batch.items()}
            )
            return original_update(bridge, batch)

        with mock.patch.object(
            DynamicsShiftBridge,
            "update_offline",
            new=capture_update,
        ):
            DynamicsShiftBridgeRuntime.create(
                config=self.enabled_config(offline_steps=2),
                dataset=dataset,
                expected_action_shape=(2,),
                action_low=-1.0,
                action_high=1.0,
                seed=5,
            )

        self.assertEqual(len(captured), 2)
        for batch in captured:
            self.assertEqual(batch["actions"].shape, (8, 2))
            np.testing.assert_allclose(
                batch["next_observations"] - batch["observations"],
                batch["actions"]
                @ np.asarray(
                    [[0.7, 0.1, -0.3], [-0.2, 0.5, 0.4]],
                    dtype=np.float32,
                ),
                atol=1e-7,
                rtol=0.0,
            )

    def test_normalization_is_deterministic_finite_and_non_mutating(self):
        transitions = extract_primitive_transitions(
            self.primitive_dataset(), (2,)
        )
        before = {
            name: value.copy() for name, value in transitions.items()
        }

        first = compute_bridge_normalization(transitions, 16)
        second = compute_bridge_normalization(transitions, 16)

        self.assertEqual(first["sample_count"], 16)
        for name in first:
            if name == "sample_count":
                continue
            np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(first[name].dtype, np.dtype(np.float32))
            self.assertTrue(np.all(np.isfinite(first[name])))
        for name in transitions:
            np.testing.assert_array_equal(transitions[name], before[name])

    def test_normalization_sample_limit_uses_uniform_indices(self):
        np.random.seed(314)
        global_state = copy.deepcopy(np.random.get_state())
        indices = deterministic_uniform_indices(101, 10)
        np.testing.assert_array_equal(
            indices,
            np.linspace(0, 100, num=10, dtype=np.int64),
        )
        self.assertEqual(len(indices), len(np.unique(indices)))
        stats = compute_bridge_normalization(
            extract_primitive_transitions(
                self.primitive_dataset(size=101), (2,)
            ),
            10,
        )
        self.assertEqual(stats["sample_count"], 10)
        self.assert_numpy_rng_state_equal(
            global_state, np.random.get_state()
        )

    def test_nonfinite_primitive_data_is_rejected_before_normalization(self):
        dataset = self.primitive_dataset(size=16)
        dataset["observations"][0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            extract_primitive_transitions(dataset, (2,))

    def test_bridge_rng_is_independent_and_deterministic(self):
        np.random.seed(101)
        global_state = copy.deepcopy(np.random.get_state())
        first = self.make_runtime()
        state_after_first = np.random.get_state()
        second = self.make_runtime()

        for before, after in zip(global_state, state_after_first):
            if isinstance(before, np.ndarray):
                np.testing.assert_array_equal(before, after)
            else:
                self.assertEqual(before, after)
        self.assertEqual(
            first.sampling_rng_state(), second.sampling_rng_state()
        )
        self.assertEqual(
            (17 + BRIDGE_MODEL_SEED_OFFSET) % (2**32),
            2_000_020,
        )
        self.assert_trees_equal(
            first.bridge.offline_model.params,
            second.bridge.offline_model.params,
        )

    def test_offline_evaluation_does_not_advance_global_numpy_rng(self):
        np.random.seed(20260728)
        expected_state = copy.deepcopy(np.random.get_state())
        random.seed(19)
        expected_python_rng = random.Random(19)
        independent_rng = np.random.default_rng(23)
        expected_independent_rng = np.random.default_rng(23)
        jax_key = jax.random.PRNGKey(29)

        result = call_preserving_global_numpy_rng(
            lambda: {
                "nested": (
                    [np.random.random() for _ in range(7)],
                    random.random(),
                    independent_rng.random(),
                    jax.random.uniform(jax_key),
                )
            }
        )

        self.assertIn("nested", result)
        self.assert_numpy_rng_state_equal(
            expected_state, np.random.get_state()
        )
        expected_python_rng.random()
        self.assertEqual(random.random(), expected_python_rng.random())
        self.assertEqual(
            independent_rng.random(),
            expected_independent_rng.random(size=2)[1],
        )
        np.testing.assert_array_equal(jax_key, jax.random.PRNGKey(29))

        def failing_evaluation():
            np.random.random()
            raise RuntimeError("evaluation failed")

        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            call_preserving_global_numpy_rng(failing_evaluation)
        self.assert_numpy_rng_state_equal(
            expected_state, np.random.get_state()
        )

    def test_bridge_seeds_are_normalized_to_supported_ranges(self):
        large_seed = 2**80 + 17
        reduced_seed = large_seed % (2**64)

        first = DynamicsShiftBridgeRuntime.create(
            config=self.enabled_config(offline_steps=0),
            dataset=self.primitive_dataset(),
            expected_action_shape=(2,),
            action_low=-1.0,
            action_high=1.0,
            seed=large_seed,
        )
        second = DynamicsShiftBridgeRuntime.create(
            config=self.enabled_config(offline_steps=0),
            dataset=self.primitive_dataset(),
            expected_action_shape=(2,),
            action_low=-1.0,
            action_high=1.0,
            seed=reduced_seed,
        )

        self.assert_trees_equal(
            first.bridge.offline_model.params,
            second.bridge.offline_model.params,
        )
        self.assertEqual(
            first.sampling_rng_state(),
            second.sampling_rng_state(),
        )

    def test_offline_pretraining_evaluation_and_synchronization(self):
        runtime = self.make_runtime(offline_steps=5)

        self.assertTrue(runtime.bridge_offline_ready)
        self.assertEqual(runtime.bridge.offline_model.step, 6)
        self.assertEqual(runtime.bridge.online_model.step, 1)
        for value in runtime.offline_eval.values():
            self.assertTrue(np.isfinite(value))
        self.assert_trees_equal(
            runtime.bridge.offline_model.params,
            runtime.bridge.online_model.params,
        )
        for offline_leaf, online_leaf in zip(
            jax.tree_util.tree_leaves(runtime.bridge.offline_model.params),
            jax.tree_util.tree_leaves(runtime.bridge.online_model.params),
        ):
            self.assertIsNot(offline_leaf, online_leaf)

    def test_normalization_remains_fixed_after_online_updates(self):
        runtime = self.make_runtime()
        scales_before = (
            np.asarray(runtime.bridge.observation_scale).copy(),
            np.asarray(runtime.bridge.action_scale).copy(),
            np.asarray(runtime.bridge.delta_scale).copy(),
        )

        runtime.maybe_update_online(4, self.make_recent_buffer(4))

        for before, after in zip(
            scales_before,
            (
                runtime.bridge.observation_scale,
                runtime.bridge.action_scale,
                runtime.bridge.delta_scale,
            ),
        ):
            np.testing.assert_array_equal(before, after)

    def test_recent_size_and_interval_gate(self):
        self.assertFalse(
            should_update_bridge_online(
                online_step=2,
                recent_size=3,
                start_size=4,
                update_interval=2,
            )
        )
        self.assertFalse(
            should_update_bridge_online(
                online_step=3,
                recent_size=4,
                start_size=4,
                update_interval=2,
            )
        )
        self.assertTrue(
            should_update_bridge_online(
                online_step=4,
                recent_size=4,
                start_size=4,
                update_interval=2,
            )
        )

    def test_online_burst_schedule_updates_only_online_model(self):
        runtime = self.make_runtime()
        recent = self.make_recent_buffer(4)
        offline_before = flax.serialization.to_state_dict(
            runtime.bridge.offline_model
        )
        online_before = flax.serialization.to_state_dict(
            runtime.bridge.online_model
        )

        self.assertFalse(runtime.maybe_update_online(3, recent))
        self.assertTrue(runtime.maybe_update_online(4, recent))

        self.assertEqual(runtime.online_update_bursts, 1)
        self.assertEqual(runtime.bridge.online_model.step, 3)
        self.assert_trees_equal(
            flax.serialization.to_state_dict(
                runtime.bridge.offline_model
            ),
            offline_before,
        )
        changed = any(
            not np.array_equal(before, after)
            for before, after in zip(
                jax.tree_util.tree_leaves(online_before),
                jax.tree_util.tree_leaves(
                    flax.serialization.to_state_dict(
                        runtime.bridge.online_model
                    )
                ),
            )
        )
        self.assertTrue(changed)
        for value in runtime.online_eval.values():
            self.assertTrue(np.isfinite(value))

    def test_online_uncertainty_refreshes_on_rolling_heldout_tail(self):
        runtime = self.make_runtime(
            online_update_interval=4,
            online_batch_size=4,
        )
        recent = self.make_recent_buffer(4, capacity=8)
        self.assertTrue(runtime.maybe_update_online(4, recent))
        online_step_after_burst = int(runtime.bridge.online_model.step)

        new_transition = {
            "observations": np.asarray(
                [0.3, -0.2, 0.1], dtype=np.float32
            ),
            "actions": np.asarray([0.91, -0.73], dtype=np.float32),
            "rewards": np.float32(0.0),
            "terminals": np.float32(0.0),
            "masks": np.float32(1.0),
            "next_observations": np.asarray(
                [0.4, -0.1, 0.2], dtype=np.float32
            ),
        }
        recent.add_transition(
            recent.prepare_transition(new_transition)
        )
        captured_actions = []
        original_evaluate = DynamicsShiftBridge.evaluate_online

        def capture_evaluate(bridge, transitions):
            captured_actions.append(
                np.asarray(transitions["actions"]).copy()
            )
            return original_evaluate(bridge, transitions)

        with mock.patch.object(
            DynamicsShiftBridge,
            "evaluate_online",
            new=capture_evaluate,
        ):
            self.assertFalse(runtime.maybe_update_online(5, recent))

        self.assertEqual(
            int(runtime.bridge.online_model.step),
            online_step_after_burst,
        )
        self.assertEqual(len(captured_actions), 1)
        self.assertEqual(captured_actions[0].shape[0], 2)
        np.testing.assert_array_equal(
            captured_actions[0][-1],
            new_transition["actions"],
        )
        for value in runtime.online_eval.values():
            self.assertTrue(np.isfinite(value))

    def test_shadow_readiness_requires_offline_and_online(self):
        runtime = self.make_runtime()
        self.assertTrue(runtime.bridge_offline_ready)
        self.assertFalse(runtime.bridge_online_ready)
        self.assertFalse(runtime.bridge_shadow_ready)

        runtime.maybe_update_online(4, self.make_recent_buffer(4))

        self.assertTrue(runtime.bridge_online_ready)
        self.assertTrue(runtime.bridge_shadow_ready)

    def test_not_ready_correction_is_exact_noop_with_zero_metrics(self):
        runtime = self.make_runtime()
        base_action = np.asarray([0.2, -0.1], dtype=np.float32)

        corrected, metrics = runtime.shadow_correct(
            np.zeros(3, dtype=np.float32), base_action
        )

        np.testing.assert_array_equal(corrected, base_action)
        for name in CORRECTION_METRIC_FIELDS:
            self.assertEqual(float(metrics[name]), 0.0)

    def test_evaluate_action_requires_an_isolated_snapshot(self):
        runtime = self.make_execution_runtime()

        with self.assertRaisesRegex(
            RuntimeError, "make_evaluation_snapshot"
        ):
            runtime.evaluate_action(
                np.zeros(3, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            )

    def test_not_ready_evaluation_snapshot_is_an_exact_noop(self):
        runtime = self.make_execution_runtime()
        runtime.bridge_online_ready = False
        snapshot = runtime.make_evaluation_snapshot()
        base_action = np.asarray([0.2, -0.1], dtype=np.float32)

        executed, diagnostics = snapshot.evaluate_action(
            np.zeros(3, dtype=np.float32), base_action
        )

        np.testing.assert_array_equal(executed, base_action)
        for name in (
            "evaluation_bridge_ready_fraction",
            "evaluation_bridge_gate_open_fraction",
            "evaluation_bridge_requested_fraction",
            "evaluation_bridge_applied_fraction",
            "evaluation_bridge_executed_residual_l2",
            "evaluation_bridge_executed_residual_abs_max",
        ):
            self.assertEqual(float(diagnostics[name]), 0.0)
        self.assertEqual(runtime.bridge_applied_steps, 0)

    def test_episode_local_snapshot_advances_only_its_own_ramp(self):
        runtime = self.make_execution_runtime()
        runtime.bridge_applied_steps = 2
        first_snapshot = runtime.make_evaluation_snapshot()
        second_snapshot = runtime.make_evaluation_snapshot()
        self.assertIs(first_snapshot.bridge, runtime.bridge)
        self.assertIsNot(
            first_snapshot.sampling_rng, runtime.sampling_rng
        )
        base = np.zeros(2, dtype=np.float32)
        candidate = np.asarray([0.08, -0.04], dtype=np.float32)

        with mock.patch.object(
            DynamicsShiftBridge,
            "correct_actions",
            return_value=(candidate, self.correction_metrics()),
        ):
            first_action, first_info = first_snapshot.evaluate_action(
                np.zeros(3, dtype=np.float32), base
            )
            second_action, _ = first_snapshot.evaluate_action(
                np.zeros(3, dtype=np.float32), base
            )
            independent_action, _ = second_snapshot.evaluate_action(
                np.zeros(3, dtype=np.float32), base
            )

        np.testing.assert_allclose(first_action, 0.75 * candidate)
        np.testing.assert_allclose(second_action, candidate)
        np.testing.assert_array_equal(
            independent_action, first_action
        )
        self.assertEqual(
            float(first_info["evaluation_bridge_applied_fraction"]),
            1.0,
        )
        self.assertEqual(first_snapshot.bridge_applied_steps, 4)
        self.assertEqual(second_snapshot.bridge_applied_steps, 3)
        self.assertEqual(runtime.bridge_applied_steps, 2)

    def test_correction_evaluation_is_deterministic_and_runtime_pure(self):
        runtime = self.make_execution_runtime()
        runtime_before = copy.deepcopy(runtime)
        candidate_delta = np.asarray([0.08, -0.04], dtype=np.float32)

        def correct_actions(bridge, observation, base_action):
            del bridge, observation
            return (
                np.asarray(base_action) + candidate_delta,
                self.correction_metrics(),
            )

        def run_evaluation():
            environment = EvaluationEnvironment(episode_length=2)
            result = call_preserving_global_numpy_rng(
                evaluate,
                agent=EvaluationAgent(),
                env=environment,
                num_eval_episodes=3,
                action_dim=2,
                episode_seeds=(30001, 30002, 30003),
                action_transform_factory=lambda: (
                    runtime.make_evaluation_snapshot().evaluate_action
                ),
            )
            return result, environment

        with mock.patch.object(
            DynamicsShiftBridge,
            "correct_actions",
            new=correct_actions,
        ):
            first, first_environment = run_evaluation()
            second, second_environment = run_evaluation()

        self.assertEqual(first[0], second[0])
        self.assertEqual(len(first[1]), len(second[1]))
        for first_trajectory, second_trajectory in zip(
            first[1], second[1]
        ):
            self.assertEqual(
                tuple(first_trajectory), tuple(second_trajectory)
            )
            for name in first_trajectory:
                for first_value, second_value in zip(
                    first_trajectory[name],
                    second_trajectory[name],
                ):
                    self.assert_nested_equal(
                        first_value, second_value
                    )
        for first_action, second_action in zip(
            first_environment.actions,
            second_environment.actions,
        ):
            np.testing.assert_array_equal(first_action, second_action)
        self.assertFalse(
            np.array_equal(
                first_environment.actions[0],
                np.asarray([0.1, -0.1], dtype=np.float32),
            )
        )
        self.assertGreater(
            first[0]["evaluation_bridge_applied_fraction"], 0.0
        )
        self.assertGreater(
            first[0]["evaluation_bridge_executed_residual_l2"], 0.0
        )
        self.assert_runtime_equal(runtime, runtime_before)

    def test_full_correction_evaluation_cannot_change_next_online_step(self):
        direct_runtime = self.make_execution_runtime()
        evaluated_runtime = self.make_execution_runtime()
        candidate_delta = np.asarray([0.08, -0.04], dtype=np.float32)
        base_action = np.asarray([0.1, -0.1], dtype=np.float32)
        observation = np.zeros(3, dtype=np.float32)

        def correct_actions(bridge, observation, proposed_action):
            del bridge, observation
            return (
                np.asarray(proposed_action) + candidate_delta,
                self.correction_metrics(),
            )

        with mock.patch.object(
            DynamicsShiftBridge,
            "correct_actions",
            new=correct_actions,
        ):
            call_preserving_global_numpy_rng(
                evaluate,
                agent=EvaluationAgent(),
                env=EvaluationEnvironment(),
                num_eval_episodes=40,
                action_dim=2,
                episode_seeds=tuple(range(30001, 30041)),
                action_transform_factory=lambda: (
                    evaluated_runtime.make_evaluation_snapshot()
                    .evaluate_action
                ),
            )
            direct_result = bridge_step_environment(
                direct_runtime,
                FakeEnvironment(),
                observation,
                base_action,
            )
            evaluated_result = bridge_step_environment(
                evaluated_runtime,
                FakeEnvironment(),
                observation,
                base_action,
            )

        np.testing.assert_array_equal(
            direct_result[1], evaluated_result[1]
        )
        np.testing.assert_array_equal(
            direct_result[2], evaluated_result[2]
        )
        self.assert_nested_equal(direct_result[3], evaluated_result[3])
        self.assert_runtime_equal(direct_runtime, evaluated_runtime)

    def test_evaluation_cannot_change_next_online_update(self):
        direct_runtime = self.make_execution_runtime()
        evaluated_runtime = self.make_execution_runtime()
        direct_recent = self.make_recent_buffer(4)
        evaluated_recent = self.make_recent_buffer(4)
        candidate_delta = np.asarray([0.08, -0.04], dtype=np.float32)

        def correct_actions(bridge, observation, proposed_action):
            del bridge, observation
            return (
                np.asarray(proposed_action) + candidate_delta,
                self.correction_metrics(),
            )

        with mock.patch.object(
            DynamicsShiftBridge,
            "correct_actions",
            new=correct_actions,
        ):
            call_preserving_global_numpy_rng(
                evaluate,
                agent=EvaluationAgent(),
                env=EvaluationEnvironment(),
                num_eval_episodes=2,
                action_dim=2,
                episode_seeds=(30001, 30002),
                action_transform_factory=lambda: (
                    evaluated_runtime.make_evaluation_snapshot()
                    .evaluate_action
                ),
            )

        self.assertTrue(direct_runtime.maybe_update_online(4, direct_recent))
        self.assertTrue(
            evaluated_runtime.maybe_update_online(4, evaluated_recent)
        )
        self.assert_runtime_equal(direct_runtime, evaluated_runtime)

    def test_all_execution_gates_are_required(self):
        base = np.zeros(2, dtype=np.float32)
        candidate = np.asarray([0.08, -0.04], dtype=np.float32)

        runtime = self.make_execution_runtime()
        runtime.bridge_online_ready = False
        executed = runtime.environment_action(
            base, candidate, self.correction_metrics()
        )
        np.testing.assert_array_equal(executed, base)
        self.assertEqual(
            runtime.last_execution_metrics["gate_readiness"], 0
        )

        runtime = self.make_execution_runtime()
        runtime.online_eval["normalized_mse"] = np.float32(0.11)
        runtime.environment_action(
            base, candidate, self.correction_metrics()
        )
        self.assertEqual(
            runtime.last_execution_metrics[
                "gate_model_confident"
            ],
            0,
        )

        runtime = self.make_execution_runtime(
            gate_uncertainty_multiplier=2.0
        )
        runtime.environment_action(
            base,
            candidate,
            self.correction_metrics(pre=0.044, post=0.010),
        )
        self.assertAlmostEqual(
            float(runtime.last_execution_metrics["shift_excess"]),
            0.004,
            places=6,
        )
        self.assertEqual(
            runtime.last_execution_metrics["gate_shift_detected"], 0
        )

        runtime = self.make_execution_runtime()
        runtime.environment_action(
            base,
            candidate,
            self.correction_metrics(pre=0.10, post=0.085),
        )
        self.assertAlmostEqual(
            float(
                runtime.last_execution_metrics[
                    "relative_match_improvement"
                ]
            ),
            0.15,
            places=6,
        )
        self.assertEqual(
            runtime.last_execution_metrics[
                "gate_correction_quality"
            ],
            0,
        )

        runtime = self.make_execution_runtime(
            gate_min_relative_improvement=0.0
        )
        runtime.environment_action(
            base,
            candidate,
            self.correction_metrics(pre=0.10, post=0.11),
        )
        self.assertLess(
            runtime.last_execution_metrics[
                "relative_match_improvement"
            ],
            0,
        )
        self.assertEqual(
            runtime.last_execution_metrics[
                "gate_correction_quality"
            ],
            0,
        )

        runtime = self.make_execution_runtime()
        executed = runtime.environment_action(
            base, candidate, self.correction_metrics()
        )
        self.assertEqual(
            runtime.last_execution_metrics["gate_open"], 1
        )
        self.assertEqual(
            runtime.last_execution_metrics["correction_requested"], 1
        )
        self.assertFalse(np.array_equal(executed, base))

    def test_nominal_uncertainty_safeguard_does_not_read_gain(self):
        runtime = self.make_execution_runtime()
        runtime.environment_action(
            np.zeros(2, dtype=np.float32),
            np.asarray([0.08, -0.04], dtype=np.float32),
            self.correction_metrics(pre=0.024, post=0.010),
        )

        self.assertEqual(
            runtime.last_execution_metrics["gate_open"], 0
        )
        self.assertEqual(runtime.bridge_applied_steps, 0)

    def test_residual_ramp_progresses_only_on_open_gate(self):
        runtime = self.make_execution_runtime()
        base = np.zeros(2, dtype=np.float32)
        candidate = np.asarray([0.08, -0.04], dtype=np.float32)
        metrics = self.correction_metrics()

        first = runtime.environment_action(base, candidate, metrics)
        np.testing.assert_allclose(
            first, base + 0.25 * (candidate - base)
        )
        self.assertEqual(runtime.bridge_applied_steps, 1)
        self.assertAlmostEqual(
            float(runtime.last_execution_metrics["ramp_scale"]),
            0.25,
        )

        second = runtime.environment_action(base, candidate, metrics)
        np.testing.assert_allclose(
            second, base + 0.50 * (candidate - base)
        )
        self.assertEqual(runtime.bridge_applied_steps, 2)

        runtime.online_eval["normalized_mse"] = np.float32(0.2)
        closed = runtime.environment_action(base, candidate, metrics)
        np.testing.assert_array_equal(closed, base)
        self.assertEqual(runtime.bridge_applied_steps, 2)
        self.assertEqual(
            runtime.last_execution_metrics["ramp_scale"], 0
        )

        runtime.online_eval["normalized_mse"] = np.float32(0.02)
        reopened = runtime.environment_action(base, candidate, metrics)
        np.testing.assert_allclose(
            reopened, base + 0.75 * (candidate - base)
        )
        self.assertEqual(runtime.bridge_applied_steps, 3)

        runtime.environment_action(base, candidate, metrics)
        capped = runtime.environment_action(base, candidate, metrics)
        np.testing.assert_allclose(capped, candidate)
        self.assertEqual(
            runtime.last_execution_metrics["ramp_scale"], 1
        )

    def test_zero_ramp_steps_means_immediate_configured_scale(self):
        runtime = self.make_execution_runtime(
            apply_ramp_steps=0,
            apply_residual_scale=0.5,
        )
        base = np.zeros(2, dtype=np.float32)
        candidate = np.asarray([0.08, -0.04], dtype=np.float32)

        executed = runtime.environment_action(
            base, candidate, self.correction_metrics()
        )

        np.testing.assert_allclose(
            executed, base + 0.5 * (candidate - base)
        )
        self.assertEqual(
            runtime.last_execution_metrics["ramp_scale"], 1
        )

    def test_applied_marker_requires_an_actual_action_change(self):
        runtime = self.make_execution_runtime(apply_ramp_steps=0)
        base = np.asarray([0.1, -0.1], dtype=np.float32)

        executed = runtime.environment_action(
            base, base.copy(), self.correction_metrics()
        )

        np.testing.assert_array_equal(executed, base)
        self.assertEqual(
            runtime.last_execution_metrics["correction_requested"], 1
        )
        self.assertEqual(
            runtime.last_execution_metrics[
                "correction_applied_to_environment"
            ],
            0,
        )

    def test_executed_action_respects_action_and_residual_bounds(self):
        runtime = self.make_execution_runtime(apply_ramp_steps=0)
        base = np.asarray([0.98, -0.98], dtype=np.float32)
        candidate = np.asarray([1.5, -1.5], dtype=np.float32)

        executed = runtime.environment_action(
            base, candidate, self.correction_metrics()
        )
        residual = executed - base

        self.assertTrue(np.all(executed <= 1.0))
        self.assertTrue(np.all(executed >= -1.0))
        self.assertTrue(np.all(np.abs(residual) <= 0.1))
        self.assertEqual(
            runtime.last_execution_metrics[
                "correction_applied_to_environment"
            ],
            1,
        )
        self.assertGreater(
            runtime.last_execution_metrics["candidate_residual_l2"],
            runtime.last_execution_metrics["executed_residual_l2"],
        )

    def test_shadow_correction_executes_base_primitive_action(self):
        runtime = self.make_runtime()
        runtime.maybe_update_online(4, self.make_recent_buffer(4))
        environment = FakeEnvironment()
        observation = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        base_action = np.asarray([0.25, -0.125], dtype=np.float64)

        (
            _,
            executed_action,
            corrected_action,
            metrics,
        ) = shadow_step_environment(
            runtime, environment, observation, base_action
        )

        np.testing.assert_array_equal(executed_action, base_action)
        np.testing.assert_array_equal(environment.actions[0], base_action)
        self.assertEqual(executed_action.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(
            runtime.last_base_action, base_action
        )
        np.testing.assert_array_equal(
            runtime.last_corrected_action, corrected_action
        )
        np.testing.assert_array_equal(
            runtime.last_executed_action, base_action
        )
        for value in metrics.values():
            self.assertTrue(np.isfinite(value))

        with self.assertRaisesRegex(ValueError, "shadow_step_environment"):
            shadow_step_environment(
                self.make_execution_runtime(),
                environment,
                observation,
                base_action,
            )

    def test_replay_and_recent_buffers_store_executed_base_action(self):
        runtime = self.make_runtime()
        runtime.maybe_update_online(4, self.make_recent_buffer(4))
        base_action = np.asarray([0.3, -0.2], dtype=np.float32)
        observation = np.zeros(3, dtype=np.float32)
        environment = FakeEnvironment()
        result, executed_action, _, _ = shadow_step_environment(
            runtime, environment, observation, base_action
        )
        next_observation = result[0]
        transition = {
            "observations": observation,
            "actions": executed_action,
            "rewards": np.float32(0.0),
            "terminals": np.float32(0.0),
            "masks": np.float32(1.0),
            "next_observations": next_observation,
        }
        replay = ReplayBuffer.create(transition, size=4)
        recent = RecentDynamicsBuffer.create(transition, capacity=4)

        replay.add_transition(transition)
        recent.add_transition(transition)

        np.testing.assert_array_equal(replay["actions"][0], base_action)
        np.testing.assert_array_equal(
            recent.ordered_data()["actions"][0], base_action
        )

    def test_execution_buffers_and_online_model_use_executed_action(self):
        runtime = self.make_execution_runtime(
            online_start_size=2,
            online_update_interval=2,
        )
        base_action = np.asarray([0.20, -0.20], dtype=np.float32)
        candidate = np.asarray([0.28, -0.24], dtype=np.float32)
        metrics = self.correction_metrics()
        environment = FakeEnvironment()

        with mock.patch.object(
            DynamicsShiftBridge,
            "correct_actions",
            return_value=(candidate, metrics),
        ):
            result, executed_action, corrected_action, _ = (
                bridge_step_environment(
                    runtime,
                    environment,
                    np.zeros(3, dtype=np.float32),
                    base_action,
                )
            )

        self.assertFalse(np.array_equal(executed_action, base_action))
        self.assertFalse(np.array_equal(executed_action, candidate))
        np.testing.assert_array_equal(corrected_action, candidate)
        np.testing.assert_array_equal(
            environment.actions[0], executed_action
        )
        transition = {
            "observations": np.zeros(3, dtype=np.float32),
            "actions": executed_action,
            "rewards": np.float32(0.0),
            "terminals": np.float32(0.0),
            "masks": np.float32(1.0),
            "next_observations": result[0],
        }
        replay = ReplayBuffer.create(transition, size=4)
        recent = RecentDynamicsBuffer.create(transition, capacity=4)
        replay.add_transition(transition)
        recent.add_transition(transition)
        dummy = {
            **transition,
            "actions": np.zeros(2, dtype=np.float32),
        }
        recent.add_transition(dummy)

        np.testing.assert_array_equal(
            replay["actions"][0], executed_action
        )
        np.testing.assert_array_equal(
            recent.ordered_data()["actions"][0], executed_action
        )
        self.assertFalse(
            np.array_equal(replay["actions"][0], candidate)
        )

        captured_actions = []
        original_update = DynamicsShiftBridge.update_online

        def capture_update(bridge, batch):
            captured_actions.append(
                np.asarray(batch["actions"]).copy()
            )
            return original_update(bridge, batch)

        with mock.patch.object(
            DynamicsShiftBridge,
            "update_online",
            new=capture_update,
        ):
            self.assertTrue(runtime.maybe_update_online(2, recent))

        self.assertTrue(captured_actions)
        for actions in captured_actions:
            np.testing.assert_array_equal(
                actions,
                np.broadcast_to(
                    executed_action, actions.shape
                ),
            )

    def test_correction_does_not_modify_future_action_queue(self):
        runtime = self.make_runtime()
        runtime.maybe_update_online(4, self.make_recent_buffer(4))
        queue = [
            np.asarray([0.1, -0.1], dtype=np.float32),
            np.asarray([0.2, -0.2], dtype=np.float32),
            np.asarray([0.3, -0.3], dtype=np.float32),
        ]
        base_action = queue.pop(0)
        future_before = [action.copy() for action in queue]

        shadow_step_environment(
            runtime,
            FakeEnvironment(),
            np.zeros(3, dtype=np.float32),
            base_action,
        )

        for before, after in zip(future_before, queue):
            np.testing.assert_array_equal(before, after)

    def test_execution_changes_only_current_primitive_queue_action(self):
        runtime = self.make_execution_runtime()
        queue = [
            np.asarray([0.1, -0.1], dtype=np.float32),
            np.asarray([0.2, -0.2], dtype=np.float32),
            np.asarray([0.3, -0.3], dtype=np.float32),
        ]
        base_action = queue.pop(0)
        future_before = [action.copy() for action in queue]
        candidate = np.asarray([0.18, -0.14], dtype=np.float32)

        with mock.patch.object(
            DynamicsShiftBridge,
            "correct_actions",
            return_value=(candidate, self.correction_metrics()),
        ):
            _, executed, _, _ = bridge_step_environment(
                runtime,
                FakeEnvironment(),
                np.zeros(3, dtype=np.float32),
                base_action,
            )

        self.assertFalse(np.array_equal(executed, base_action))
        for before, after in zip(future_before, queue):
            np.testing.assert_array_equal(before, after)

    def test_single_step_and_chunk_queue_share_primitive_boundary(self):
        runtime = self.make_runtime()
        runtime.maybe_update_online(4, self.make_recent_buffer(4))
        observation = np.zeros(3, dtype=np.float32)
        single_action = np.asarray([0.1, 0.2], dtype=np.float32)
        chunk = np.asarray(
            [[0.1, 0.2], [0.3, 0.4], [-0.2, 0.1]],
            dtype=np.float32,
        )

        single_env = FakeEnvironment()
        shadow_step_environment(
            runtime, single_env, observation, single_action
        )
        queue = [action.copy() for action in chunk]
        chunk_env = FakeEnvironment()
        first_base = queue.pop(0)
        future_before = [action.copy() for action in queue]
        shadow_step_environment(
            runtime, chunk_env, observation, first_base
        )

        np.testing.assert_array_equal(
            single_env.actions[0], single_action
        )
        np.testing.assert_array_equal(chunk_env.actions[0], chunk[0])
        for before, after in zip(future_before, queue):
            np.testing.assert_array_equal(before, after)

    def test_correction_metrics_are_aggregated_and_reset(self):
        runtime = self.make_runtime()
        runtime.maybe_update_online(4, self.make_recent_buffer(4))
        observation = np.zeros(3, dtype=np.float32)
        actions = (
            np.asarray([0.1, 0.2], dtype=np.float32),
            np.asarray([-0.2, 0.3], dtype=np.float32),
        )
        recorded = []
        for action in actions:
            _, metrics = runtime.shadow_correct(observation, action)
            recorded.append(metrics)

        row = runtime.log_row(6, 4)

        for name in CORRECTION_METRIC_FIELDS:
            self.assertAlmostEqual(
                float(row[name]),
                float(np.mean([metrics[name] for metrics in recorded])),
                places=6,
            )
        reset_row = runtime.log_row(7, 4)
        for name in CORRECTION_METRIC_FIELDS:
            self.assertEqual(float(reset_row[name]), 0.0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            runtime.log_row(7, 4)

    def test_execution_rates_are_aggregated_and_reset(self):
        runtime = self.make_execution_runtime()
        base = np.zeros(2, dtype=np.float32)
        candidate = np.asarray([0.08, -0.04], dtype=np.float32)
        metrics = self.correction_metrics()
        runtime.environment_action(base, candidate, metrics)
        runtime.online_eval["normalized_mse"] = np.float32(0.2)
        runtime.environment_action(base, candidate, metrics)

        row = runtime.log_row(2, 2)

        self.assertEqual(row["gate_open"], 0)
        self.assertEqual(row["correction_applied_to_environment"], 0)
        self.assertAlmostEqual(row["gate_open_fraction"], 0.5)
        self.assertAlmostEqual(
            row["correction_requested_fraction"], 0.5
        )
        self.assertAlmostEqual(
            row["correction_applied_fraction"], 0.5
        )
        self.assertEqual(row["bridge_applied_steps"], 1)

        empty = runtime.log_row(3, 3)
        for name in (
            *EXECUTION_METRIC_FIELDS,
            "gate_open_fraction",
            "correction_requested_fraction",
            "correction_applied_fraction",
        ):
            self.assertEqual(float(empty[name]), 0.0)
        self.assertEqual(empty["bridge_applied_steps"], 1)

    def test_invalid_clip_fraction_is_rejected(self):
        runtime = self.make_runtime()
        runtime.maybe_update_online(4, self.make_recent_buffer(4))
        metrics = {
            name: np.float32(0.0)
            for name in CORRECTION_METRIC_FIELDS
        }
        metrics["action_clip_fraction"] = np.float32(1.1)

        with mock.patch.object(
            DynamicsShiftBridge,
            "correct_actions",
            return_value=(
                np.zeros(2, dtype=np.float32),
                metrics,
            ),
        ):
            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                runtime.shadow_correct(
                    np.zeros(3, dtype=np.float32),
                    np.zeros(2, dtype=np.float32),
                )

    def test_csv_schema_and_values_are_finite(self):
        runtime = self.make_execution_runtime()
        with mock.patch.object(
            DynamicsShiftBridge,
            "correct_actions",
            return_value=(
                np.asarray([0.08, -0.04], dtype=np.float32),
                self.correction_metrics(),
            ),
        ):
            bridge_step_environment(
                runtime,
                FakeEnvironment(),
                np.zeros(3, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            )
        row = runtime.log_row(4, 4, reset=False)

        self.assertEqual(tuple(row), BRIDGE_LOG_FIELDS)
        self.assertEqual(row["correction_applied_to_environment"], 1)
        self.assertEqual(row["correction_requested"], 1)
        self.assertEqual(row["gate_open"], 1)
        self.assertEqual(row["gate_open_fraction"], 1)
        self.assertEqual(row["correction_requested_fraction"], 1)
        self.assertEqual(row["correction_applied_fraction"], 1)
        self.assertEqual(row["bridge_applied_steps"], 1)
        self.assertGreater(
            row["candidate_residual_l2"],
            row["executed_residual_l2"],
        )
        for name in (
            "gate_readiness",
            "gate_model_confident",
            "gate_shift_detected",
            "gate_correction_quality",
            "gate_open",
            "correction_requested",
            "correction_applied_to_environment",
        ):
            self.assertIn(float(row[name]), (0.0, 1.0))
        for name in EXECUTION_METRIC_FIELDS:
            self.assertTrue(np.isfinite(row[name]))
        for value in row.values():
            self.assertTrue(np.isfinite(value))

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(
                directory, "dynamics_shift_bridge.csv"
            )
            logger = CsvLogger(path)
            logger.log(dict(row), step=1004)
            logger.close()
            with open(path, "r") as file:
                header = file.readline().strip().split(",")
            self.assertEqual(
                header, [*BRIDGE_LOG_FIELDS, "step"]
            )

    def test_main_preserves_direct_env_step_when_bridge_is_disabled(self):
        source = (
            Path(__file__).resolve().parents[1] / "main.py"
        ).read_text()
        disabled_branch = source.index("if bridge_runtime is None:")
        direct_step = source.index("env.step(", disabled_branch)
        bridge_branch = source.index("bridge_step_environment(", direct_step)

        self.assertLess(disabled_branch, direct_step)
        self.assertLess(direct_step, bridge_branch)

        extraction_guard = source.index(
            "if bridge_runtime_config.enabled:"
        )
        primitive_extraction = source.index(
            "bridge_primitive_transitions = extract_primitive_transitions"
        )
        first_sequence_sample = source.index(
            "train_dataset.sample_sequence("
        )
        self.assertLess(extraction_guard, primitive_extraction)
        self.assertLess(primitive_extraction, first_sequence_sample)


if __name__ == "__main__":
    unittest.main()
