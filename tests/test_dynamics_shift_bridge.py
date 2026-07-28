import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

from utils.dynamics_shift_bridge import (
    DynamicsModel,
    DynamicsShiftBridge,
    DynamicsShiftBridgeConfig,
)


class DynamicsShiftBridgeTest(unittest.TestCase):
    _linear_results = {}
    _nonlinear_result = None

    def assert_trees_allclose(self, actual, expected, **kwargs):
        actual_leaves = jax.tree_util.tree_leaves(actual)
        expected_leaves = jax.tree_util.tree_leaves(expected)
        self.assertEqual(len(actual_leaves), len(expected_leaves))
        for actual_leaf, expected_leaf in zip(
            actual_leaves, expected_leaves
        ):
            np.testing.assert_allclose(
                np.asarray(actual_leaf),
                np.asarray(expected_leaf),
                **kwargs,
            )

    def assert_trees_do_not_alias(self, first, second):
        first_leaves = jax.tree_util.tree_leaves(first)
        second_leaves = jax.tree_util.tree_leaves(second)
        self.assertEqual(len(first_leaves), len(second_leaves))
        for first_leaf, second_leaf in zip(first_leaves, second_leaves):
            self.assertIsNot(first_leaf, second_leaf)

    def assert_scalar_finite_float32_metrics(self, metrics):
        for name, value in metrics.items():
            array = np.asarray(value)
            self.assertEqual(array.shape, (), msg=name)
            self.assertEqual(array.dtype, np.dtype(np.float32), msg=name)
            self.assertTrue(np.isfinite(array), msg=name)

    def linear_transition_batch(self, batch_size=32, gain=1.0):
        actions = np.linspace(
            -0.8, 0.8, batch_size * 2, dtype=np.float32
        ).reshape(batch_size, 2)
        observations = np.stack(
            (
                np.linspace(10.0, 11.0, batch_size),
                np.linspace(-8.0, -7.0, batch_size),
                np.linspace(4.0, 5.0, batch_size),
            ),
            axis=-1,
        ).astype(np.float32)
        matrix = np.array(
            [[0.7, -0.2], [0.1, 0.5], [-0.3, 0.4]],
            dtype=np.float32,
        )
        deltas = (gain * actions) @ matrix.T
        return {
            "observations": observations,
            "actions": actions,
            "next_observations": observations + deltas,
        }

    @classmethod
    def trained_linear_result(cls, gain):
        if gain in cls._linear_results:
            return cls._linear_results[gain]
        generator = np.random.default_rng(123)
        matrix = np.array(
            [[0.7, -0.2], [0.1, 0.5], [-0.3, 0.4]],
            dtype=np.float32,
        )
        observations = generator.uniform(
            -0.5, 0.5, (256, 3)
        ).astype(np.float32)
        actions = generator.uniform(
            -0.7, 0.7, (256, 2)
        ).astype(np.float32)
        offline_batch = {
            "observations": observations,
            "actions": actions,
            "next_observations": observations + actions @ matrix.T,
        }
        config = DynamicsShiftBridgeConfig(
            hidden_dim=32,
            num_hidden_layers=2,
            learning_rate=3e-3,
            correction_steps=40,
            correction_step_size=0.2,
            dynamics_match_weight=1.0,
            action_l2_weight=1e-3,
            max_residual=0.5,
        )
        bridge = DynamicsShiftBridge.create(
            seed=5,
            example_observations=observations,
            example_actions=actions,
            config=config,
        )
        for _ in range(500):
            bridge, _ = bridge.update_offline(offline_batch)
        bridge = bridge.synchronize_online_from_offline()
        online_batch = {
            **offline_batch,
            "next_observations": (
                observations + (gain * actions) @ matrix.T
            ),
        }
        for _ in range(300):
            bridge, _ = bridge.update_online(online_batch)

        held_observations = generator.uniform(
            -0.5, 0.5, (64, 3)
        ).astype(np.float32)
        base_actions = (
            generator.uniform(-0.7, 0.7, (64, 2)).astype(np.float32)
            * 0.6
        )
        corrected_actions, metrics = bridge.correct_actions(
            held_observations, base_actions
        )
        result = (
            bridge,
            held_observations,
            base_actions,
            np.asarray(corrected_actions),
            metrics,
        )
        cls._linear_results[gain] = result
        return result

    @classmethod
    def trained_nonlinear_result(cls):
        if cls._nonlinear_result is not None:
            return cls._nonlinear_result
        generator = np.random.default_rng(321)
        state_matrix = np.array(
            [
                [0.4, -0.2, 0.1],
                [0.1, 0.3, -0.4],
                [-0.2, 0.2, 0.5],
            ],
            dtype=np.float32,
        )
        action_matrix = np.array(
            [[0.6, -0.1], [0.2, 0.5], [-0.4, 0.3]],
            dtype=np.float32,
        )

        def delta(observations, actions, gain):
            return np.tanh(
                observations @ state_matrix.T
                + (gain * actions) @ action_matrix.T
            ).astype(np.float32)

        observations = generator.uniform(
            -0.5, 0.5, (512, 3)
        ).astype(np.float32)
        actions = generator.uniform(
            -0.7, 0.7, (512, 2)
        ).astype(np.float32)
        offline_batch = {
            "observations": observations,
            "actions": actions,
            "next_observations": (
                observations + delta(observations, actions, 1.0)
            ),
        }
        bridge = DynamicsShiftBridge.create(
            seed=21,
            example_observations=observations,
            example_actions=actions,
            config=DynamicsShiftBridgeConfig(
                hidden_dim=64,
                num_hidden_layers=2,
                learning_rate=3e-3,
                correction_steps=60,
                correction_step_size=0.15,
                action_l2_weight=1e-3,
                max_residual=0.5,
            ),
        )
        for _ in range(700):
            bridge, _ = bridge.update_offline(offline_batch)
        bridge = bridge.synchronize_online_from_offline()
        online_batch = {
            **offline_batch,
            "next_observations": (
                observations + delta(observations, actions, 0.7)
            ),
        }
        for _ in range(500):
            bridge, _ = bridge.update_online(online_batch)
        held_observations = generator.uniform(
            -0.5, 0.5, (128, 3)
        ).astype(np.float32)
        base_actions = generator.uniform(
            -0.4, 0.4, (128, 2)
        ).astype(np.float32)
        corrected_actions, metrics = bridge.correct_actions(
            held_observations, base_actions
        )
        cls._nonlinear_result = (
            bridge,
            held_observations,
            base_actions,
            np.asarray(corrected_actions),
            metrics,
        )
        return cls._nonlinear_result

    def test_create_builds_independent_models_with_expected_output_shape(self):
        observations = np.zeros((4, 3), dtype=np.float64)
        actions = np.zeros((4, 2), dtype=np.float64)

        bridge = DynamicsShiftBridge.create(
            seed=7,
            example_observations=observations,
            example_actions=actions,
            config=DynamicsShiftBridgeConfig(
                hidden_dim=8,
                num_hidden_layers=2,
            ),
        )

        offline_prediction = bridge.predict_offline(
            observations, actions
        )
        online_prediction = bridge.predict_online(observations, actions)

        self.assertIsNot(bridge.offline_model, bridge.online_model)
        self.assertEqual(offline_prediction.shape, observations.shape)
        self.assertEqual(online_prediction.shape, observations.shape)
        self.assertEqual(offline_prediction.dtype, np.dtype(np.float32))
        self.assertEqual(online_prediction.dtype, np.dtype(np.float32))

    def test_initial_parameter_shapes_and_zero_initialized_output(self):
        batch = self.linear_transition_batch(batch_size=4)
        bridge = DynamicsShiftBridge.create(
            seed=41,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(
                hidden_dim=8,
                num_hidden_layers=2,
            ),
        )
        params = bridge.offline_model.params

        self.assertEqual(params["hidden_0"]["kernel"].shape, (5, 8))
        self.assertEqual(params["hidden_1"]["kernel"].shape, (8, 8))
        self.assertEqual(params["delta"]["kernel"].shape, (8, 3))
        np.testing.assert_array_equal(
            bridge.predict_offline(
                batch["observations"], batch["actions"]
            ),
            np.zeros_like(batch["observations"]),
        )

    def test_dynamics_model_flattens_structured_observations_and_actions(self):
        observations = jnp.zeros((4, 2, 3), dtype=jnp.float32)
        actions = jnp.zeros((4, 2, 2), dtype=jnp.float32)
        model = DynamicsModel(
            observation_dim=6,
            hidden_dim=8,
            num_hidden_layers=2,
        )
        params = model.init(
            jax.random.PRNGKey(59), observations, actions
        )

        predictions = model.apply(params, observations, actions)

        self.assertEqual(predictions.shape, (4, 6))
        self.assertEqual(predictions.dtype, np.dtype(np.float32))

    def test_float64_correction_inputs_are_explicitly_cast_to_float32(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )

        corrected_actions, metrics = bridge.correct_actions(
            observations.astype(np.float64),
            base_actions.astype(np.float64),
        )

        self.assertEqual(corrected_actions.dtype, np.dtype(np.float32))
        for value in metrics.values():
            self.assertEqual(np.asarray(value).dtype, np.dtype(np.float32))

    def test_offline_update_uses_delta_target_and_reduces_loss(self):
        batch = self.linear_transition_batch()
        bridge = DynamicsShiftBridge.create(
            seed=11,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(
                hidden_dim=16,
                learning_rate=1e-2,
            ),
        )

        bridge, first_metrics = bridge.update_offline(batch)
        for _ in range(100):
            bridge, final_metrics = bridge.update_offline(batch)

        expected_target_abs_mean = np.mean(
            np.abs(
                batch["next_observations"] - batch["observations"]
            )
        )
        self.assertAlmostEqual(
            float(first_metrics["target_abs_mean"]),
            float(expected_target_abs_mean),
            places=6,
        )
        self.assertLess(
            float(final_metrics["loss"]),
            0.1 * float(first_metrics["loss"]),
        )
        self.assertAlmostEqual(
            float(final_metrics["loss"]),
            float(final_metrics["normalized_prediction_mse"]),
            places=7,
        )
        self.assertTrue(np.isfinite(float(final_metrics["raw_prediction_mse"])))

    def test_online_sync_resets_optimizer_and_update_keeps_offline_frozen(self):
        offline_batch = self.linear_transition_batch(gain=1.0)
        online_batch = self.linear_transition_batch(gain=0.7)
        bridge = DynamicsShiftBridge.create(
            seed=13,
            example_observations=offline_batch["observations"],
            example_actions=offline_batch["actions"],
            config=DynamicsShiftBridgeConfig(
                hidden_dim=16,
                learning_rate=1e-2,
            ),
        )
        for _ in range(20):
            bridge, _ = bridge.update_offline(offline_batch)

        offline_before = flax.serialization.to_state_dict(
            bridge.offline_model
        )
        bridge = bridge.synchronize_online_from_offline()

        self.assert_trees_allclose(
            bridge.online_model.params, bridge.offline_model.params
        )
        self.assertEqual(bridge.online_model.step, 1)
        self.assertGreater(bridge.offline_model.step, 1)
        self.assertIsNot(
            bridge.online_model.opt_state, bridge.offline_model.opt_state
        )

        online_before = bridge.online_model.params
        bridge, _ = bridge.update_online(online_batch)

        self.assert_trees_allclose(
            flax.serialization.to_state_dict(bridge.offline_model),
            offline_before,
        )
        changed = any(
            not np.array_equal(np.asarray(before), np.asarray(after))
            for before, after in zip(
                jax.tree_util.tree_leaves(online_before),
                jax.tree_util.tree_leaves(bridge.online_model.params),
            )
        )
        self.assertTrue(changed)

    def test_offline_and_online_updates_are_isolated_in_both_directions(self):
        batch = self.linear_transition_batch(batch_size=8)
        bridge = DynamicsShiftBridge.create(
            seed=61,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(hidden_dim=8),
        )
        online_before = flax.serialization.to_state_dict(
            bridge.online_model
        )

        bridge, offline_metrics = bridge.update_offline(batch)

        self.assert_trees_allclose(
            flax.serialization.to_state_dict(bridge.online_model),
            online_before,
        )
        offline_before = flax.serialization.to_state_dict(
            bridge.offline_model
        )

        bridge, online_metrics = bridge.update_online(batch)

        self.assert_trees_allclose(
            flax.serialization.to_state_dict(bridge.offline_model),
            offline_before,
        )
        self.assertEqual(set(offline_metrics), set(online_metrics))

    def test_online_synchronization_makes_parameters_equal(self):
        batch = self.linear_transition_batch(batch_size=8)
        bridge = DynamicsShiftBridge.create(
            seed=43,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(hidden_dim=8),
        )
        bridge, _ = bridge.update_offline(batch)

        synchronized = bridge.synchronize_online_from_offline()

        self.assert_trees_allclose(
            synchronized.online_model.params,
            synchronized.offline_model.params,
        )
        self.assertIsNot(
            synchronized.online_model.params,
            synchronized.offline_model.params,
        )
        self.assert_trees_do_not_alias(
            synchronized.online_model.params,
            synchronized.offline_model.params,
        )

    def test_online_synchronization_reinitializes_optimizer_state(self):
        batch = self.linear_transition_batch(batch_size=8)
        bridge = DynamicsShiftBridge.create(
            seed=47,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(hidden_dim=8),
        )
        for _ in range(3):
            bridge, _ = bridge.update_offline(batch)

        synchronized = bridge.synchronize_online_from_offline()

        self.assertEqual(synchronized.online_model.step, 1)
        self.assertNotEqual(
            synchronized.online_model.step,
            synchronized.offline_model.step,
        )
        self.assertIsNot(
            synchronized.online_model.opt_state,
            synchronized.offline_model.opt_state,
        )

    def test_zero_correction_steps_is_exact_noop_for_single_and_batch(self):
        batch = self.linear_transition_batch(batch_size=4)
        bridge = DynamicsShiftBridge.create(
            seed=17,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(
                hidden_dim=8,
                correction_steps=0,
            ),
        )
        observations_before = batch["observations"].copy()
        actions_before = batch["actions"].copy()

        corrected_batch, metrics = bridge.correct_actions(
            batch["observations"], batch["actions"]
        )
        corrected_single, _ = bridge.correct_actions(
            batch["observations"][0], batch["actions"][0]
        )

        np.testing.assert_array_equal(corrected_batch, actions_before)
        np.testing.assert_array_equal(corrected_single, actions_before[0])
        np.testing.assert_array_equal(
            batch["observations"], observations_before
        )
        np.testing.assert_array_equal(batch["actions"], actions_before)
        self.assertAlmostEqual(
            float(metrics["match_improvement"]),
            float(metrics["pre_match_mse"] - metrics["post_match_mse"]),
            places=7,
        )
        for value in metrics.values():
            self.assertTrue(np.all(np.isfinite(np.asarray(value))))

    def test_all_disabled_correction_modes_are_exact_noops(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        cases = {
            "correction_step_size": bridge.replace(
                config=flax.core.freeze(
                    {
                        **dict(bridge.config),
                        "correction_step_size": 0.0,
                    }
                )
            ),
            "dynamics_match_weight": bridge.replace(
                config=flax.core.freeze(
                    {
                        **dict(bridge.config),
                        "dynamics_match_weight": 0.0,
                    }
                )
            ),
            "max_residual": bridge.replace(
                max_residual=jnp.zeros_like(bridge.max_residual)
            ),
        }

        for name, disabled_bridge in cases.items():
            with self.subTest(name=name):
                corrected, metrics = disabled_bridge.correct_actions(
                    observations, base_actions
                )
                np.testing.assert_array_equal(corrected, base_actions)
                self.assertEqual(float(metrics["match_improvement"]), 0.0)
                self.assertEqual(
                    float(metrics["match_improvement_raw"]), 0.0
                )
                self.assert_scalar_finite_float32_metrics(metrics)

    def test_nominal_correction_is_near_noop_and_does_not_worsen_match(self):
        _, _, base_actions, corrected_actions, metrics = (
            self.trained_linear_result(1.0)
        )

        np.testing.assert_allclose(
            corrected_actions, base_actions, atol=3e-2, rtol=0.0
        )
        self.assertLess(
            float(metrics["residual_l2_mean"]), 3e-2
        )
        self.assertLessEqual(
            float(metrics["post_match_mse"]),
            float(metrics["pre_match_mse"]) + 1e-8,
        )

    def test_gain_0p7_correction_significantly_reduces_match_mse(self):
        _, _, _, _, metrics = self.trained_linear_result(0.7)

        self.assertLessEqual(
            float(metrics["post_match_mse"]),
            0.5 * float(metrics["pre_match_mse"]),
        )

    def test_gain_0p7_correction_increases_action_magnitude(self):
        _, _, base_actions, corrected_actions, _ = (
            self.trained_linear_result(0.7)
        )
        residual = corrected_actions - base_actions

        self.assertGreater(
            float(np.mean(np.sum(residual * base_actions, axis=-1))),
            0.0,
        )
        self.assertGreater(
            float(np.mean(np.linalg.norm(corrected_actions, axis=-1))),
            float(np.mean(np.linalg.norm(base_actions, axis=-1))),
        )

    def test_gain_1p3_correction_reduces_match_mse(self):
        _, _, _, _, metrics = self.trained_linear_result(1.3)

        self.assertLess(
            float(metrics["post_match_mse"]),
            float(metrics["pre_match_mse"]),
        )

    def test_gain_1p3_correction_decreases_action_magnitude(self):
        _, _, base_actions, corrected_actions, _ = (
            self.trained_linear_result(1.3)
        )
        residual = corrected_actions - base_actions

        self.assertLess(
            float(np.mean(np.sum(residual * base_actions, axis=-1))),
            0.0,
        )
        self.assertLess(
            float(np.mean(np.linalg.norm(corrected_actions, axis=-1))),
            float(np.mean(np.linalg.norm(base_actions, axis=-1))),
        )

    def test_nonlinear_dynamics_correction_reduces_held_out_match_mse(self):
        _, _, _, _, metrics = self.trained_nonlinear_result()

        self.assertLess(
            float(metrics["post_match_mse"]),
            float(metrics["pre_match_mse"]),
        )

    def test_correction_respects_action_and_residual_bounds(self):
        bridge, _, base_actions, corrected_actions, _ = (
            self.trained_linear_result(0.7)
        )
        flattened_residual = np.abs(corrected_actions - base_actions)

        self.assertTrue(
            np.all(corrected_actions >= np.asarray(bridge.action_low))
        )
        self.assertTrue(
            np.all(corrected_actions <= np.asarray(bridge.action_high))
        )
        self.assertTrue(
            np.all(
                flattened_residual
                <= np.asarray(bridge.max_residual) + 1e-7
            )
        )

    def test_residual_clip_fraction_reports_active_clipping(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        bridge = bridge.replace(
            max_residual=jnp.full_like(bridge.max_residual, 0.01)
        )

        corrected_actions, metrics = bridge.correct_actions(
            observations, base_actions
        )

        self.assertGreater(float(metrics["residual_clip_fraction"]), 0.0)
        self.assertLessEqual(
            float(np.max(np.abs(corrected_actions - base_actions))),
            0.01 + 1e-7,
        )

    def test_action_clip_fraction_reports_active_clipping(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        bridge = bridge.replace(
            action_low=jnp.full_like(bridge.action_low, -0.2),
            action_high=jnp.full_like(bridge.action_high, 0.2),
        )
        base_actions = np.full_like(base_actions, 0.19)

        corrected_actions, metrics = bridge.correct_actions(
            observations, base_actions
        )

        self.assertGreater(float(metrics["action_clip_fraction"]), 0.0)
        self.assertTrue(np.all(corrected_actions <= 0.2))
        self.assertTrue(np.all(corrected_actions >= -0.2))

    def test_asymmetric_action_bounds_preserve_both_bounds(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        action_low = jnp.asarray([-0.2, -1.0], dtype=jnp.float32)
        action_high = jnp.asarray([0.7, 0.4], dtype=jnp.float32)
        bridge = bridge.replace(
            action_low=action_low,
            action_high=action_high,
            max_residual=jnp.asarray([0.15, 0.08], dtype=jnp.float32),
        )
        base_actions = np.clip(
            base_actions,
            np.asarray(action_low) + 1e-3,
            np.asarray(action_high) - 1e-3,
        )

        corrected_actions, metrics = bridge.correct_actions(
            observations, base_actions
        )
        residual = np.abs(corrected_actions - base_actions)

        self.assertTrue(
            np.all(corrected_actions >= np.asarray(action_low))
        )
        self.assertTrue(
            np.all(corrected_actions <= np.asarray(action_high))
        )
        self.assertTrue(
            np.all(residual <= np.asarray(bridge.max_residual) + 1e-7)
        )
        for name in ("action_clip_fraction", "residual_clip_fraction"):
            self.assertGreaterEqual(float(metrics[name]), 0.0)
            self.assertLessEqual(float(metrics[name]), 1.0)

    def test_batch_examples_do_not_influence_other_corrected_actions(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        changed_observations = observations.copy()
        changed_actions = base_actions.copy()
        changed_observations[0] += np.asarray(
            [0.25, -0.4, 0.15], dtype=np.float32
        )
        changed_actions[0] *= -0.5

        original, _ = bridge.correct_actions(
            observations, base_actions
        )
        changed, _ = bridge.correct_actions(
            changed_observations, changed_actions
        )

        np.testing.assert_array_equal(original[1:], changed[1:])

    def test_zero_max_residual_is_exact_noop(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        bridge = bridge.replace(
            max_residual=jnp.zeros_like(bridge.max_residual)
        )

        corrected_actions, metrics = bridge.correct_actions(
            observations, base_actions
        )

        np.testing.assert_array_equal(corrected_actions, base_actions)
        self.assertEqual(float(metrics["residual_abs_max"]), 0.0)

    def test_zero_dynamics_match_weight_cannot_create_correction(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        bridge = bridge.replace(
            config=flax.core.freeze(
                {
                    **dict(bridge.config),
                    "dynamics_match_weight": 0.0,
                }
            )
        )

        corrected_actions, _ = bridge.correct_actions(
            observations, base_actions
        )

        np.testing.assert_array_equal(corrected_actions, base_actions)

    def test_nonfinite_inputs_are_rejected(self):
        batch = self.linear_transition_batch(batch_size=4)
        bridge = DynamicsShiftBridge.create(
            seed=23,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
        )
        invalid_observations = batch["observations"].copy()
        invalid_observations[0, 0] = np.nan
        invalid_transition = dict(batch)
        invalid_transition["next_observations"] = (
            batch["next_observations"].copy()
        )
        invalid_transition["next_observations"][0, 0] = np.inf

        with self.assertRaisesRegex(ValueError, "finite"):
            bridge.predict_offline(
                invalid_observations, batch["actions"]
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            bridge.update_online(invalid_transition)
        with self.assertRaisesRegex(ValueError, "finite"):
            bridge.correct_actions(
                invalid_observations, batch["actions"]
            )

    def test_shape_mismatches_are_rejected_without_broadcasting(self):
        batch = self.linear_transition_batch(batch_size=4)
        bridge = DynamicsShiftBridge.create(
            seed=29,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
        )

        with self.assertRaisesRegex(ValueError, "shape"):
            bridge.correct_actions(
                batch["observations"][:, :2], batch["actions"]
            )
        with self.assertRaisesRegex(ValueError, "batch size"):
            bridge.correct_actions(
                batch["observations"], batch["actions"][:3]
            )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            bridge.update_offline(
                {
                    **batch,
                    "next_observations": batch["next_observations"][
                        :, :2
                    ],
                }
            )

    def test_single_correction_preserves_action_shape(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )

        corrected_actions, _ = bridge.correct_actions(
            observations[0], base_actions[0]
        )

        self.assertEqual(corrected_actions.shape, bridge.action_shape)

    def test_batch_correction_preserves_action_shape(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )

        corrected_actions, _ = bridge.correct_actions(
            observations, base_actions
        )

        self.assertEqual(
            corrected_actions.shape,
            (observations.shape[0],) + bridge.action_shape,
        )

    def test_correction_does_not_modify_input_arrays(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        observations_before = observations.copy()
        base_actions_before = base_actions.copy()

        bridge.correct_actions(observations, base_actions)

        np.testing.assert_array_equal(observations, observations_before)
        np.testing.assert_array_equal(base_actions, base_actions_before)

    def test_correction_does_not_update_either_model_state(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        state_before = flax.serialization.to_state_dict(bridge)

        bridge.correct_actions(observations, base_actions)

        self.assert_trees_allclose(
            flax.serialization.to_state_dict(bridge),
            state_before,
        )

    def test_clip_fraction_is_per_component_averaged_over_steps(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        steps = 3
        bridge = bridge.replace(
            config=flax.core.freeze(
                {
                    **dict(bridge.config),
                    "correction_steps": steps,
                }
            ),
            max_residual=jnp.full_like(bridge.max_residual, 0.01),
            action_low=jnp.asarray([-0.2, -1.0], dtype=jnp.float32),
            action_high=jnp.asarray([0.7, 0.4], dtype=jnp.float32),
        )
        base_actions = np.clip(
            base_actions,
            np.asarray(bridge.action_low) + 1e-3,
            np.asarray(bridge.action_high) - 1e-3,
        )
        flat_observations, flat_base, _ = (
            bridge._prepare_correction_inputs(
                observations, base_actions
            )
        )
        target = jax.lax.stop_gradient(
            bridge._normalized_model_prediction(
                bridge.offline_model,
                flat_observations,
                flat_base,
            )
        )

        def objective(actions):
            prediction = bridge._normalized_model_prediction(
                bridge.online_model, flat_observations, actions
            )
            match = jnp.mean(
                jnp.square(prediction - target), axis=-1
            )
            regularization = jnp.mean(
                jnp.square(actions - flat_base), axis=-1
            )
            return jnp.sum(
                bridge.config["dynamics_match_weight"] * match
                + bridge.config["action_l2_weight"] * regularization
            )

        actions = flat_base
        expected_action_fraction = 0.0
        expected_residual_fraction = 0.0
        for _ in range(steps):
            proposed = (
                actions
                - bridge.config["correction_step_size"]
                * jax.grad(objective)(actions)
            )
            proposed_residual = proposed - flat_base
            clipped_residual = jnp.clip(
                proposed_residual,
                -bridge.max_residual,
                bridge.max_residual,
            )
            expected_residual_fraction += float(
                jnp.mean(proposed_residual != clipped_residual)
            )
            residual_bounded = flat_base + clipped_residual
            actions = jnp.clip(
                residual_bounded,
                bridge.action_low,
                bridge.action_high,
            )
            expected_action_fraction += float(
                jnp.mean(residual_bounded != actions)
            )

        _, metrics = bridge.correct_actions(
            observations, base_actions
        )

        self.assertAlmostEqual(
            float(metrics["action_clip_fraction"]),
            expected_action_fraction / steps,
            places=7,
        )
        self.assertAlmostEqual(
            float(metrics["residual_clip_fraction"]),
            expected_residual_fraction / steps,
            places=7,
        )

    def test_correction_metrics_are_complete_finite_and_consistent(self):
        _, _, _, _, metrics = self.trained_linear_result(0.7)
        required = {
            "pre_match_mse",
            "post_match_mse",
            "match_improvement",
            "residual_l2_mean",
            "residual_abs_max",
            "action_clip_fraction",
            "residual_clip_fraction",
            "pre_match_mse_normalized",
            "post_match_mse_normalized",
            "pre_match_mse_raw",
            "post_match_mse_raw",
            "match_improvement_raw",
        }

        self.assertTrue(required.issubset(metrics))
        self.assertAlmostEqual(
            float(metrics["match_improvement"]),
            float(metrics["pre_match_mse"] - metrics["post_match_mse"]),
            places=7,
        )
        self.assertAlmostEqual(
            float(metrics["match_improvement_raw"]),
            float(
                metrics["pre_match_mse_raw"]
                - metrics["post_match_mse_raw"]
            ),
            places=7,
        )
        self.assert_scalar_finite_float32_metrics(metrics)
        for name in ("action_clip_fraction", "residual_clip_fraction"):
            self.assertGreaterEqual(float(metrics[name]), 0.0)
            self.assertLessEqual(float(metrics[name]), 1.0)

    def test_correction_is_deterministic_and_jittable(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )

        corrected_first, metrics_first = bridge.correct_actions(
            observations, base_actions
        )
        corrected_second, metrics_second = bridge.correct_actions(
            observations, base_actions
        )
        jitted_correct = jax.jit(
            lambda obs, actions: bridge.correct_actions(obs, actions)
        )
        corrected_jitted, metrics_jitted = jitted_correct(
            jnp.asarray(observations), jnp.asarray(base_actions)
        )

        np.testing.assert_array_equal(
            corrected_first, corrected_second
        )
        np.testing.assert_allclose(
            corrected_first, corrected_jitted, atol=1e-7, rtol=0.0
        )
        for name in metrics_first:
            np.testing.assert_array_equal(
                metrics_first[name], metrics_second[name]
            )
            np.testing.assert_allclose(
                metrics_first[name],
                metrics_jitted[name],
                atol=1e-7,
                rtol=0.0,
            )

    def test_jitted_correction_supports_different_batch_sizes(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        jitted_correct = jax.jit(
            lambda obs, actions: bridge.correct_actions(obs, actions)
        )

        full_actions, full_metrics = jitted_correct(
            jnp.asarray(observations), jnp.asarray(base_actions)
        )
        short_actions, short_metrics = jitted_correct(
            jnp.asarray(observations[:7]), jnp.asarray(base_actions[:7])
        )

        self.assertEqual(full_actions.shape, base_actions.shape)
        self.assertEqual(short_actions.shape, base_actions[:7].shape)
        self.assert_scalar_finite_float32_metrics(full_metrics)
        self.assert_scalar_finite_float32_metrics(short_metrics)

    def test_identity_normalization_matches_explicit_identity_metadata(self):
        batch = self.linear_transition_batch(batch_size=8)
        base_config = dict(
            hidden_dim=8,
            num_hidden_layers=2,
            learning_rate=1e-3,
        )
        implicit = DynamicsShiftBridge.create(
            seed=31,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(**base_config),
        )
        explicit = DynamicsShiftBridge.create(
            seed=31,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(
                **base_config,
                observation_mean=np.zeros(3, dtype=np.float32),
                observation_std=np.ones(3, dtype=np.float32),
                action_mean=np.zeros(2, dtype=np.float32),
                action_std=np.ones(2, dtype=np.float32),
                delta_mean=np.zeros(3, dtype=np.float32),
                delta_std=np.ones(3, dtype=np.float32),
            ),
        )

        implicit, implicit_metrics = implicit.update_offline(batch)
        explicit, explicit_metrics = explicit.update_offline(batch)

        self.assert_trees_allclose(
            implicit.offline_model.params,
            explicit.offline_model.params,
        )
        for name in implicit_metrics:
            np.testing.assert_allclose(
                implicit_metrics[name],
                explicit_metrics[name],
                atol=1e-7,
                rtol=0.0,
            )

    def test_zero_normalization_std_is_safe_and_metrics_are_explicit(self):
        batch = self.linear_transition_batch(batch_size=8)
        bridge = DynamicsShiftBridge.create(
            seed=37,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(
                hidden_dim=8,
                observation_std=np.zeros(3, dtype=np.float32),
                action_std=np.zeros(2, dtype=np.float32),
                delta_std=np.zeros(3, dtype=np.float32),
                normalization_epsilon=0.25,
            ),
        )

        bridge, update_metrics = bridge.update_offline(batch)
        _, correction_metrics = bridge.correct_actions(
            batch["observations"], batch["actions"]
        )

        np.testing.assert_array_equal(
            bridge.observation_scale,
            np.full(3, 0.25, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            bridge.action_scale,
            np.full(2, 0.25, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            bridge.delta_scale,
            np.full(3, 0.25, dtype=np.float32),
        )
        self.assertIn("normalized_prediction_mse", update_metrics)
        self.assertIn("raw_prediction_mse", update_metrics)
        self.assertIn(
            "post_match_mse_normalized", correction_metrics
        )
        self.assertIn("post_match_mse_raw", correction_metrics)
        for metrics in (update_metrics, correction_metrics):
            for value in metrics.values():
                self.assertTrue(np.all(np.isfinite(np.asarray(value))))

    def test_nonidentity_delta_normalization_separates_raw_and_normalized_mse(self):
        batch = self.linear_transition_batch(batch_size=8)
        bridge = DynamicsShiftBridge.create(
            seed=53,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(
                hidden_dim=8,
                delta_std=np.full(3, 2.0, dtype=np.float32),
            ),
        )

        _, metrics = bridge.update_offline(batch)

        self.assertAlmostEqual(
            float(metrics["raw_prediction_mse"]),
            4.0 * float(metrics["normalized_prediction_mse"]),
            places=6,
        )

    def test_evaluation_api_is_pure_deterministic_and_jittable(self):
        batch = self.linear_transition_batch(batch_size=8)
        bridge = DynamicsShiftBridge.create(
            seed=67,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
            config=DynamicsShiftBridgeConfig(hidden_dim=8),
        )
        bridge, _ = bridge.update_offline(batch)
        bridge = bridge.synchronize_online_from_offline()
        state_before = flax.serialization.to_state_dict(bridge)
        expected_keys = {
            "normalized_mse",
            "raw_mse",
            "prediction_abs_mean",
            "target_abs_mean",
        }

        offline_first = bridge.evaluate_offline(batch)
        offline_second = bridge.evaluate_offline(batch)
        online_metrics = bridge.evaluate_online(batch)
        jitted_evaluate = jax.jit(
            lambda transitions: bridge.evaluate_offline(transitions)
        )
        jitted_metrics = jitted_evaluate(
            {name: jnp.asarray(value) for name, value in batch.items()}
        )

        self.assertEqual(set(offline_first), expected_keys)
        self.assertEqual(set(online_metrics), expected_keys)
        self.assert_scalar_finite_float32_metrics(offline_first)
        self.assert_scalar_finite_float32_metrics(online_metrics)
        for name in expected_keys:
            np.testing.assert_array_equal(
                offline_first[name], offline_second[name]
            )
            np.testing.assert_allclose(
                offline_first[name],
                jitted_metrics[name],
                atol=1e-7,
                rtol=0.0,
            )
        self.assert_trees_allclose(
            flax.serialization.to_state_dict(bridge),
            state_before,
        )

    def test_invalid_configuration_is_rejected(self):
        batch = self.linear_transition_batch(batch_size=4)
        invalid_scalars = {
            "hidden_dim": 0,
            "num_hidden_layers": 0,
            "learning_rate": 0.0,
            "learning_rate_nan": ("learning_rate", np.nan),
            "clip_grad_norm": np.inf,
            "correction_steps": -1,
            "correction_step_size": -0.1,
            "dynamics_match_weight": np.nan,
            "action_l2_weight": -0.1,
            "max_residual": -0.1,
            "normalization_epsilon": 0.0,
        }
        for case_name, value in invalid_scalars.items():
            if isinstance(value, tuple):
                field, value = value
            else:
                field = case_name
            with self.subTest(case=case_name):
                with self.assertRaises(ValueError):
                    DynamicsShiftBridge.create(
                        seed=71,
                        example_observations=batch["observations"],
                        example_actions=batch["actions"],
                        config=DynamicsShiftBridgeConfig(
                            **{field: value}
                        ),
                    )

        invalid_metadata = (
            {"action_low": [-np.inf, -1.0]},
            {"action_high": [1.0, np.nan]},
            {"max_residual": [0.1, np.inf]},
            {"action_low": [-0.2, -1.0], "action_high": [-0.2, 0.4]},
            {"observation_mean": np.zeros(2, dtype=np.float32)},
            {"action_std": np.ones(3, dtype=np.float32)},
            {"delta_mean": np.zeros(2, dtype=np.float32)},
            {"delta_std": np.full(3, -1.0, dtype=np.float32)},
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    DynamicsShiftBridge.create(
                        seed=73,
                        example_observations=batch["observations"],
                        example_actions=batch["actions"],
                        config=DynamicsShiftBridgeConfig(**metadata),
                    )

    def test_invalid_transition_batches_are_rejected(self):
        batch = self.linear_transition_batch(batch_size=4)
        bridge = DynamicsShiftBridge.create(
            seed=79,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
        )
        invalid_batches = (
            {
                "actions": batch["actions"],
                "next_observations": batch["next_observations"],
            },
            {**batch, "actions": batch["actions"][:3]},
            {**batch, "actions": batch["actions"][:, :1]},
            {**batch, "observations": batch["observations"][:, :2]},
            {name: value[:0] for name, value in batch.items()},
            {**batch, "actions": batch["actions"].astype(object)},
            {**batch, "observations": batch["observations"].astype(bool)},
            {
                **batch,
                "next_observations": np.full_like(
                    batch["next_observations"], np.nan
                ),
            },
        )
        for index, invalid_batch in enumerate(invalid_batches):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    bridge.update_offline(invalid_batch)
                with self.assertRaises(ValueError):
                    bridge.update_online(invalid_batch)
                with self.assertRaises(ValueError):
                    bridge.evaluate_offline(invalid_batch)

    def test_prediction_and_correction_shape_contract_rejects_broadcasting(self):
        batch = self.linear_transition_batch(batch_size=4)
        bridge = DynamicsShiftBridge.create(
            seed=83,
            example_observations=batch["observations"],
            example_actions=batch["actions"],
        )
        invalid_pairs = (
            (batch["observations"][0], batch["actions"]),
            (batch["observations"], batch["actions"][0]),
            (batch["observations"][:0], batch["actions"][:0]),
            (batch["observations"], batch["actions"][:, :1]),
            (batch["observations"][:, :2], batch["actions"]),
        )
        for index, (observations, actions) in enumerate(invalid_pairs):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    bridge.predict_offline(observations, actions)
                with self.assertRaises(ValueError):
                    bridge.correct_actions(observations, actions)

    def test_serialization_round_trip_preserves_predictions_and_correction(self):
        bridge, observations, base_actions, _, _ = (
            self.trained_linear_result(0.7)
        )
        state_dict = flax.serialization.to_state_dict(bridge)
        restored = flax.serialization.from_state_dict(
            bridge, state_dict
        )

        np.testing.assert_array_equal(
            restored.predict_offline(observations, base_actions),
            bridge.predict_offline(observations, base_actions),
        )
        np.testing.assert_array_equal(
            restored.predict_online(observations, base_actions),
            bridge.predict_online(observations, base_actions),
        )
        restored_actions, restored_metrics = restored.correct_actions(
            observations, base_actions
        )
        original_actions, original_metrics = bridge.correct_actions(
            observations, base_actions
        )
        np.testing.assert_array_equal(restored_actions, original_actions)
        for name in original_metrics:
            np.testing.assert_array_equal(
                restored_metrics[name], original_metrics[name]
            )


if __name__ == "__main__":
    unittest.main()
