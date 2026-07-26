import unittest

import flax
import jax
import numpy as np

import utils.transition_occupancy as transition_occupancy
from utils.datasets import Dataset, ReplayBuffer
from utils.recent_dynamics_buffer import RecentDynamicsBuffer
from utils.transition_occupancy import (
    TransitionOccupancyDetector,
    pack_transition_features,
    should_update_occupancy_detector,
)


class TransitionOccupancyTest(unittest.TestCase):
    def transition_batch(self, batch_size=8, offset=0.0):
        observations = np.arange(
            batch_size * 2, dtype=np.float32
        ).reshape(batch_size, 2)
        observations = observations / max(batch_size, 1) + offset
        actions = np.linspace(
            -1.0, 1.0, batch_size, dtype=np.float32
        )[:, None]
        return {
            "observations": observations,
            "actions": actions,
            "next_observations": observations + actions,
        }

    def full_transition_dataset(self, size=32):
        observations = np.arange(
            size * 2, dtype=np.float32
        ).reshape(size, 2)
        actions = np.linspace(
            -1.0, 1.0, size, dtype=np.float32
        )[:, None]
        return Dataset.create(
            observations=observations,
            actions=actions,
            rewards=np.linspace(0.0, 1.0, size, dtype=np.float32),
            terminals=np.zeros(size, dtype=np.float32),
            masks=np.ones(size, dtype=np.float32),
            next_observations=observations + actions,
        )

    def sampling_components(self):
        dataset = self.full_transition_dataset()
        example = {
            name: np.asarray(values[0]).copy()
            for name, values in dataset.items()
        }
        recent = RecentDynamicsBuffer.create(example, capacity=8)
        for index in range(8):
            recent.add_transition(
                {
                    name: np.asarray(values[index]).copy()
                    for name, values in dataset.items()
                }
            )
        detector = TransitionOccupancyDetector.create(
            seed=101,
            example_offline_transition={
                name: dataset[name][:2]
                for name in (
                    "observations",
                    "actions",
                    "next_observations",
                )
            },
            config={"hidden_dim": 8},
        )
        return detector, dataset, recent

    def assert_numpy_rng_state_equal(self, actual, expected):
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2:], expected[2:])

    def assert_batches_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_pack_transition_features_orders_and_flattens_fields(self):
        batch = {
            "observations": np.array(
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                dtype=np.float32,
            ),
            "actions": np.array(
                [[[9.0, 10.0]], [[11.0, 12.0]]],
                dtype=np.float32,
            ),
            "next_observations": np.array(
                [[[13.0, 14.0], [15.0, 16.0]], [[17.0, 18.0], [19.0, 20.0]]],
                dtype=np.float32,
            ),
        }

        features = pack_transition_features(batch)

        np.testing.assert_array_equal(
            features,
            np.array(
                [
                    [
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        9.0,
                        10.0,
                        13.0,
                        14.0,
                        15.0,
                        16.0,
                    ],
                    [
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        11.0,
                        12.0,
                        17.0,
                        18.0,
                        19.0,
                        20.0,
                    ],
                ],
                dtype=np.float32,
            ),
        )

    def test_pack_transition_features_rejects_missing_fields(self):
        batch = {
            "observations": np.zeros((2, 3), dtype=np.float32),
            "actions": np.zeros((2, 1), dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "next_observations"):
            pack_transition_features(batch)

    def test_pack_transition_features_rejects_batch_size_mismatch(self):
        batch = {
            "observations": np.zeros((2, 3), dtype=np.float32),
            "actions": np.zeros((3, 1), dtype=np.float32),
            "next_observations": np.zeros((2, 3), dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "batch size"):
            pack_transition_features(batch)

    def test_pack_transition_features_rejects_object_dtype(self):
        batch = {
            "observations": np.zeros((2, 3), dtype=np.float32),
            "actions": np.array([[object()], [object()]], dtype=object),
            "next_observations": np.zeros((2, 3), dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "object dtype"):
            pack_transition_features(batch)

    def test_pack_transition_features_rejects_non_numeric_or_non_finite(self):
        base = self.transition_batch(batch_size=2)
        invalid_actions = (
            np.array([["a"], ["b"]]),
            np.array([[np.nan], [0.0]], dtype=np.float32),
        )
        for actions in invalid_actions:
            with self.subTest(dtype=actions.dtype):
                batch = dict(base)
                batch["actions"] = actions
                with self.assertRaisesRegex(
                    ValueError, "numeric|finite"
                ):
                    pack_transition_features(batch)

    def test_pack_transition_features_rejects_empty_batch(self):
        batch = {
            "observations": np.zeros((0, 3), dtype=np.float32),
            "actions": np.zeros((0, 1), dtype=np.float32),
            "next_observations": np.zeros((0, 3), dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "empty"):
            pack_transition_features(batch)

    def test_pack_transition_features_explicitly_normalizes_float_dtype(self):
        batch = {
            "observations": np.array(
                [[1.0, 2.0], [3.0, 4.0]], dtype=np.float64
            ),
            "actions": np.array([[5.0], [6.0]], dtype=np.float32),
            "next_observations": np.array(
                [[7, 8], [9, 10]], dtype=np.int32
            ),
        }

        features = pack_transition_features(batch)

        self.assertEqual(features.dtype, np.dtype(np.float32))
        self.assertEqual(features.shape, (2, 5))
        np.testing.assert_array_equal(
            features,
            np.array(
                [[1, 2, 5, 7, 8], [3, 4, 6, 9, 10]],
                dtype=np.float32,
            ),
        )

    def test_unbatched_transition_is_not_misinterpreted_as_batch(self):
        unbatched = {
            "observations": np.array([1.0, 2.0], dtype=np.float32),
            "actions": np.array([3.0, 4.0], dtype=np.float32),
            "next_observations": np.array([5.0, 6.0], dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "batch.*feature"):
            pack_transition_features(unbatched)
        with self.assertRaisesRegex(ValueError, "batch.*feature"):
            TransitionOccupancyDetector.create(
                seed=1,
                example_offline_transition=unbatched,
            )

    def test_create_infers_flattened_feature_dimension_for_batched_updates(self):
        batch = {
            "observations": np.arange(
                24, dtype=np.float64
            ).reshape(4, 2, 3),
            "actions": np.arange(
                8, dtype=np.float32
            ).reshape(4, 2),
            "next_observations": np.arange(
                24, dtype=np.float32
            ).reshape(4, 2, 3),
        }
        detector = TransitionOccupancyDetector.create(
            seed=2,
            example_offline_transition=batch,
            config={"hidden_dim": 8},
        )

        self.assertEqual(
            detector.network.params["hidden_0"]["kernel"].shape,
            (14, 8),
        )
        updated, metrics = detector.update(batch, batch)
        self.assertEqual(updated.logits(batch).shape, (4,))
        self.assertTrue(np.isfinite(float(metrics["loss"])))

    def test_detector_starts_with_neutral_logits_and_probabilities(self):
        batch = self.transition_batch()
        detector = TransitionOccupancyDetector.create(
            seed=7,
            example_offline_transition=batch,
            config={
                "hidden_dim": 16,
                "num_hidden_layers": 2,
                "learning_rate": 3e-4,
                "clip_grad_norm": 10.0,
            },
        )

        logits = detector.logits(batch)
        probabilities = detector.online_probability(batch)

        self.assertEqual(logits.shape, (8,))
        np.testing.assert_allclose(logits, np.zeros(8), atol=1e-7)
        np.testing.assert_allclose(
            probabilities, np.full(8, 0.5), atol=1e-7
        )

    def test_update_uses_offline_zero_and_online_one_labels(self):
        batch = self.transition_batch()
        detector = TransitionOccupancyDetector.create(
            seed=3,
            example_offline_transition=batch,
            config={"hidden_dim": 16},
        )

        _, metrics = detector.update(batch, batch)

        self.assertAlmostEqual(float(metrics["loss"]), np.log(2.0), places=6)
        self.assertAlmostEqual(
            float(metrics["offline_loss"]), np.log(2.0), places=6
        )
        self.assertAlmostEqual(
            float(metrics["online_loss"]), np.log(2.0), places=6
        )
        self.assertEqual(float(metrics["offline_accuracy"]), 0.0)
        self.assertEqual(float(metrics["online_accuracy"]), 1.0)
        self.assertEqual(float(metrics["balanced_accuracy"]), 0.5)

    def test_update_rejects_unequal_class_batch_sizes(self):
        offline_batch = self.transition_batch(batch_size=8)
        online_batch = self.transition_batch(batch_size=7)
        detector = TransitionOccupancyDetector.create(
            seed=3,
            example_offline_transition=offline_batch,
            config={"hidden_dim": 16},
        )

        with self.assertRaisesRegex(ValueError, "same batch size"):
            detector.update(offline_batch, online_batch)

    def test_update_metrics_are_finite_and_follow_public_definitions(self):
        offline_batch = self.transition_batch()
        online_batch = self.transition_batch(offset=2.0)
        detector = TransitionOccupancyDetector.create(
            seed=5,
            example_offline_transition=offline_batch,
            config={"hidden_dim": 16},
        )

        detector, metrics = detector.update(offline_batch, online_batch)

        expected_keys = {
            "loss",
            "offline_loss",
            "online_loss",
            "offline_accuracy",
            "online_accuracy",
            "balanced_accuracy",
            "offline_logit_mean",
            "online_logit_mean",
            "offline_probability_mean",
            "online_probability_mean",
            "logit_gap",
        }
        self.assertEqual(set(metrics), expected_keys)
        for name, value in metrics.items():
            with self.subTest(metric=name):
                self.assertTrue(np.isfinite(float(value)))
        self.assertAlmostEqual(
            float(metrics["balanced_accuracy"]),
            0.5
            * (
                float(metrics["offline_accuracy"])
                + float(metrics["online_accuracy"])
            ),
            places=7,
        )
        self.assertAlmostEqual(
            float(metrics["logit_gap"]),
            float(metrics["online_logit_mean"])
            - float(metrics["offline_logit_mean"]),
            places=7,
        )
        np.testing.assert_array_equal(
            detector.log_density_ratio_proxy(online_batch),
            detector.logits(online_batch),
        )

    def test_update_changes_parameters_and_step_but_not_sampling_rng(self):
        offline_batch = self.transition_batch()
        online_batch = self.transition_batch(offset=3.0)
        detector = TransitionOccupancyDetector.create(
            seed=11,
            example_offline_transition=offline_batch,
            config={"hidden_dim": 16},
        )
        before_params = jax.tree_util.tree_leaves(detector.network.params)
        before_step = int(detector.network.step)
        before_rng = np.asarray(detector.rng).copy()

        updated, _ = detector.update(offline_batch, online_batch)

        after_params = jax.tree_util.tree_leaves(updated.network.params)
        self.assertTrue(
            any(
                not np.array_equal(before, after)
                for before, after in zip(before_params, after_params)
            )
        )
        self.assertEqual(int(updated.network.step), before_step + 1)
        np.testing.assert_array_equal(updated.rng, before_rng)

    def test_detector_sampling_does_not_modify_global_numpy_rng(self):
        detector, dataset, recent = self.sampling_components()
        np.random.seed(2027)
        state_before = np.random.get_state()

        detector, offline_batch, online_batch = (
            transition_occupancy.sample_occupancy_transition_batches(
                detector,
                offline_dataset=dataset,
                recent_buffer=recent,
                batch_size=6,
            )
        )

        self.assert_numpy_rng_state_equal(
            np.random.get_state(), state_before
        )
        self.assertEqual(
            set(offline_batch),
            {
                "observations",
                "actions",
                "next_observations",
            },
        )
        self.assertEqual(set(online_batch), set(offline_batch))
        self.assertEqual(offline_batch["observations"].shape[0], 6)
        self.assertEqual(online_batch["observations"].shape[0], 6)

    def test_detector_sampling_does_not_change_next_dataset_sequence(self):
        detector, dataset, recent = self.sampling_components()
        np.random.seed(3031)
        expected = dataset.sample_sequence(
            5, sequence_length=3, discount=0.99
        )

        np.random.seed(3031)
        for _ in range(3):
            detector, _, _ = (
                transition_occupancy.sample_occupancy_transition_batches(
                    detector,
                    offline_dataset=dataset,
                    recent_buffer=recent,
                    batch_size=7,
                )
            )
        actual = dataset.sample_sequence(
            5, sequence_length=3, discount=0.99
        )

        self.assert_batches_equal(actual, expected)

    def test_detector_sampling_does_not_change_next_replay_sequence(self):
        detector, dataset, recent = self.sampling_components()
        replay = ReplayBuffer.create_from_initial_dataset(
            dict(dataset), size=dataset.size + 8
        )
        np.random.seed(4049)
        expected = replay.sample_sequence(
            5, sequence_length=3, discount=0.99
        )

        np.random.seed(4049)
        for _ in range(3):
            detector, _, _ = (
                transition_occupancy.sample_occupancy_transition_batches(
                    detector,
                    offline_dataset=dataset,
                    recent_buffer=recent,
                    batch_size=7,
                )
            )
        actual = replay.sample_sequence(
            5, sequence_length=3, discount=0.99
        )

        self.assert_batches_equal(actual, expected)

    def test_detector_sampling_is_reproducible_from_same_state(self):
        detector, dataset, recent = self.sampling_components()

        first_detector, first_offline, first_online = (
            transition_occupancy.sample_occupancy_transition_batches(
                detector,
                offline_dataset=dataset,
                recent_buffer=recent,
                batch_size=16,
            )
        )
        second_detector, second_offline, second_online = (
            transition_occupancy.sample_occupancy_transition_batches(
                detector,
                offline_dataset=dataset,
                recent_buffer=recent,
                batch_size=16,
            )
        )

        self.assert_batches_equal(first_offline, second_offline)
        self.assert_batches_equal(first_online, second_online)
        np.testing.assert_array_equal(
            first_detector.rng, second_detector.rng
        )

    def test_detector_sampling_advances_only_detector_rng(self):
        detector, dataset, recent = self.sampling_components()
        rng_before = np.asarray(detector.rng).copy()

        sampled_detector, _, _ = (
            transition_occupancy.sample_occupancy_transition_batches(
                detector,
                offline_dataset=dataset,
                recent_buffer=recent,
                batch_size=4,
            )
        )

        self.assertFalse(
            np.array_equal(sampled_detector.rng, rng_before)
        )

    def test_zero_initialized_output_does_not_block_hidden_layer_training(self):
        offline_batch = self.transition_batch(batch_size=32)
        online_batch = self.transition_batch(batch_size=32, offset=3.0)
        detector = TransitionOccupancyDetector.create(
            seed=12,
            example_offline_transition=offline_batch,
            config={"hidden_dim": 16, "learning_rate": 1e-3},
        )
        before_hidden = jax.tree_util.tree_leaves(
            {
                name: params
                for name, params in detector.network.params.items()
                if name.startswith("hidden_")
            }
        )

        for _ in range(4):
            detector, _ = detector.update(
                offline_batch, online_batch
            )

        after_hidden = jax.tree_util.tree_leaves(
            {
                name: params
                for name, params in detector.network.params.items()
                if name.startswith("hidden_")
            }
        )
        self.assertTrue(
            any(
                not np.array_equal(before, after)
                for before, after in zip(before_hidden, after_hidden)
            )
        )

    def test_evaluate_is_finite_and_does_not_modify_state_or_rng(self):
        offline_batch = self.transition_batch(batch_size=16)
        online_batch = self.transition_batch(
            batch_size=16, offset=2.0
        )
        detector = TransitionOccupancyDetector.create(
            seed=13,
            example_offline_transition=offline_batch,
            config={"hidden_dim": 16, "learning_rate": 1e-3},
        )
        for _ in range(3):
            detector, _ = detector.update(
                offline_batch, online_batch
            )
        before = flax.serialization.to_state_dict(detector)

        metrics = detector.evaluate(offline_batch, online_batch)

        after = flax.serialization.to_state_dict(detector)
        for expected, actual in zip(
            jax.tree_util.tree_leaves(before),
            jax.tree_util.tree_leaves(after),
        ):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            set(metrics),
            {
                "loss",
                "offline_loss",
                "online_loss",
                "offline_accuracy",
                "online_accuracy",
                "balanced_accuracy",
                "offline_logit_mean",
                "online_logit_mean",
                "offline_probability_mean",
                "online_probability_mean",
                "logit_gap",
            },
        )
        for value in metrics.values():
            self.assertTrue(np.isfinite(float(value)))

    def test_identical_paired_transitions_hold_out_at_random_classifier(self):
        training_batch = self.transition_batch(batch_size=32)
        held_out_batch = self.transition_batch(
            batch_size=24, offset=7.0
        )
        detector = TransitionOccupancyDetector.create(
            seed=14,
            example_offline_transition=training_batch,
            config={
                "hidden_dim": 32,
                "learning_rate": 1e-3,
            },
        )

        for _ in range(25):
            detector, _ = detector.update(
                training_batch, training_batch
            )

        metrics = detector.evaluate(held_out_batch, held_out_batch)
        probabilities = np.asarray(
            detector.online_probability(held_out_batch)
        )
        np.testing.assert_allclose(
            probabilities, np.full(24, 0.5), atol=1e-2
        )
        self.assertAlmostEqual(float(metrics["loss"]), np.log(2.0), places=6)
        self.assertAlmostEqual(float(metrics["logit_gap"]), 0.0, places=7)

    def controlled_shift_batches(self, seed, batch_size):
        rng = np.random.default_rng(seed)
        observations = rng.normal(
            size=(batch_size, 2)
        ).astype(np.float32)
        actions = rng.uniform(
            -1.0, 1.0, size=(batch_size, 1)
        ).astype(np.float32)
        shared = {
            "observations": observations,
            "actions": actions,
        }
        return (
            {
                **shared,
                "next_observations": observations + actions,
            },
            {
                **shared,
                "next_observations": (
                    observations + 0.5 * actions + 1.5
                ),
            },
        )

    def test_controlled_next_state_shift_is_detected_on_held_out_batch(self):
        offline_batch, online_batch = self.controlled_shift_batches(
            seed=19, batch_size=128
        )
        held_out_offline, held_out_online = (
            self.controlled_shift_batches(seed=20, batch_size=128)
        )
        detector = TransitionOccupancyDetector.create(
            seed=23,
            example_offline_transition=offline_batch,
            config={
                "hidden_dim": 64,
                "learning_rate": 3e-3,
            },
        )

        for _ in range(200):
            detector, _ = detector.update(
                offline_batch, online_batch
            )

        metrics = detector.evaluate(
            held_out_offline, held_out_online
        )
        self.assertGreaterEqual(
            float(metrics["balanced_accuracy"]), 0.90
        )
        self.assertGreater(
            float(metrics["online_logit_mean"]),
            float(metrics["offline_logit_mean"]),
        )
        self.assertGreater(float(metrics["logit_gap"]), 0.0)

    def test_flax_state_round_trip_preserves_logits_and_continuation(self):
        offline_batch = self.transition_batch()
        online_batch = self.transition_batch(offset=2.0)
        config = {"hidden_dim": 16, "learning_rate": 1e-3}
        detector = TransitionOccupancyDetector.create(
            seed=29,
            example_offline_transition=offline_batch,
            config=config,
        )
        for _ in range(5):
            detector, _ = detector.update(offline_batch, online_batch)
        template = TransitionOccupancyDetector.create(
            seed=999,
            example_offline_transition=offline_batch,
            config=config,
        )

        restored = flax.serialization.from_state_dict(
            template, flax.serialization.to_state_dict(detector)
        )

        np.testing.assert_array_equal(
            restored.logits(online_batch),
            detector.logits(online_batch),
        )
        uninterrupted, uninterrupted_metrics = detector.update(
            offline_batch, online_batch
        )
        resumed, resumed_metrics = restored.update(
            offline_batch, online_batch
        )
        for expected, actual in zip(
            jax.tree_util.tree_leaves(uninterrupted),
            jax.tree_util.tree_leaves(resumed),
        ):
            np.testing.assert_array_equal(actual, expected)
        for name in uninterrupted_metrics:
            np.testing.assert_array_equal(
                resumed_metrics[name], uninterrupted_metrics[name]
            )

    def test_occupancy_update_schedule_requires_enabled_ready_interval(self):
        self.assertFalse(
            should_update_occupancy_detector(
                online_step=20,
                recent_size=20,
                enabled=False,
                start_size=10,
                update_interval=5,
            )
        )
        self.assertFalse(
            should_update_occupancy_detector(
                online_step=0,
                recent_size=10,
                enabled=True,
                start_size=10,
                update_interval=5,
            )
        )
        self.assertFalse(
            should_update_occupancy_detector(
                online_step=20,
                recent_size=9,
                enabled=True,
                start_size=10,
                update_interval=5,
            )
        )
        self.assertFalse(
            should_update_occupancy_detector(
                online_step=21,
                recent_size=20,
                enabled=True,
                start_size=10,
                update_interval=5,
            )
        )
        self.assertTrue(
            should_update_occupancy_detector(
                online_step=20,
                recent_size=10,
                enabled=True,
                start_size=10,
                update_interval=5,
            )
        )

    def test_occupancy_update_schedule_boundaries_and_resume_step(self):
        arguments = {
            "enabled": True,
            "start_size": 10,
            "update_interval": 20,
        }
        self.assertFalse(
            should_update_occupancy_detector(
                online_step=20, recent_size=9, **arguments
            )
        )
        self.assertTrue(
            should_update_occupancy_detector(
                online_step=20, recent_size=10, **arguments
            )
        )
        self.assertFalse(
            should_update_occupancy_detector(
                online_step=19, recent_size=10, **arguments
            )
        )
        saved_step = 20
        self.assertTrue(
            should_update_occupancy_detector(
                online_step=saved_step,
                recent_size=10,
                **arguments,
            )
        )
        self.assertFalse(
            should_update_occupancy_detector(
                online_step=saved_step + 1,
                recent_size=10,
                **arguments,
            )
        )

    def test_average_occupancy_metrics_means_each_update(self):
        metrics = transition_occupancy.average_occupancy_metrics(
            [
                {"loss": np.float32(1.0), "logit_gap": np.float32(2.0)},
                {"loss": np.float32(3.0), "logit_gap": np.float32(6.0)},
            ]
        )

        self.assertEqual(float(metrics["loss"]), 2.0)
        self.assertEqual(float(metrics["logit_gap"]), 4.0)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            transition_occupancy.average_occupancy_metrics([])
        with self.assertRaisesRegex(ValueError, "same metric"):
            transition_occupancy.average_occupancy_metrics(
                [{"loss": 1.0}, {"logit_gap": 2.0}]
            )

    def test_occupancy_update_schedule_rejects_invalid_parameters(self):
        valid = {
            "online_step": 10,
            "recent_size": 10,
            "enabled": True,
            "start_size": 5,
            "update_interval": 5,
        }
        invalid_cases = (
            ("online_step", -1),
            ("recent_size", -1),
            ("enabled", 1),
            ("start_size", 0),
            ("update_interval", 0),
        )
        for name, value in invalid_cases:
            with self.subTest(name=name, value=value):
                arguments = dict(valid)
                arguments[name] = value
                with self.assertRaisesRegex(ValueError, name):
                    should_update_occupancy_detector(**arguments)

    def test_create_rejects_invalid_detector_config(self):
        batch = self.transition_batch()
        invalid_configs = (
            {"hidden_dim": 0},
            {"num_hidden_layers": True},
            {"learning_rate": np.nan},
            {"clip_grad_norm": 0.0},
            {"unknown": 1},
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "config|hidden|layer|learning|clip"):
                    TransitionOccupancyDetector.create(
                        seed=1,
                        example_offline_transition=batch,
                        config=config,
                    )


if __name__ == "__main__":
    unittest.main()
