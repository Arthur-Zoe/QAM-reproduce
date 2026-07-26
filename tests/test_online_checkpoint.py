import os
import pickle
import random
import tempfile
import unittest
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np

from utils.datasets import Dataset, ReplayBuffer
from utils.online_checkpoint import (
    CHECKPOINT_FILENAME,
    FORMAT_VERSION,
    OnlineCheckpointError,
    load_online_checkpoint,
    online_start_step,
    read_progress,
    restore_recent_dynamics_buffer,
    restore_replay_buffer,
    restore_rng_states,
    save_online_checkpoint,
    should_save_online_checkpoint,
    validate_online_checkpoint_restore,
)
from utils.recent_dynamics_buffer import RecentDynamicsBuffer
from utils.transition_occupancy import (
    TransitionOccupancyDetector,
    sample_occupancy_transition_batches,
)


class DummyAgent(flax.struct.PyTreeNode):
    params: Any
    opt_state: Any


class OnlineCheckpointTest(unittest.TestCase):
    env_name = "dummy-singletask-task1-v0"
    horizon_length = 3
    action_dim = 1
    offline_steps = 10

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.save_dir = self.temporary_directory.name
        self.agent = DummyAgent(
            params={"weight": jnp.array([1.0, 2.0], dtype=jnp.float32)},
            opt_state={"momentum": jnp.array([0.25, 0.5], dtype=jnp.float32)},
        )
        self.online_rng = jax.random.PRNGKey(17)
        np.random.seed(123)
        random.seed(456)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def initial_dataset(self):
        return {
            "observations": np.array([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
            "actions": np.array([[0.1], [0.2]], dtype=np.float32),
            "rewards": np.array([1.0, 2.0], dtype=np.float32),
            "terminals": np.array([0.0, 1.0], dtype=np.float32),
            "masks": np.array([1.0, 0.0], dtype=np.float32),
            "next_observations": np.array(
                [[12.0, 13.0], [22.0, 23.0]], dtype=np.float32
            ),
        }

    def transition(self, value):
        return {
            "observations": np.array([value, value + 0.1], dtype=np.float32),
            "actions": np.array([value / 10.0], dtype=np.float32),
            "rewards": np.float32(value),
            "terminals": np.float32(0.0),
            "masks": np.float32(1.0),
            "next_observations": np.array(
                [value + 1.0, value + 1.1], dtype=np.float32
            ),
        }

    def make_non_balanced_buffer(self):
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        replay_buffer.add_transition(self.transition(30.0))
        replay_buffer.add_transition(self.transition(40.0))
        return replay_buffer

    def make_balanced_buffer(self):
        replay_buffer = ReplayBuffer.create(self.transition(0.0), size=5)
        replay_buffer.add_transition(self.transition(1.0))
        replay_buffer.add_transition(self.transition(2.0))
        replay_buffer.add_transition(self.transition(3.0))
        return replay_buffer

    def make_recent_buffer(self, capacity, values):
        buffer = RecentDynamicsBuffer.create(
            self.transition(0.0), capacity=capacity
        )
        for value in values:
            buffer.add_transition(self.transition(float(value)))
        return buffer

    def occupancy_config(self, **overrides):
        config = {
            "hidden_dim": 16,
            "num_hidden_layers": 2,
            "learning_rate": 1e-3,
            "clip_grad_norm": 10.0,
            "batch_size": 2,
            "start_size": 2,
            "update_interval": 5,
            "updates_per_interval": 3,
        }
        config.update(overrides)
        return config

    def make_occupancy_detector(self, seed=71, updates=0):
        batch = {
            field: self.initial_dataset()[field]
            for field in (
                "observations",
                "actions",
                "next_observations",
            )
        }
        detector = TransitionOccupancyDetector.create(
            seed=seed,
            example_offline_transition=batch,
            config={
                key: self.occupancy_config()[key]
                for key in (
                    "hidden_dim",
                    "num_hidden_layers",
                    "learning_rate",
                    "clip_grad_norm",
                )
            },
        )
        online_batch = dict(batch)
        online_batch["next_observations"] = (
            online_batch["next_observations"] + 2.0
        )
        for _ in range(updates):
            detector, _ = detector.update(batch, online_batch)
        return detector

    def run_synthetic_detector_steps(
        self,
        detector,
        replay_buffer,
        recent_buffer,
        offline_dataset,
        online_rng,
        start_step,
        end_step,
    ):
        sampling_trace = []
        reference_observations = np.asarray(
            offline_dataset["observations"]
        )

        def offline_indices(batch):
            indices = []
            for observation in np.asarray(batch["observations"]):
                matches = np.flatnonzero(
                    np.all(
                        reference_observations == observation,
                        axis=tuple(
                            range(1, reference_observations.ndim)
                        ),
                    )
                )
                self.assertEqual(matches.size, 1)
                indices.append(matches[0])
            return np.asarray(indices, dtype=np.int64)

        for step in range(start_step, end_step + 1):
            online_rng, _ = jax.random.split(online_rng)
            transition = self.transition(float(step))
            replay_buffer.add_transition(transition)
            recent_buffer.add_transition(transition)
            detector, offline_sample, online_sample = (
                sample_occupancy_transition_batches(
                    detector,
                    offline_dataset,
                    recent_buffer,
                    batch_size=2,
                )
            )
            detector, _ = detector.update(
                offline_sample,
                online_sample,
            )
            detector, evaluation_offline, evaluation_online = (
                sample_occupancy_transition_batches(
                    detector,
                    offline_dataset,
                    recent_buffer,
                    batch_size=2,
                )
            )
            detector.evaluate(
                evaluation_offline,
                evaluation_online,
            )
            sampling_trace.append(
                {
                    "train_offline_indices": offline_indices(
                        offline_sample
                    ),
                    **{
                        f"train_offline_{field}": np.asarray(
                            offline_sample[field]
                        ).copy()
                        for field in (
                            "observations",
                            "actions",
                            "next_observations",
                        )
                    },
                    **{
                        f"train_online_{field}": np.asarray(
                            online_sample[field]
                        ).copy()
                        for field in (
                            "observations",
                            "actions",
                            "next_observations",
                        )
                    },
                    "eval_offline_indices": offline_indices(
                        evaluation_offline
                    ),
                    **{
                        f"eval_offline_{field}": np.asarray(
                            evaluation_offline[field]
                        ).copy()
                        for field in (
                            "observations",
                            "actions",
                            "next_observations",
                        )
                    },
                    **{
                        f"eval_online_{field}": np.asarray(
                            evaluation_online[field]
                        ).copy()
                        for field in (
                            "observations",
                            "actions",
                            "next_observations",
                        )
                    },
                }
            )
            random.random()
        return detector, online_rng, sampling_trace

    def save(
        self,
        replay_buffer,
        online_step,
        balanced_sampling,
        recent_dynamics_buffer=None,
        recent_dynamics_capacity=0,
        occupancy_detector=None,
        occupancy_detector_config=None,
        online_rng=None,
    ):
        initial_replay_size = 0 if balanced_sampling else 2
        return save_online_checkpoint(
            save_dir=self.save_dir,
            agent=self.agent,
            replay_buffer=replay_buffer,
            online_rng=(
                self.online_rng if online_rng is None else online_rng
            ),
            online_step=online_step,
            offline_steps=self.offline_steps,
            balanced_sampling=balanced_sampling,
            initial_replay_size=initial_replay_size,
            action_dim=self.action_dim,
            horizon_length=self.horizon_length,
            env_name=self.env_name,
            done=True,
            action_queue=[],
            recent_dynamics_buffer=recent_dynamics_buffer,
            recent_dynamics_capacity=recent_dynamics_capacity,
            occupancy_detector=occupancy_detector,
            occupancy_detector_config=occupancy_detector_config,
        )

    def load(
        self,
        agent,
        balanced_sampling,
        initial_replay_size,
        recent_dynamics_capacity=0,
        occupancy_detector=None,
        occupancy_detector_config=None,
    ):
        try:
            checkpoint = self.read_raw_checkpoint()
            max_size = checkpoint["replay_buffer"]["max_size"]
        except Exception:
            # Let the production loader report missing/corrupt checkpoint
            # errors; these defaults only provide compatible empty templates.
            max_size = 5 if balanced_sampling else 6
        if balanced_sampling:
            replay_buffer = ReplayBuffer.create(
                self.transition(0.0), size=max_size
            )
        else:
            replay_buffer = ReplayBuffer.create_from_initial_dataset(
                self.initial_dataset(), size=max_size
            )
        recent_dynamics_buffer = (
            self.make_recent_buffer(recent_dynamics_capacity, [])
            if recent_dynamics_capacity > 0
            else None
        )
        (
            restored_agent,
            restored_detector,
            _,
            checkpoint,
        ) = load_online_checkpoint(
            self.save_dir,
            agent,
            replay_buffer=replay_buffer,
            recent_dynamics_buffer=recent_dynamics_buffer,
            expected_env_name=self.env_name,
            expected_horizon_length=self.horizon_length,
            expected_balanced_sampling=balanced_sampling,
            expected_initial_replay_size=initial_replay_size,
            expected_action_dim=self.action_dim,
            expected_offline_steps=self.offline_steps,
            expected_recent_dynamics_capacity=recent_dynamics_capacity,
            occupancy_detector=occupancy_detector,
            expected_occupancy_detector_config=(
                occupancy_detector_config
            ),
        )
        return restored_agent, restored_detector, checkpoint

    def read_raw_checkpoint(self):
        with open(os.path.join(self.save_dir, CHECKPOINT_FILENAME), "rb") as file:
            return pickle.load(file)

    def write_raw_checkpoint(self, checkpoint):
        with open(
            os.path.join(self.save_dir, CHECKPOINT_FILENAME), "wb"
        ) as file:
            pickle.dump(checkpoint, file)

    def assert_numpy_rng_state_equal(self, actual, expected):
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2:], expected[2:])

    def test_non_balanced_replay_buffer_round_trip(self):
        source = self.make_non_balanced_buffer()
        self.save(source, online_step=2, balanced_sampling=False)
        target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        _, _, checkpoint = self.load(self.agent, False, 2)

        restore_replay_buffer(target, checkpoint, False, 2)

        for key in source:
            np.testing.assert_array_equal(target[key][:4], source[key][:4])

    def test_balanced_replay_buffer_round_trip(self):
        source = self.make_balanced_buffer()
        self.save(source, online_step=3, balanced_sampling=True)
        target = ReplayBuffer.create(self.transition(0.0), size=5)
        _, _, checkpoint = self.load(self.agent, True, 0)

        restore_replay_buffer(target, checkpoint, True, 0)

        for key in source:
            np.testing.assert_array_equal(target[key][:3], source[key][:3])

    def test_agent_state_round_trip(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        template = DummyAgent(
            params={"weight": jnp.zeros(2, dtype=jnp.float32)},
            opt_state={"momentum": jnp.zeros(2, dtype=jnp.float32)},
        )

        restored_agent, _, _ = self.load(template, False, 2)

        np.testing.assert_array_equal(
            restored_agent.params["weight"], self.agent.params["weight"]
        )
        np.testing.assert_array_equal(
            restored_agent.opt_state["momentum"], self.agent.opt_state["momentum"]
        )

    def test_pointer_size_and_max_size_round_trip(self):
        source = self.make_non_balanced_buffer()
        self.save(source, 2, False)
        target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        _, _, checkpoint = self.load(self.agent, False, 2)

        restore_replay_buffer(target, checkpoint, False, 2)

        self.assertEqual(target.pointer, source.pointer)
        self.assertEqual(target.size, source.size)
        self.assertEqual(target.max_size, source.max_size)

    def test_online_data_shape_and_dtype_round_trip(self):
        source = self.make_balanced_buffer()
        self.save(source, 3, True)
        target = ReplayBuffer.create(self.transition(0.0), size=5)
        _, _, checkpoint = self.load(self.agent, True, 0)
        restore_replay_buffer(target, checkpoint, True, 0)

        for key in source:
            self.assertEqual(target[key].shape, source[key].shape)
            self.assertEqual(target[key].dtype, source[key].dtype)
            np.testing.assert_array_equal(target[key][:3], source[key][:3])

    def test_checkpoint_does_not_repeat_offline_dataset_region(self):
        self.save(self.make_non_balanced_buffer(), 2, False)

        replay_state = self.read_raw_checkpoint()["replay_buffer"]

        self.assertEqual(replay_state["data_start"], 2)
        self.assertEqual(replay_state["data_count"], 2)
        for saved in replay_state["online_data"].values():
            self.assertEqual(saved.shape[0], 2)
        np.testing.assert_array_equal(
            replay_state["online_data"]["observations"][:, 0],
            np.array([30.0, 40.0], dtype=np.float32),
        )

    def test_jax_online_rng_round_trip(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        _, _, checkpoint = self.load(self.agent, False, 2)

        restored_rng = restore_rng_states(checkpoint)

        np.testing.assert_array_equal(restored_rng, self.online_rng)

    def test_numpy_rng_state_round_trip(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        expected = np.random.random(4)
        np.random.seed(999)
        _, _, checkpoint = self.load(self.agent, False, 2)

        restore_rng_states(checkpoint)

        np.testing.assert_array_equal(np.random.random(4), expected)

    def test_python_random_state_round_trip(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        expected = [random.random() for _ in range(4)]
        random.seed(999)
        _, _, checkpoint = self.load(self.agent, False, 2)

        restore_rng_states(checkpoint)

        self.assertEqual([random.random() for _ in range(4)], expected)

    def test_invalid_rng_restore_does_not_modify_global_states(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        _, _, checkpoint = self.load(self.agent, False, 2)
        checkpoint["python_rng_state"] = ("invalid",)
        numpy_before = np.random.get_state()
        python_before = random.getstate()

        with self.assertRaisesRegex(OnlineCheckpointError, "random state"):
            restore_rng_states(checkpoint)

        self.assert_numpy_rng_state_equal(
            np.random.get_state(), numpy_before
        )
        self.assertEqual(random.getstate(), python_before)

    def test_invalid_jax_rng_layout_is_rejected(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        _, _, checkpoint = self.load(self.agent, False, 2)
        checkpoint["online_rng"] = np.zeros(3, dtype=np.uint32)

        with self.assertRaisesRegex(
            OnlineCheckpointError, r"online_rng.*shape \(2,\).*uint32"
        ):
            restore_rng_states(checkpoint)

    def test_incompatible_env_name_raises(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        with self.assertRaisesRegex(OnlineCheckpointError, "env_name"):
            load_online_checkpoint(
                self.save_dir,
                self.agent,
                replay_buffer=ReplayBuffer.create_from_initial_dataset(
                    self.initial_dataset(), size=6
                ),
                recent_dynamics_buffer=None,
                expected_env_name="different-env",
                expected_horizon_length=self.horizon_length,
                expected_balanced_sampling=False,
                expected_initial_replay_size=2,
                expected_action_dim=self.action_dim,
                expected_offline_steps=self.offline_steps,
            )

    def test_incompatible_horizon_length_raises(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        with self.assertRaisesRegex(OnlineCheckpointError, "horizon_length"):
            load_online_checkpoint(
                self.save_dir,
                self.agent,
                replay_buffer=ReplayBuffer.create_from_initial_dataset(
                    self.initial_dataset(), size=6
                ),
                recent_dynamics_buffer=None,
                expected_env_name=self.env_name,
                expected_horizon_length=99,
                expected_balanced_sampling=False,
                expected_initial_replay_size=2,
                expected_action_dim=self.action_dim,
                expected_offline_steps=self.offline_steps,
            )

    def test_incompatible_balanced_sampling_raises(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        with self.assertRaisesRegex(OnlineCheckpointError, "balanced_sampling"):
            self.load(self.agent, True, 2)

    def test_replay_buffer_key_shape_and_dtype_mismatch_raise(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        _, _, checkpoint = self.load(self.agent, False, 2)

        with self.subTest("key"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            modified["replay_buffer"]["online_data"].pop("actions")
            with self.assertRaisesRegex(OnlineCheckpointError, "key mismatch"):
                restore_replay_buffer(target, modified, False, 2)

        with self.subTest("shape"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            modified["replay_buffer"]["online_data"]["actions"] = np.zeros(
                (2, 2), dtype=np.float32
            )
            with self.assertRaisesRegex(OnlineCheckpointError, "shape mismatch"):
                restore_replay_buffer(target, modified, False, 2)

        with self.subTest("dtype"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            modified["replay_buffer"]["online_data"]["actions"] = modified[
                "replay_buffer"
            ]["online_data"]["actions"].astype(np.float64)
            with self.assertRaisesRegex(OnlineCheckpointError, "dtype mismatch"):
                restore_replay_buffer(target, modified, False, 2)

    def test_corrupt_checkpoint_raises_clear_error(self):
        with open(os.path.join(self.save_dir, CHECKPOINT_FILENAME), "wb") as file:
            file.write(b"not a pickle")

        with self.assertRaisesRegex(OnlineCheckpointError, "Failed to load"):
            self.load(self.agent, False, 2)

    def test_atomic_save_leaves_no_temporary_files(self):
        checkpoint_path = self.save(
            self.make_non_balanced_buffer(), 2, False
        )

        self.assertTrue(os.path.isfile(checkpoint_path))
        self.assertEqual(
            sorted(os.listdir(self.save_dir)),
            [CHECKPOINT_FILENAME, "progress.tk"],
        )

    def test_online_progress_continues_from_saved_step_plus_one(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        stage, saved_step = read_progress(self.save_dir)
        _, _, checkpoint = self.load(self.agent, False, 2)

        self.assertEqual(stage, "online")
        self.assertEqual(saved_step, 2)
        self.assertEqual(online_start_step(checkpoint), 3)

    def test_progress_step_mismatch_fails_before_restore_mutation(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        replay_target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        replay_before = {
            key: value.copy() for key, value in replay_target.items()
        }
        pointer_before = replay_target.pointer
        size_before = replay_target.size
        numpy_before = np.random.get_state()
        python_before = random.getstate()

        with self.assertRaisesRegex(
            OnlineCheckpointError, "progress.*online step"
        ):
            load_online_checkpoint(
                self.save_dir,
                self.agent,
                replay_buffer=replay_target,
                recent_dynamics_buffer=None,
                expected_env_name=self.env_name,
                expected_horizon_length=self.horizon_length,
                expected_balanced_sampling=False,
                expected_initial_replay_size=2,
                expected_action_dim=self.action_dim,
                expected_offline_steps=self.offline_steps,
                expected_recent_dynamics_capacity=0,
                expected_online_step=3,
            )

        self.assertEqual(replay_target.pointer, pointer_before)
        self.assertEqual(replay_target.size, size_before)
        for key, value in replay_target.items():
            np.testing.assert_array_equal(value, replay_before[key])
        self.assert_numpy_rng_state_equal(
            np.random.get_state(), numpy_before
        )
        self.assertEqual(random.getstate(), python_before)

    def test_unknown_progress_stage_raises(self):
        with open(os.path.join(self.save_dir, "progress.tk"), "w") as file:
            file.write("mystery,3")

        with self.assertRaisesRegex(OnlineCheckpointError, "unknown stage"):
            read_progress(self.save_dir)

    def test_online_progress_requires_checkpoint_file(self):
        with open(os.path.join(self.save_dir, "progress.tk"), "w") as file:
            file.write("online,3")

        with self.assertRaisesRegex(OnlineCheckpointError, "missing"):
            self.load(self.agent, False, 2)

    def test_online_save_interval_zero_disables_saving(self):
        self.assertFalse(
            should_save_online_checkpoint(
                online_step=100,
                online_save_interval=0,
                last_saved_online_step=0,
                done=True,
                action_queue=[],
            )
        )
        self.assertFalse(os.path.exists(os.path.join(self.save_dir, CHECKPOINT_FILENAME)))

    def test_checkpoint_requires_done_and_empty_action_queue(self):
        self.assertFalse(
            should_save_online_checkpoint(1000, 1000, 0, False, [])
        )
        self.assertFalse(
            should_save_online_checkpoint(1000, 1000, 0, True, [np.zeros(1)])
        )
        self.assertTrue(
            should_save_online_checkpoint(1137, 1000, 0, True, [])
        )

        replay_buffer = self.make_non_balanced_buffer()
        kwargs = dict(
            save_dir=self.save_dir,
            agent=self.agent,
            replay_buffer=replay_buffer,
            online_rng=self.online_rng,
            online_step=2,
            offline_steps=self.offline_steps,
            balanced_sampling=False,
            initial_replay_size=2,
            action_dim=self.action_dim,
            horizon_length=self.horizon_length,
            env_name=self.env_name,
        )
        with self.assertRaisesRegex(OnlineCheckpointError, "done=True"):
            save_online_checkpoint(**kwargs, done=False, action_queue=[])
        with self.assertRaisesRegex(OnlineCheckpointError, "empty action_queue"):
            save_online_checkpoint(
                **kwargs, done=True, action_queue=[np.zeros(1)]
            )

    def test_missing_field_and_unsupported_version_raise(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        checkpoint_path = os.path.join(self.save_dir, CHECKPOINT_FILENAME)

        checkpoint = self.read_raw_checkpoint()
        checkpoint.pop("agent")
        with open(checkpoint_path, "wb") as file:
            pickle.dump(checkpoint, file)
        with self.assertRaisesRegex(OnlineCheckpointError, "missing required fields"):
            self.load(self.agent, False, 2)

        checkpoint["agent"] = flax.serialization.to_state_dict(self.agent)
        checkpoint["format_version"] = FORMAT_VERSION + 1
        with open(checkpoint_path, "wb") as file:
            pickle.dump(checkpoint, file)
        with self.assertRaisesRegex(OnlineCheckpointError, "Unsupported"):
            self.load(self.agent, False, 2)

    def test_pointer_and_size_out_of_bounds_raise(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        _, _, checkpoint = self.load(self.agent, False, 2)

        with self.subTest("pointer"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            modified["replay_buffer"]["pointer"] = 6
            with self.assertRaisesRegex(OnlineCheckpointError, "pointer"):
                restore_replay_buffer(target, modified, False, 2)

        with self.subTest("size"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            modified["replay_buffer"]["size"] = 7
            with self.assertRaisesRegex(OnlineCheckpointError, "size"):
                restore_replay_buffer(target, modified, False, 2)

    def test_in_range_wrong_non_balanced_pointer_raises(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        _, _, checkpoint = self.load(self.agent, False, 2)
        checkpoint["replay_buffer"]["pointer"] = 5

        with self.assertRaisesRegex(
            OnlineCheckpointError, r"pointer 5.*expected pointer 4"
        ):
            restore_replay_buffer(target, checkpoint, False, 2)

    def test_in_range_wrong_balanced_pointer_raises(self):
        self.save(self.make_balanced_buffer(), 3, True)
        target = ReplayBuffer.create(self.transition(0.0), size=5)
        _, _, checkpoint = self.load(self.agent, True, 0)
        checkpoint["replay_buffer"]["pointer"] = 4

        with self.assertRaisesRegex(
            OnlineCheckpointError, r"pointer 4.*expected pointer 3"
        ):
            restore_replay_buffer(target, checkpoint, True, 0)

    def test_in_range_wrong_size_raises(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        _, _, checkpoint = self.load(self.agent, False, 2)
        checkpoint["replay_buffer"]["size"] = 3

        with self.assertRaisesRegex(
            OnlineCheckpointError, r"size 3.*expected one of \[4\]"
        ):
            restore_replay_buffer(target, checkpoint, False, 2)

    def test_save_rejects_inconsistent_live_layout_without_files(self):
        replay_buffer = self.make_non_balanced_buffer()
        expected_pointer = replay_buffer.pointer
        expected_size = replay_buffer.size

        with self.subTest("pointer"):
            replay_buffer.pointer = expected_pointer - 1
            with self.assertRaisesRegex(
                OnlineCheckpointError, r"pointer 3.*expected pointer 4"
            ):
                self.save(replay_buffer, 2, False)
            self.assertEqual(os.listdir(self.save_dir), [])

        replay_buffer.pointer = expected_pointer
        with self.subTest("size"):
            replay_buffer.size = expected_size - 1
            with self.assertRaisesRegex(
                OnlineCheckpointError, r"size 3.*expected one of \[4\]"
            ):
                self.save(replay_buffer, 2, False)
            self.assertEqual(os.listdir(self.save_dir), [])

    def test_full_capacity_layout_is_compatible_and_strict(self):
        source = ReplayBuffer.create(self.transition(0.0), size=5)
        for value in range(1, 6):
            source.add_transition(self.transition(float(value)))
        self.assertEqual(source.pointer, 0)
        self.assertEqual(source.size, 4)

        self.save(source, online_step=5, balanced_sampling=True)
        _, _, checkpoint = self.load(self.agent, True, 0)
        target = ReplayBuffer.create(self.transition(0.0), size=5)
        restore_replay_buffer(target, checkpoint, True, 0)
        self.assertEqual(target.pointer, 0)
        self.assertEqual(target.size, 4)
        for key in source:
            np.testing.assert_array_equal(target[key], source[key])

        conventional_full_size = pickle.loads(pickle.dumps(checkpoint))
        conventional_full_size["replay_buffer"]["size"] = 5
        conventional_target = ReplayBuffer.create(self.transition(0.0), size=5)
        restore_replay_buffer(
            conventional_target, conventional_full_size, True, 0
        )
        self.assertEqual(conventional_target.pointer, 0)
        self.assertEqual(conventional_target.size, 5)

        invalid_full_size = pickle.loads(pickle.dumps(checkpoint))
        invalid_full_size["replay_buffer"]["size"] = 3
        with self.assertRaisesRegex(
            OnlineCheckpointError, r"size 3.*expected one of \[4, 5\]"
        ):
            restore_replay_buffer(
                ReplayBuffer.create(self.transition(0.0), size=5),
                invalid_full_size,
                True,
                0,
            )

    def test_restore_validates_all_fields_before_writing(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        before = {key: value.copy() for key, value in target.items()}
        _, _, checkpoint = self.load(self.agent, False, 2)
        checkpoint["replay_buffer"]["online_data"]["terminals"] = np.zeros(
            (2, 1), dtype=np.float32
        )

        with self.assertRaisesRegex(OnlineCheckpointError, "shape mismatch"):
            restore_replay_buffer(target, checkpoint, False, 2)
        for key in target:
            np.testing.assert_array_equal(target[key], before[key])

    def test_replay_data_count_must_match_online_step(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        _, _, checkpoint = self.load(self.agent, False, 2)
        checkpoint["replay_buffer"]["data_count"] = 1
        for key in checkpoint["replay_buffer"]["online_data"]:
            checkpoint["replay_buffer"]["online_data"][key] = checkpoint[
                "replay_buffer"
            ]["online_data"][key][:1]

        with self.assertRaisesRegex(OnlineCheckpointError, "does not match"):
            restore_replay_buffer(target, checkpoint, False, 2)

    def test_version_three_recent_buffer_disabled_round_trip(self):
        self.save(self.make_non_balanced_buffer(), 2, False)

        _, _, checkpoint = self.load(self.agent, False, 2)
        restored = restore_recent_dynamics_buffer(
            None, checkpoint, expected_capacity=0
        )

        self.assertEqual(checkpoint["format_version"], FORMAT_VERSION)
        self.assertEqual(checkpoint["recent_dynamics_capacity"], 0)
        self.assertIsNone(checkpoint["recent_dynamics_buffer"])
        self.assertIsNone(restored)

    def test_version_three_detector_disabled_round_trip(self):
        self.save(self.make_non_balanced_buffer(), 2, False)

        checkpoint = self.read_raw_checkpoint()
        _, restored_detector, loaded_checkpoint = self.load(
            self.agent, False, 2
        )

        self.assertEqual(FORMAT_VERSION, 3)
        self.assertEqual(checkpoint["format_version"], 3)
        self.assertIs(checkpoint["occupancy_detector_enabled"], False)
        self.assertIsNone(checkpoint["occupancy_detector_config"])
        self.assertIsNone(checkpoint["occupancy_detector"])
        self.assertIsNone(restored_detector)
        self.assertEqual(loaded_checkpoint["format_version"], 3)

    def test_version_three_detector_enabled_round_trip_restores_full_state(self):
        source = self.make_occupancy_detector(updates=4)
        config = self.occupancy_config()
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            occupancy_detector=source,
            occupancy_detector_config=config,
        )
        template = self.make_occupancy_detector(seed=999)

        _, restored, checkpoint = self.load(
            self.agent,
            False,
            2,
            occupancy_detector=template,
            occupancy_detector_config=config,
        )

        self.assertTrue(checkpoint["occupancy_detector_enabled"])
        self.assertEqual(checkpoint["occupancy_detector_config"], config)
        self.assertEqual(restored.network.step, source.network.step)
        np.testing.assert_array_equal(restored.rng, source.rng)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(restored.network.params),
            jax.tree_util.tree_leaves(source.network.params),
        ):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(restored.network.opt_state),
            jax.tree_util.tree_leaves(source.network.opt_state),
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_checkpoint_restored_detector_continues_exactly(self):
        source = self.make_occupancy_detector(updates=3)
        config = self.occupancy_config()
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            occupancy_detector=source,
            occupancy_detector_config=config,
        )
        _, restored, _ = self.load(
            self.agent,
            False,
            2,
            occupancy_detector=self.make_occupancy_detector(seed=999),
            occupancy_detector_config=config,
        )
        offline_batch = {
            field: self.initial_dataset()[field]
            for field in (
                "observations",
                "actions",
                "next_observations",
            )
        }
        online_batch = dict(offline_batch)
        online_batch["next_observations"] = (
            online_batch["next_observations"] + 2.0
        )

        uninterrupted, _ = source.update(
            offline_batch, online_batch
        )
        resumed, _ = restored.update(offline_batch, online_batch)

        for expected, actual in zip(
            jax.tree_util.tree_leaves(uninterrupted),
            jax.tree_util.tree_leaves(resumed),
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_interrupted_and_uninterrupted_detector_loops_match_exactly(self):
        split_step = 4
        final_step = 7
        recent_capacity = 8
        replay_capacity = 9
        config = self.occupancy_config()
        offline_dataset = Dataset.create(**self.initial_dataset())

        np.random.seed(2025)
        random.seed(2026)
        detector_a = self.make_occupancy_detector(seed=303)
        replay_a = ReplayBuffer.create(
            self.transition(0.0), size=replay_capacity
        )
        recent_a = self.make_recent_buffer(recent_capacity, [])
        online_rng_a = jax.random.PRNGKey(404)
        detector_a, online_rng_a, sampling_trace_a = (
            self.run_synthetic_detector_steps(
                detector_a,
                replay_a,
                recent_a,
                offline_dataset,
                online_rng_a,
                start_step=1,
                end_step=final_step,
            )
        )
        numpy_state_a = np.random.get_state()
        python_state_a = random.getstate()

        np.random.seed(2025)
        random.seed(2026)
        detector_b = self.make_occupancy_detector(seed=303)
        replay_b = ReplayBuffer.create(
            self.transition(0.0), size=replay_capacity
        )
        recent_b = self.make_recent_buffer(recent_capacity, [])
        online_rng_b = jax.random.PRNGKey(404)
        (
            detector_b,
            online_rng_b,
            sampling_trace_b_before,
        ) = self.run_synthetic_detector_steps(
            detector_b,
            replay_b,
            recent_b,
            offline_dataset,
            online_rng_b,
            start_step=1,
            end_step=split_step,
        )
        self.save(
            replay_b,
            online_step=split_step,
            balanced_sampling=True,
            recent_dynamics_buffer=recent_b,
            recent_dynamics_capacity=recent_capacity,
            occupancy_detector=detector_b,
            occupancy_detector_config=config,
            online_rng=online_rng_b,
        )

        replay_restored = ReplayBuffer.create(
            self.transition(0.0), size=replay_capacity
        )
        recent_restored = self.make_recent_buffer(recent_capacity, [])
        detector_template = self.make_occupancy_detector(seed=999)
        agent_template = DummyAgent(
            params={"weight": jnp.zeros(2, dtype=jnp.float32)},
            opt_state={"momentum": jnp.zeros(2, dtype=jnp.float32)},
        )
        np.random.seed(999)
        random.seed(999)
        (
            restored_agent,
            detector_restored,
            online_rng_restored,
            checkpoint,
        ) = load_online_checkpoint(
            self.save_dir,
            agent_template,
            replay_buffer=replay_restored,
            recent_dynamics_buffer=recent_restored,
            expected_env_name=self.env_name,
            expected_horizon_length=self.horizon_length,
            expected_balanced_sampling=True,
            expected_initial_replay_size=0,
            expected_action_dim=self.action_dim,
            expected_offline_steps=self.offline_steps,
            expected_recent_dynamics_capacity=recent_capacity,
            occupancy_detector=detector_template,
            expected_occupancy_detector_config=config,
        )
        self.assertEqual(online_start_step(checkpoint), split_step + 1)
        for expected, actual in zip(
            jax.tree_util.tree_leaves(self.agent),
            jax.tree_util.tree_leaves(restored_agent),
        ):
            np.testing.assert_array_equal(actual, expected)

        (
            detector_restored,
            online_rng_restored,
            sampling_trace_b_after,
        ) = (
            self.run_synthetic_detector_steps(
                detector_restored,
                replay_restored,
                recent_restored,
                offline_dataset,
                online_rng_restored,
                start_step=online_start_step(checkpoint),
                end_step=final_step,
            )
        )
        numpy_state_b = np.random.get_state()
        python_state_b = random.getstate()

        sampling_trace_b = (
            sampling_trace_b_before + sampling_trace_b_after
        )
        self.assertEqual(len(sampling_trace_b), len(sampling_trace_a))
        for expected, actual in zip(
            sampling_trace_a, sampling_trace_b
        ):
            self.assertEqual(set(actual), set(expected))
            for name in expected:
                np.testing.assert_array_equal(
                    actual[name], expected[name]
                )

        for expected, actual in zip(
            jax.tree_util.tree_leaves(detector_a),
            jax.tree_util.tree_leaves(detector_restored),
        ):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            int(detector_restored.network.step),
            int(detector_a.network.step),
        )
        np.testing.assert_array_equal(
            detector_restored.rng, detector_a.rng
        )
        np.testing.assert_array_equal(online_rng_restored, online_rng_a)

        self.assertEqual(replay_restored.pointer, replay_a.pointer)
        self.assertEqual(replay_restored.size, replay_a.size)
        for key in replay_a:
            np.testing.assert_array_equal(
                replay_restored[key], replay_a[key]
            )
        for field in ("size", "write_index", "total_added"):
            self.assertEqual(
                getattr(recent_restored, field),
                getattr(recent_a, field),
            )
        for key in recent_a.data:
            np.testing.assert_array_equal(
                recent_restored.data[key], recent_a.data[key]
            )

        self.assert_numpy_rng_state_equal(numpy_state_b, numpy_state_a)
        self.assertEqual(python_state_b, python_state_a)
        evaluation_batch = {
            field: self.initial_dataset()[field]
            for field in (
                "observations",
                "actions",
                "next_observations",
            )
        }
        np.testing.assert_array_equal(
            detector_restored.logits(evaluation_batch),
            detector_a.logits(evaluation_batch),
        )

        np.random.set_state(numpy_state_a)
        expected_offline_sample = offline_dataset.sample(4)
        expected_online_sample = recent_a.sample(4)
        np.random.set_state(numpy_state_b)
        restored_offline_sample = offline_dataset.sample(4)
        restored_online_sample = recent_restored.sample(4)
        for key in expected_offline_sample:
            np.testing.assert_array_equal(
                restored_offline_sample[key],
                expected_offline_sample[key],
            )
        for key in expected_online_sample:
            np.testing.assert_array_equal(
                restored_online_sample[key],
                expected_online_sample[key],
            )

    def test_version_three_detector_config_mismatch_raises(self):
        source = self.make_occupancy_detector(updates=1)
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            occupancy_detector=source,
            occupancy_detector_config=self.occupancy_config(),
        )

        with self.assertRaisesRegex(
            OnlineCheckpointError, "occupancy_detector_config"
        ):
            self.load(
                self.agent,
                False,
                2,
                occupancy_detector=self.make_occupancy_detector(seed=999),
                occupancy_detector_config=self.occupancy_config(
                    batch_size=4
                ),
            )

    def test_save_rejects_detector_metadata_that_mismatches_live_config(self):
        with self.assertRaisesRegex(
            OnlineCheckpointError, "detector config.*learning_rate"
        ):
            self.save(
                self.make_non_balanced_buffer(),
                2,
                False,
                occupancy_detector=self.make_occupancy_detector(),
                occupancy_detector_config=self.occupancy_config(
                    learning_rate=2e-3
                ),
            )

        self.assertEqual(os.listdir(self.save_dir), [])

    def test_version_three_detector_enabled_disabled_mismatch_raises(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        with self.subTest("checkpoint disabled, current enabled"):
            with self.assertRaisesRegex(
                OnlineCheckpointError, "occupancy_detector_enabled"
            ):
                self.load(
                    self.agent,
                    False,
                    2,
                    occupancy_detector=self.make_occupancy_detector(),
                    occupancy_detector_config=self.occupancy_config(),
                )

        os.remove(os.path.join(self.save_dir, CHECKPOINT_FILENAME))
        os.remove(os.path.join(self.save_dir, "progress.tk"))
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            occupancy_detector=self.make_occupancy_detector(updates=1),
            occupancy_detector_config=self.occupancy_config(),
        )
        with self.subTest("checkpoint enabled, current disabled"):
            with self.assertRaisesRegex(
                OnlineCheckpointError, "occupancy_detector_enabled"
            ):
                self.load(self.agent, False, 2)

    def test_version_three_corrupt_detector_state_raises(self):
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            occupancy_detector=self.make_occupancy_detector(updates=1),
            occupancy_detector_config=self.occupancy_config(),
        )
        checkpoint = self.read_raw_checkpoint()
        checkpoint["occupancy_detector"]["rng"] = np.zeros(
            3, dtype=np.uint32
        )
        self.write_raw_checkpoint(checkpoint)

        with self.assertRaisesRegex(
            OnlineCheckpointError,
            r"occupancy_detector.*rng.*shape mismatch",
        ):
            self.load(
                self.agent,
                False,
                2,
                occupancy_detector=self.make_occupancy_detector(seed=999),
                occupancy_detector_config=self.occupancy_config(),
            )

    def test_detector_restore_failure_does_not_modify_replay_or_recent(self):
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            recent_dynamics_buffer=self.make_recent_buffer(4, [1, 2]),
            recent_dynamics_capacity=4,
            occupancy_detector=self.make_occupancy_detector(updates=1),
            occupancy_detector_config=self.occupancy_config(),
        )
        checkpoint = self.read_raw_checkpoint()
        checkpoint["occupancy_detector"]["rng"] = np.zeros(
            3, dtype=np.uint32
        )
        self.write_raw_checkpoint(checkpoint)
        replay_target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        replay_before = {
            key: value.copy() for key, value in replay_target.items()
        }
        replay_pointer_before = replay_target.pointer
        replay_size_before = replay_target.size
        recent_target = self.make_recent_buffer(4, [99])
        recent_before = recent_target.state_dict()
        numpy_before = np.random.get_state()
        python_before = random.getstate()

        with self.assertRaisesRegex(
            OnlineCheckpointError, "occupancy_detector"
        ):
            load_online_checkpoint(
                self.save_dir,
                self.agent,
                replay_buffer=replay_target,
                recent_dynamics_buffer=recent_target,
                expected_env_name=self.env_name,
                expected_horizon_length=self.horizon_length,
                expected_balanced_sampling=False,
                expected_initial_replay_size=2,
                expected_action_dim=self.action_dim,
                expected_offline_steps=self.offline_steps,
                expected_recent_dynamics_capacity=4,
                occupancy_detector=self.make_occupancy_detector(seed=999),
                expected_occupancy_detector_config=(
                    self.occupancy_config()
                ),
            )

        self.assertEqual(replay_target.pointer, replay_pointer_before)
        self.assertEqual(replay_target.size, replay_size_before)
        for key, value in replay_target.items():
            np.testing.assert_array_equal(value, replay_before[key])
        recent_after = recent_target.state_dict()
        for field in ("capacity", "size", "write_index", "total_added"):
            self.assertEqual(recent_after[field], recent_before[field])
        for key, value in recent_after["data"].items():
            np.testing.assert_array_equal(
                value, recent_before["data"][key]
            )
        self.assert_numpy_rng_state_equal(
            np.random.get_state(), numpy_before
        )
        self.assertEqual(random.getstate(), python_before)

    def test_recent_buffer_underfilled_round_trip(self):
        source = self.make_recent_buffer(4, [1, 2])
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            recent_dynamics_buffer=source,
            recent_dynamics_capacity=4,
        )
        _, _, checkpoint = self.load(self.agent, False, 2, 4)
        target = self.make_recent_buffer(4, [])

        restored = restore_recent_dynamics_buffer(
            target, checkpoint, expected_capacity=4
        )

        self.assertIs(restored, target)
        self.assertEqual(target.size, 2)
        self.assertEqual(target.write_index, 2)
        self.assertEqual(target.total_added, 2)
        for key in source.data:
            np.testing.assert_array_equal(
                target.ordered_data()[key],
                source.ordered_data()[key],
            )

    def test_recent_buffer_wrapped_round_trip(self):
        replay_buffer = ReplayBuffer.create(self.transition(0.0), size=6)
        for value in range(1, 6):
            replay_buffer.add_transition(self.transition(float(value)))
        source = self.make_recent_buffer(3, [1, 2, 3, 4, 5])
        self.save(
            replay_buffer,
            5,
            True,
            recent_dynamics_buffer=source,
            recent_dynamics_capacity=3,
        )
        _, _, checkpoint = self.load(self.agent, True, 0, 3)
        target = self.make_recent_buffer(3, [])

        restore_recent_dynamics_buffer(
            target, checkpoint, expected_capacity=3
        )

        self.assertEqual(target.size, 3)
        self.assertEqual(target.write_index, 2)
        self.assertEqual(target.total_added, 5)
        np.testing.assert_array_equal(
            target.ordered_data()["rewards"],
            np.array([3, 4, 5], dtype=np.float32),
        )

    def test_recent_buffer_next_write_position_after_restore(self):
        replay_buffer = ReplayBuffer.create(self.transition(0.0), size=6)
        for value in range(1, 6):
            replay_buffer.add_transition(self.transition(float(value)))
        source = self.make_recent_buffer(3, [1, 2, 3, 4, 5])
        self.save(
            replay_buffer,
            5,
            True,
            recent_dynamics_buffer=source,
            recent_dynamics_capacity=3,
        )
        _, _, checkpoint = self.load(self.agent, True, 0, 3)
        target = self.make_recent_buffer(3, [])
        restore_recent_dynamics_buffer(
            target, checkpoint, expected_capacity=3
        )

        target.add_transition(self.transition(6.0))

        np.testing.assert_array_equal(
            target.ordered_data()["rewards"],
            np.array([4, 5, 6], dtype=np.float32),
        )
        self.assertEqual(target.write_index, 0)
        self.assertEqual(target.total_added, 6)

    def test_recent_buffer_capacity_mismatch_raises(self):
        recent = self.make_recent_buffer(4, [1, 2])
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            recent_dynamics_buffer=recent,
            recent_dynamics_capacity=4,
        )

        with self.assertRaisesRegex(
            OnlineCheckpointError, "recent_dynamics_capacity"
        ):
            self.load(self.agent, False, 2, 5)

    def test_recent_buffer_enabled_disabled_mismatch_raises(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        with self.subTest("checkpoint disabled, current enabled"):
            with self.assertRaisesRegex(
                OnlineCheckpointError, "recent_dynamics_capacity"
            ):
                self.load(self.agent, False, 2, 2)

        os.remove(os.path.join(self.save_dir, CHECKPOINT_FILENAME))
        os.remove(os.path.join(self.save_dir, "progress.tk"))
        recent = self.make_recent_buffer(2, [1, 2])
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            recent_dynamics_buffer=recent,
            recent_dynamics_capacity=2,
        )
        with self.subTest("checkpoint enabled, current disabled"):
            with self.assertRaisesRegex(
                OnlineCheckpointError, "recent_dynamics_capacity"
            ):
                self.load(self.agent, False, 2, 0)

    def test_recent_buffer_key_shape_and_dtype_mismatch_raise(self):
        recent = self.make_recent_buffer(4, [1, 2])
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            recent_dynamics_buffer=recent,
            recent_dynamics_capacity=4,
        )
        _, _, checkpoint = self.load(self.agent, False, 2, 4)

        with self.subTest("key"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            modified["recent_dynamics_buffer"]["data"].pop("actions")
            with self.assertRaisesRegex(OnlineCheckpointError, "key mismatch"):
                restore_recent_dynamics_buffer(
                    self.make_recent_buffer(4, []),
                    modified,
                    expected_capacity=4,
                )

        with self.subTest("shape"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            modified["recent_dynamics_buffer"]["data"]["actions"] = np.zeros(
                (4, 2), dtype=np.float32
            )
            with self.assertRaisesRegex(OnlineCheckpointError, "shape mismatch"):
                restore_recent_dynamics_buffer(
                    self.make_recent_buffer(4, []),
                    modified,
                    expected_capacity=4,
                )

        with self.subTest("dtype"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            actions = modified["recent_dynamics_buffer"]["data"]["actions"]
            modified["recent_dynamics_buffer"]["data"]["actions"] = (
                actions.astype(np.float64)
            )
            with self.assertRaisesRegex(OnlineCheckpointError, "dtype mismatch"):
                restore_recent_dynamics_buffer(
                    self.make_recent_buffer(4, []),
                    modified,
                    expected_capacity=4,
                )

    def test_recent_buffer_corrupt_metadata_raises(self):
        recent = self.make_recent_buffer(4, [1, 2])
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            recent_dynamics_buffer=recent,
            recent_dynamics_capacity=4,
        )
        checkpoint = self.read_raw_checkpoint()

        with self.subTest("size"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            modified["recent_dynamics_buffer"]["size"] = 1
            with self.assertRaisesRegex(OnlineCheckpointError, "size"):
                restore_recent_dynamics_buffer(
                    self.make_recent_buffer(4, []),
                    modified,
                    expected_capacity=4,
                )

        with self.subTest("write_index"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            modified["recent_dynamics_buffer"]["write_index"] = 3
            with self.assertRaisesRegex(OnlineCheckpointError, "write_index"):
                restore_recent_dynamics_buffer(
                    self.make_recent_buffer(4, []),
                    modified,
                    expected_capacity=4,
                )

        with self.subTest("total_added"):
            modified = pickle.loads(pickle.dumps(checkpoint))
            modified["recent_dynamics_buffer"]["total_added"] = 3
            modified["recent_dynamics_buffer"]["size"] = 3
            modified["recent_dynamics_buffer"]["write_index"] = 3
            with self.assertRaisesRegex(
                OnlineCheckpointError, "total_added 3.*online_step 2"
            ):
                restore_recent_dynamics_buffer(
                    self.make_recent_buffer(4, []),
                    modified,
                    expected_capacity=4,
                )

    def test_version_one_checkpoint_loads_when_recent_disabled(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        checkpoint = self.read_raw_checkpoint()
        checkpoint["format_version"] = 1
        checkpoint.pop("recent_dynamics_capacity")
        checkpoint.pop("recent_dynamics_buffer")
        self.write_raw_checkpoint(checkpoint)

        _, _, restored_checkpoint = self.load(self.agent, False, 2, 0)

        self.assertEqual(restored_checkpoint["format_version"], 1)
        self.assertIsNone(
            restore_recent_dynamics_buffer(
                None, restored_checkpoint, expected_capacity=0
            )
        )

    def test_version_one_checkpoint_rejected_when_recent_enabled(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        checkpoint = self.read_raw_checkpoint()
        checkpoint["format_version"] = 1
        checkpoint.pop("recent_dynamics_capacity")
        checkpoint.pop("recent_dynamics_buffer")
        self.write_raw_checkpoint(checkpoint)

        with self.assertRaisesRegex(
            OnlineCheckpointError,
            "format_version 1.*does not contain.*recent_dynamics_buffer",
        ):
            self.load(self.agent, False, 2, 4)

    def test_version_two_checkpoint_loads_when_detector_disabled(self):
        recent = self.make_recent_buffer(4, [1, 2])
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            recent_dynamics_buffer=recent,
            recent_dynamics_capacity=4,
        )
        checkpoint = self.read_raw_checkpoint()
        checkpoint["format_version"] = 2
        checkpoint.pop("occupancy_detector_enabled")
        checkpoint.pop("occupancy_detector_config")
        checkpoint.pop("occupancy_detector")
        self.write_raw_checkpoint(checkpoint)

        _, restored_detector, restored_checkpoint = self.load(
            self.agent, False, 2, 4
        )
        restored_recent = restore_recent_dynamics_buffer(
            self.make_recent_buffer(4, []),
            restored_checkpoint,
            expected_capacity=4,
        )

        self.assertIsNone(restored_detector)
        np.testing.assert_array_equal(
            restored_recent.ordered_data()["rewards"],
            np.array([1, 2], dtype=np.float32),
        )

    def test_version_one_checkpoint_rejected_when_detector_enabled(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        checkpoint = self.read_raw_checkpoint()
        checkpoint["format_version"] = 1
        checkpoint.pop("recent_dynamics_capacity")
        checkpoint.pop("recent_dynamics_buffer")
        checkpoint.pop("occupancy_detector_enabled")
        checkpoint.pop("occupancy_detector_config")
        checkpoint.pop("occupancy_detector")
        self.write_raw_checkpoint(checkpoint)

        with self.assertRaisesRegex(
            OnlineCheckpointError,
            "format_version 1.*does not contain occupancy detector",
        ):
            self.load(
                self.agent,
                False,
                2,
                occupancy_detector=self.make_occupancy_detector(),
                occupancy_detector_config=self.occupancy_config(),
            )

    def test_version_two_checkpoint_rejected_when_detector_enabled(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        checkpoint = self.read_raw_checkpoint()
        checkpoint["format_version"] = 2
        checkpoint.pop("occupancy_detector_enabled")
        checkpoint.pop("occupancy_detector_config")
        checkpoint.pop("occupancy_detector")
        self.write_raw_checkpoint(checkpoint)

        with self.assertRaisesRegex(
            OnlineCheckpointError,
            "format_version 2.*does not contain occupancy detector",
        ):
            self.load(
                self.agent,
                False,
                2,
                occupancy_detector=self.make_occupancy_detector(),
                occupancy_detector_config=self.occupancy_config(),
            )

    def test_save_rejects_recent_total_added_mismatch_without_files(self):
        recent = self.make_recent_buffer(4, [1])

        with self.assertRaisesRegex(
            OnlineCheckpointError, "total_added 1.*online_step 2"
        ):
            self.save(
                self.make_non_balanced_buffer(),
                2,
                False,
                recent_dynamics_buffer=recent,
                recent_dynamics_capacity=4,
            )

        self.assertEqual(os.listdir(self.save_dir), [])

    def test_save_rejects_recent_enabled_disabled_mismatch(self):
        with self.subTest("capacity zero with buffer"):
            with self.assertRaisesRegex(
                OnlineCheckpointError, "must be None"
            ):
                self.save(
                    self.make_non_balanced_buffer(),
                    2,
                    False,
                    recent_dynamics_buffer=self.make_recent_buffer(2, [1, 2]),
                    recent_dynamics_capacity=0,
                )
            self.assertEqual(os.listdir(self.save_dir), [])

        with self.subTest("positive capacity without buffer"):
            with self.assertRaisesRegex(
                OnlineCheckpointError, "RecentDynamicsBuffer"
            ):
                self.save(
                    self.make_non_balanced_buffer(),
                    2,
                    False,
                    recent_dynamics_buffer=None,
                    recent_dynamics_capacity=2,
                )
            self.assertEqual(os.listdir(self.save_dir), [])

    def test_corrupt_recent_checkpoint_does_not_modify_template(self):
        recent = self.make_recent_buffer(4, [1, 2])
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            recent_dynamics_buffer=recent,
            recent_dynamics_capacity=4,
        )
        _, _, checkpoint = self.load(self.agent, False, 2, 4)
        checkpoint["recent_dynamics_buffer"]["data"]["actions"] = np.zeros(
            (4, 2), dtype=np.float32
        )
        target = self.make_recent_buffer(4, [99])
        before = target.state_dict()

        with self.assertRaisesRegex(OnlineCheckpointError, "shape mismatch"):
            restore_recent_dynamics_buffer(
                target, checkpoint, expected_capacity=4
            )

        after = target.state_dict()
        for field in ("capacity", "size", "write_index", "total_added"):
            self.assertEqual(after[field], before[field])
        for key in before["data"]:
            np.testing.assert_array_equal(
                after["data"][key], before["data"][key]
            )

    def test_restore_preflight_rejects_recent_layout_before_mutation(self):
        recent = self.make_recent_buffer(4, [1, 2])
        self.save(
            self.make_non_balanced_buffer(),
            2,
            False,
            recent_dynamics_buffer=recent,
            recent_dynamics_capacity=4,
        )
        _, _, checkpoint = self.load(self.agent, False, 2, 4)
        replay_target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        replay_before = {
            key: value.copy() for key, value in replay_target.items()
        }
        pointer_before = replay_target.pointer
        size_before = replay_target.size
        incompatible_transition = self.transition(0.0)
        incompatible_transition["actions"] = np.zeros(2, dtype=np.float32)
        recent_target = RecentDynamicsBuffer.create(
            incompatible_transition, capacity=4
        )
        recent_before = recent_target.state_dict()
        numpy_before = np.random.get_state()
        python_before = random.getstate()

        with self.assertRaisesRegex(OnlineCheckpointError, "shape mismatch"):
            validate_online_checkpoint_restore(
                replay_target,
                recent_target,
                checkpoint,
                balanced_sampling=False,
                initial_replay_size=2,
                expected_recent_dynamics_capacity=4,
            )

        self.assertEqual(replay_target.pointer, pointer_before)
        self.assertEqual(replay_target.size, size_before)
        for key, value in replay_target.items():
            np.testing.assert_array_equal(value, replay_before[key])
        recent_after = recent_target.state_dict()
        for field in ("capacity", "size", "write_index", "total_added"):
            self.assertEqual(recent_after[field], recent_before[field])
        for key, value in recent_after["data"].items():
            np.testing.assert_array_equal(
                value, recent_before["data"][key]
            )
        self.assert_numpy_rng_state_equal(
            np.random.get_state(), numpy_before
        )
        self.assertEqual(random.getstate(), python_before)


if __name__ == "__main__":
    unittest.main()
