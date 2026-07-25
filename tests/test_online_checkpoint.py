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

from utils.datasets import ReplayBuffer
from utils.online_checkpoint import (
    CHECKPOINT_FILENAME,
    FORMAT_VERSION,
    OnlineCheckpointError,
    load_online_checkpoint,
    online_start_step,
    read_progress,
    restore_replay_buffer,
    restore_rng_states,
    save_online_checkpoint,
    should_save_online_checkpoint,
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

    def save(self, replay_buffer, online_step, balanced_sampling):
        initial_replay_size = 0 if balanced_sampling else 2
        return save_online_checkpoint(
            save_dir=self.save_dir,
            agent=self.agent,
            replay_buffer=replay_buffer,
            online_rng=self.online_rng,
            online_step=online_step,
            offline_steps=self.offline_steps,
            balanced_sampling=balanced_sampling,
            initial_replay_size=initial_replay_size,
            action_dim=self.action_dim,
            horizon_length=self.horizon_length,
            env_name=self.env_name,
            done=True,
            action_queue=[],
        )

    def load(self, agent, balanced_sampling, initial_replay_size):
        return load_online_checkpoint(
            self.save_dir,
            agent,
            expected_env_name=self.env_name,
            expected_horizon_length=self.horizon_length,
            expected_balanced_sampling=balanced_sampling,
            expected_initial_replay_size=initial_replay_size,
            expected_action_dim=self.action_dim,
            expected_offline_steps=self.offline_steps,
        )

    def read_raw_checkpoint(self):
        with open(os.path.join(self.save_dir, CHECKPOINT_FILENAME), "rb") as file:
            return pickle.load(file)

    def test_non_balanced_replay_buffer_round_trip(self):
        source = self.make_non_balanced_buffer()
        self.save(source, online_step=2, balanced_sampling=False)
        target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        _, checkpoint = self.load(self.agent, False, 2)

        restore_replay_buffer(target, checkpoint, False, 2)

        for key in source:
            np.testing.assert_array_equal(target[key][:4], source[key][:4])

    def test_balanced_replay_buffer_round_trip(self):
        source = self.make_balanced_buffer()
        self.save(source, online_step=3, balanced_sampling=True)
        target = ReplayBuffer.create(self.transition(0.0), size=5)
        _, checkpoint = self.load(self.agent, True, 0)

        restore_replay_buffer(target, checkpoint, True, 0)

        for key in source:
            np.testing.assert_array_equal(target[key][:3], source[key][:3])

    def test_agent_state_round_trip(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        template = DummyAgent(
            params={"weight": jnp.zeros(2, dtype=jnp.float32)},
            opt_state={"momentum": jnp.zeros(2, dtype=jnp.float32)},
        )

        restored_agent, _ = self.load(template, False, 2)

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
        _, checkpoint = self.load(self.agent, False, 2)

        restore_replay_buffer(target, checkpoint, False, 2)

        self.assertEqual(target.pointer, source.pointer)
        self.assertEqual(target.size, source.size)
        self.assertEqual(target.max_size, source.max_size)

    def test_online_data_shape_and_dtype_round_trip(self):
        source = self.make_balanced_buffer()
        self.save(source, 3, True)
        target = ReplayBuffer.create(self.transition(0.0), size=5)
        _, checkpoint = self.load(self.agent, True, 0)
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
        _, checkpoint = self.load(self.agent, False, 2)

        restored_rng = restore_rng_states(checkpoint)

        np.testing.assert_array_equal(restored_rng, self.online_rng)

    def test_numpy_rng_state_round_trip(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        expected = np.random.random(4)
        np.random.seed(999)
        _, checkpoint = self.load(self.agent, False, 2)

        restore_rng_states(checkpoint)

        np.testing.assert_array_equal(np.random.random(4), expected)

    def test_python_random_state_round_trip(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        expected = [random.random() for _ in range(4)]
        random.seed(999)
        _, checkpoint = self.load(self.agent, False, 2)

        restore_rng_states(checkpoint)

        self.assertEqual([random.random() for _ in range(4)], expected)

    def test_incompatible_env_name_raises(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        with self.assertRaisesRegex(OnlineCheckpointError, "env_name"):
            load_online_checkpoint(
                self.save_dir,
                self.agent,
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
        _, checkpoint = self.load(self.agent, False, 2)

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
        _, checkpoint = self.load(self.agent, False, 2)

        self.assertEqual(stage, "online")
        self.assertEqual(saved_step, 2)
        self.assertEqual(online_start_step(checkpoint), 3)

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
        _, checkpoint = self.load(self.agent, False, 2)

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

    def test_replay_data_count_must_match_online_step(self):
        self.save(self.make_non_balanced_buffer(), 2, False)
        target = ReplayBuffer.create_from_initial_dataset(
            self.initial_dataset(), size=6
        )
        _, checkpoint = self.load(self.agent, False, 2)
        checkpoint["replay_buffer"]["data_count"] = 1
        for key in checkpoint["replay_buffer"]["online_data"]:
            checkpoint["replay_buffer"]["online_data"][key] = checkpoint[
                "replay_buffer"
            ]["online_data"][key][:1]

        with self.assertRaisesRegex(OnlineCheckpointError, "does not match"):
            restore_replay_buffer(target, checkpoint, False, 2)


if __name__ == "__main__":
    unittest.main()
