import unittest

import numpy as np

from utils.recent_dynamics_buffer import (
    ONLINE_TRANSITION_FIELDS,
    RecentDynamicsBuffer,
    create_recent_transition_template,
)


class RecentDynamicsBufferTest(unittest.TestCase):
    def example_transition(self):
        return {
            "observations": np.zeros(2, dtype=np.float32),
            "actions": np.zeros(1, dtype=np.float32),
            "rewards": np.float32(0.0),
            "terminals": np.bool_(False),
        }

    def transition(self, value):
        return {
            "observations": np.array(
                [value, value + 0.25], dtype=np.float32
            ),
            "actions": np.array([value], dtype=np.float32),
            "rewards": np.float32(value),
            "terminals": np.bool_(value < 0),
        }

    def make_buffer(self, capacity=4):
        return RecentDynamicsBuffer.create(
            self.example_transition(), capacity=capacity
        )

    def add_values(self, buffer, values):
        for value in values:
            buffer.add_transition(self.transition(float(value)))

    def test_create_empty_buffer(self):
        buffer = self.make_buffer(3)

        self.assertEqual(buffer.capacity, 3)
        self.assertEqual(buffer.size, 0)
        self.assertEqual(buffer.write_index, 0)
        self.assertEqual(buffer.total_added, 0)
        for key, value in buffer.ordered_data().items():
            self.assertEqual(
                value.shape, (0, *self.example_transition()[key].shape)
            )

    def test_invalid_capacity(self):
        for capacity in (0, -1, True, 1.5):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(ValueError, "capacity"):
                    RecentDynamicsBuffer.create(
                        self.example_transition(), capacity
                    )

    def test_online_template_uses_single_step_storage_layout(self):
        replay_storage = {
            "observations": np.zeros((8, 3), dtype=np.float32),
            "actions": np.zeros((8, 2), dtype=np.float32),
            "rewards": np.zeros(8, dtype=np.float32),
            "terminals": np.zeros(8, dtype=np.float32),
            "masks": np.zeros(8, dtype=np.float32),
            "next_observations": np.zeros((8, 3), dtype=np.float32),
            "extra_metadata": np.zeros((8, 1), dtype=np.float64),
        }
        chunked_dataset_actions = np.zeros((5, 2), dtype=np.float32)

        template = create_recent_transition_template(replay_storage)

        self.assertEqual(set(template), set(ONLINE_TRANSITION_FIELDS))
        self.assertEqual(template["observations"].shape, (3,))
        self.assertEqual(template["actions"].shape, (2,))
        self.assertEqual(template["rewards"].shape, ())
        self.assertEqual(template["actions"].dtype, np.dtype(np.float32))
        self.assertNotEqual(
            template["actions"].shape,
            chunked_dataset_actions.shape,
        )

    def test_prepare_online_transition_matches_template(self):
        replay_storage = {
            "observations": np.zeros((8, 3), dtype=np.float32),
            "actions": np.zeros((8, 2), dtype=np.float32),
            "rewards": np.zeros(8, dtype=np.float32),
            "terminals": np.zeros(8, dtype=np.float32),
            "masks": np.zeros(8, dtype=np.float32),
            "next_observations": np.zeros((8, 3), dtype=np.float32),
        }
        template = create_recent_transition_template(replay_storage)
        buffer = RecentDynamicsBuffer.create(template, capacity=4)
        raw_transition = {
            "observations": [1.0, 2.0, 3.0],
            "actions": [0.1, 0.2],
            "rewards": 1.5,
            "terminals": 0.0,
            "masks": 1.0,
            "next_observations": [2.0, 3.0, 4.0],
        }

        prepared = buffer.prepare_transition(raw_transition)

        self.assertEqual(set(prepared), set(template))
        for key, value in prepared.items():
            self.assertEqual(value.shape, template[key].shape)
            self.assertEqual(value.dtype, template[key].dtype)
        buffer.add_transition(prepared)

    def test_object_dtype_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "object dtype"):
            RecentDynamicsBuffer.create(
                {"objects": np.array([object()], dtype=object)},
                capacity=2,
            )

    def test_ordered_data_before_full(self):
        buffer = self.make_buffer()
        self.add_values(buffer, [1, 2, 3])

        np.testing.assert_array_equal(
            buffer.ordered_data()["rewards"],
            np.array([1, 2, 3], dtype=np.float32),
        )

    def test_ordered_data_exactly_full(self):
        buffer = self.make_buffer()
        self.add_values(buffer, [1, 2, 3, 4])

        np.testing.assert_array_equal(
            buffer.ordered_data()["rewards"],
            np.array([1, 2, 3, 4], dtype=np.float32),
        )
        self.assertEqual(buffer.write_index, 0)

    def test_ordered_data_after_one_wrap(self):
        buffer = self.make_buffer()
        self.add_values(buffer, [1, 2, 3, 4, 5, 6])

        np.testing.assert_array_equal(
            buffer.ordered_data()["rewards"],
            np.array([3, 4, 5, 6], dtype=np.float32),
        )

    def test_multiple_wraps_keep_only_recent_capacity(self):
        buffer = self.make_buffer(3)
        self.add_values(buffer, range(1, 11))

        np.testing.assert_array_equal(
            buffer.ordered_data()["rewards"],
            np.array([8, 9, 10], dtype=np.float32),
        )
        self.assertEqual(buffer.size, 3)
        self.assertEqual(buffer.total_added, 10)
        self.assertEqual(buffer.write_index, 1)

    def test_multiple_field_shapes_are_preserved(self):
        buffer = self.make_buffer()
        buffer.add_transition(self.transition(2.0))
        ordered = buffer.ordered_data()

        self.assertEqual(ordered["observations"].shape, (1, 2))
        self.assertEqual(ordered["actions"].shape, (1, 1))
        self.assertEqual(ordered["rewards"].shape, (1,))
        self.assertEqual(ordered["terminals"].shape, (1,))

    def test_multiple_field_dtypes_are_preserved(self):
        buffer = self.make_buffer()
        buffer.add_transition(self.transition(2.0))
        ordered = buffer.ordered_data()

        self.assertEqual(ordered["observations"].dtype, np.dtype(np.float32))
        self.assertEqual(ordered["actions"].dtype, np.dtype(np.float32))
        self.assertEqual(ordered["rewards"].dtype, np.dtype(np.float32))
        self.assertEqual(ordered["terminals"].dtype, np.dtype(np.bool_))

    def test_sample_uses_only_valid_data(self):
        buffer = self.make_buffer(3)
        self.add_values(buffer, [1, 2, 3, 4, 5])

        batch = buffer.sample(100, rng=np.random.default_rng(9))

        self.assertEqual(batch["rewards"].shape, (100,))
        self.assertTrue(
            set(np.unique(batch["rewards"])).issubset({3.0, 4.0, 5.0})
        )

    def test_sample_empty_buffer_fails(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.make_buffer().sample(1)

    def test_invalid_batch_size(self):
        buffer = self.make_buffer()
        buffer.add_transition(self.transition(1.0))
        for batch_size in (0, -1, True, 1.5):
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    buffer.sample(batch_size)

    def test_transition_missing_key_fails(self):
        buffer = self.make_buffer()
        transition = self.transition(1.0)
        transition.pop("actions")

        with self.assertRaisesRegex(ValueError, "key mismatch"):
            buffer.add_transition(transition)

    def test_transition_extra_key_fails(self):
        buffer = self.make_buffer()
        transition = self.transition(1.0)
        transition["extra"] = np.float32(0)

        with self.assertRaisesRegex(ValueError, "key mismatch"):
            buffer.add_transition(transition)

    def test_transition_shape_mismatch_fails(self):
        buffer = self.make_buffer()
        transition = self.transition(1.0)
        transition["actions"] = np.zeros(2, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            buffer.add_transition(transition)

    def test_transition_dtype_mismatch_fails(self):
        buffer = self.make_buffer()
        transition = self.transition(1.0)
        transition["actions"] = transition["actions"].astype(np.float64)

        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            buffer.add_transition(transition)

    def test_state_dict_round_trip(self):
        source = self.make_buffer()
        self.add_values(source, [1, 2])
        target = self.make_buffer()

        target.load_state_dict(source.state_dict())

        self.assertEqual(target.size, source.size)
        self.assertEqual(target.write_index, source.write_index)
        self.assertEqual(target.total_added, source.total_added)
        for key in source.data:
            np.testing.assert_array_equal(target.data[key], source.data[key])

    def test_wrapped_state_dict_round_trip(self):
        source = self.make_buffer()
        self.add_values(source, [1, 2, 3, 4, 5, 6])
        target = self.make_buffer()

        target.load_state_dict(source.state_dict())

        for key in source.data:
            np.testing.assert_array_equal(
                target.ordered_data()[key],
                source.ordered_data()[key],
            )

    def test_next_write_position_after_restore(self):
        source = self.make_buffer()
        self.add_values(source, [1, 2, 3, 4, 5])
        target = self.make_buffer()
        target.load_state_dict(source.state_dict())

        target.add_transition(self.transition(6.0))

        np.testing.assert_array_equal(
            target.ordered_data()["rewards"],
            np.array([3, 4, 5, 6], dtype=np.float32),
        )

    def test_corrupt_size_is_rejected(self):
        state = self.make_buffer().state_dict()
        state["size"] = 1

        with self.assertRaisesRegex(ValueError, "size"):
            self.make_buffer().load_state_dict(state)

    def test_corrupt_write_index_is_rejected(self):
        source = self.make_buffer()
        source.add_transition(self.transition(1.0))
        state = source.state_dict()
        state["write_index"] = 2

        with self.assertRaisesRegex(ValueError, "write_index"):
            self.make_buffer().load_state_dict(state)

    def test_corrupt_total_added_is_rejected(self):
        source = self.make_buffer()
        source.add_transition(self.transition(1.0))
        state = source.state_dict()
        state["total_added"] = 0

        with self.assertRaisesRegex(ValueError, "total_added|size"):
            self.make_buffer().load_state_dict(state)

    def test_capacity_mismatch_is_rejected(self):
        state = self.make_buffer(3).state_dict()

        with self.assertRaisesRegex(ValueError, "capacity mismatch"):
            self.make_buffer(4).load_state_dict(state)

    def test_state_key_mismatch_is_rejected(self):
        state = self.make_buffer().state_dict()
        state["data"].pop("actions")

        with self.assertRaisesRegex(ValueError, "key mismatch"):
            self.make_buffer().load_state_dict(state)

    def test_state_shape_mismatch_is_rejected(self):
        state = self.make_buffer().state_dict()
        state["data"]["actions"] = np.zeros((4, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.make_buffer().load_state_dict(state)

    def test_state_dtype_mismatch_is_rejected(self):
        state = self.make_buffer().state_dict()
        state["data"]["actions"] = state["data"]["actions"].astype(np.float64)

        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            self.make_buffer().load_state_dict(state)

    def test_clear_makes_buffer_logically_empty(self):
        buffer = self.make_buffer()
        self.add_values(buffer, [1, 2, 3])

        buffer.clear()

        self.assertEqual(buffer.size, 0)
        self.assertEqual(buffer.write_index, 0)
        self.assertEqual(buffer.total_added, 0)
        for value in buffer.ordered_data().values():
            self.assertEqual(value.shape[0], 0)

    def test_ordered_data_returns_safe_copies(self):
        buffer = self.make_buffer()
        buffer.add_transition(self.transition(1.0))

        ordered = buffer.ordered_data()
        ordered["rewards"][0] = np.float32(99)

        self.assertEqual(buffer.ordered_data()["rewards"][0], np.float32(1))

    def test_sample_returns_safe_copies(self):
        buffer = self.make_buffer()
        buffer.add_transition(self.transition(1.0))

        sample = buffer.sample(1, rng=np.random.default_rng(0))
        sample["rewards"][0] = np.float32(99)

        self.assertEqual(buffer.ordered_data()["rewards"][0], np.float32(1))

    def test_state_dict_returns_safe_copies(self):
        buffer = self.make_buffer()
        buffer.add_transition(self.transition(1.0))

        state = buffer.state_dict()
        state["data"]["rewards"][0] = np.float32(99)

        self.assertEqual(buffer.ordered_data()["rewards"][0], np.float32(1))


if __name__ == "__main__":
    unittest.main()
