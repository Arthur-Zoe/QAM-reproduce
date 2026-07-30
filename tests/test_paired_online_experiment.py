import hashlib
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import flax
import jax.numpy as jnp
import numpy as np

from evaluation import evaluate
from main import (
    FLAGS,
    build_bridge_evaluation_action_transform_factory,
    build_evaluation_episode_seeds,
    complete_bridge_evaluation_diagnostics,
    reset_online_environment,
    resolve_offline_training_start_step,
    resolve_phase_eval_interval,
    restore_offline_agent_checkpoint,
    validate_offline_agent_checkpoint_flags,
)


class TinyNetwork(flax.struct.PyTreeNode):
    step: object
    value: object


class TinyAgent(flax.struct.PyTreeNode):
    network: TinyNetwork


class SeededEvaluationEnvironment:
    def __init__(self):
        self.reset_seeds = []
        self.reset_kwargs = []
        self.current_seed = None

    def reset(self, **kwargs):
        self.reset_kwargs.append(dict(kwargs))
        if "seed" in kwargs:
            self.current_seed = kwargs["seed"]
        elif self.current_seed is None:
            self.current_seed = 9
        self.reset_seeds.append(self.current_seed)
        return (
            np.asarray([self.current_seed or -1], dtype=np.float32),
            {},
        )

    def step(self, action):
        reward = float(self.current_seed)
        info = {
            "success": float(self.current_seed % 2),
            "episode": {
                "return": reward,
                "length": 1,
            },
        }
        return (
            np.zeros(1, dtype=np.float32),
            reward,
            True,
            False,
            info,
        )


class ResetRecordingEnvironment:
    def __init__(self):
        self.reset_seeds = []
        self.reset_kwargs = []

    def reset(self, **kwargs):
        self.reset_kwargs.append(dict(kwargs))
        self.reset_seeds.append(kwargs.get("seed"))
        return np.zeros(1, dtype=np.float32), {}


class DeterministicEvaluationAgent:
    def sample_actions(self, observations, rng):
        del observations, rng
        return jnp.zeros(1, dtype=jnp.float32)


class ChunkEvaluationAgent:
    def sample_actions(self, observations, rng):
        del observations, rng
        return jnp.asarray([[0.1], [0.2]], dtype=jnp.float32)


class PrimitiveEvaluationEnvironment:
    def __init__(self, episode_length=2):
        self.episode_length = episode_length
        self.actions = []
        self.episode_actions = []
        self.current_episode_actions = None
        self.step_count = 0
        self.episode_return = 0.0

    def reset(self, **kwargs):
        del kwargs
        self.step_count = 0
        self.episode_return = 0.0
        self.current_episode_actions = []
        self.episode_actions.append(self.current_episode_actions)
        return np.asarray([0.0], dtype=np.float32), {}

    def step(self, action):
        action = np.asarray(action).copy()
        self.actions.append(action)
        self.current_episode_actions.append(action)
        self.step_count += 1
        reward = float(action.reshape(-1)[0])
        self.episode_return += reward
        done = self.step_count >= self.episode_length
        info = {
            "success": float(done),
            "episode": {
                "return": self.episode_return,
                "length": self.step_count,
            },
        }
        return (
            np.asarray([self.step_count], dtype=np.float32),
            reward,
            done,
            False,
            info,
        )


class PairedOnlineExperimentProtocolTest(unittest.TestCase):
    def write_tiny_checkpoint(self, checkpoint_path, *, step=6):
        saved_agent = TinyAgent(
            network=TinyNetwork(
                step=np.asarray(step),
                value=np.asarray([3.0, 4.0], dtype=np.float32),
            )
        )
        checkpoint_bytes = pickle.dumps(
            {"agent": flax.serialization.to_state_dict(saved_agent)}
        )
        with open(checkpoint_path, "wb") as checkpoint_file:
            checkpoint_file.write(checkpoint_bytes)
        return (
            saved_agent,
            hashlib.sha256(checkpoint_bytes).hexdigest(),
        )

    def test_phase_eval_interval_defaults_to_legacy_interval(self):
        self.assertEqual(
            resolve_phase_eval_interval(-1, legacy_interval=50000),
            50000,
        )

    def test_new_flags_default_to_legacy_behavior(self):
        expected_defaults = {
            "offline_agent_checkpoint": None,
            "offline_agent_checkpoint_sha256": None,
            "offline_eval_interval": -1,
            "online_eval_interval": -1,
            "online_env_seed_base": None,
            "eval_seed_base": None,
        }

        self.assertEqual(
            {
                name: FLAGS[name].default
                for name in expected_defaults
            },
            expected_defaults,
        )

    def test_phase_eval_intervals_can_be_configured_independently(self):
        offline_interval = resolve_phase_eval_interval(
            0, legacy_interval=50000
        )
        online_interval = resolve_phase_eval_interval(
            500, legacy_interval=50000
        )

        self.assertEqual(offline_interval, 0)
        self.assertEqual(online_interval, 500)

    def test_checkpoint_with_matching_sha_restores_complete_agent(self):
        empty_agent = TinyAgent(
            network=TinyNetwork(
                step=np.asarray(1),
                value=np.zeros(2, dtype=np.float32),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = f"{directory}/params_5.pkl"
            saved_agent, checkpoint_sha256 = (
                self.write_tiny_checkpoint(checkpoint_path)
            )

            restored_agent = restore_offline_agent_checkpoint(
                empty_agent,
                checkpoint_path=checkpoint_path,
                expected_sha256=checkpoint_sha256,
                offline_steps=5,
            )

        self.assertEqual(int(restored_agent.network.step), 6)
        np.testing.assert_array_equal(
            restored_agent.network.value,
            saved_agent.network.value,
        )

    def test_checkpoint_sha_mismatch_fails_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = f"{directory}/params_5.pkl"
            with open(checkpoint_path, "wb") as checkpoint_file:
                checkpoint_file.write(b"not a valid checkpoint")

            with self.assertRaisesRegex(
                ValueError, "SHA-256 mismatch"
            ):
                restore_offline_agent_checkpoint(
                    object(),
                    checkpoint_path=checkpoint_path,
                    expected_sha256="0" * 64,
                    offline_steps=5,
                )

    def test_checkpoint_path_and_sha_flags_must_be_provided_together(self):
        invalid_pairs = (
            ("/checkpoint.pkl", None),
            (None, "0" * 64),
        )
        for checkpoint_path, checkpoint_sha256 in invalid_pairs:
            with self.subTest(
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
            ):
                with self.assertRaises(ValueError):
                    validate_offline_agent_checkpoint_flags(
                        checkpoint_path,
                        checkpoint_sha256,
                    )

    def test_checkpoint_network_step_uses_offline_steps_plus_one(self):
        empty_agent = TinyAgent(
            network=TinyNetwork(
                step=np.asarray(1),
                value=np.zeros(2, dtype=np.float32),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = f"{directory}/params_5.pkl"
            _, checkpoint_sha256 = self.write_tiny_checkpoint(
                checkpoint_path,
                step=6,
            )

            with self.assertRaisesRegex(
                ValueError,
                (
                    "restored network step=6, offline_steps=6, "
                    "expected network step=7"
                ),
            ):
                restore_offline_agent_checkpoint(
                    empty_agent,
                    checkpoint_path=checkpoint_path,
                    expected_sha256=checkpoint_sha256,
                    offline_steps=6,
                )

    def test_external_checkpoint_skips_qam_offline_updates(self):
        offline_steps = 300000
        start_step = resolve_offline_training_start_step(
            offline_steps=offline_steps,
            external_checkpoint_loaded=True,
            load_stage=None,
            load_step=None,
        )

        self.assertEqual(start_step, offline_steps + 1)
        self.assertEqual(
            list(range(start_step, offline_steps + 1)),
            [],
        )

    def test_external_checkpoint_and_save_dir_resume_are_mutually_exclusive(self):
        for load_stage in ("offline", "online"):
            with self.subTest(load_stage=load_stage):
                with self.assertRaisesRegex(
                    ValueError, "cannot be combined"
                ):
                    resolve_offline_training_start_step(
                        offline_steps=300000,
                        external_checkpoint_loaded=True,
                        load_stage=load_stage,
                        load_step=100,
                    )

    def test_bridge_primitive_dataset_is_built_before_checkpoint_restore(self):
        source = (
            Path(__file__).resolve().parents[1] / "main.py"
        ).read_text()
        primitive_extraction = source.index(
            "bridge_primitive_transitions = extract_primitive_transitions"
        )
        agent_creation = source.index("agent = agent_class.create(")
        checkpoint_restore = source.index(
            "agent = restore_offline_agent_checkpoint(",
            agent_creation,
        )
        offline_loop = source.index("for i in tqdm.tqdm(range(start_step")

        self.assertLess(primitive_extraction, agent_creation)
        self.assertLess(agent_creation, checkpoint_restore)
        self.assertLess(checkpoint_restore, offline_loop)

    def test_main_wires_phase_intervals_and_paired_seeds_to_real_loops(self):
        source = (
            Path(__file__).resolve().parents[1] / "main.py"
        ).read_text()

        self.assertEqual(
            source.count("episode_seeds=evaluation_episode_seeds"),
            2,
        )
        self.assertIn("offline_eval_interval != 0", source)
        self.assertIn("i % offline_eval_interval == 0", source)
        self.assertIn("online_eval_interval != 0", source)
        self.assertIn("i % online_eval_interval == 0", source)
        self.assertEqual(
            source.count("reset_online_environment("),
            3,
        )
        self.assertIn(
            'env_info["online_episode_index"] = online_episode_index',
            source,
        )
        self.assertIn(
            'env_info["online_episode_seed"] = online_episode_seed',
            source,
        )

    def test_evaluation_episode_seeds_are_repeatable(self):
        episode_seeds = [30001, 30002, 30003]
        first_env = SeededEvaluationEnvironment()
        second_env = SeededEvaluationEnvironment()

        _, first_trajectories, _ = evaluate(
            agent=DeterministicEvaluationAgent(),
            env=first_env,
            num_eval_episodes=3,
            action_dim=1,
            episode_seeds=episode_seeds,
        )
        _, second_trajectories, _ = evaluate(
            agent=DeterministicEvaluationAgent(),
            env=second_env,
            num_eval_episodes=3,
            action_dim=1,
            episode_seeds=episode_seeds,
        )

        self.assertEqual(first_env.reset_seeds, episode_seeds)
        self.assertEqual(second_env.reset_seeds, episode_seeds)
        self.assertEqual(
            [trajectory["reward"] for trajectory in first_trajectories],
            [trajectory["reward"] for trajectory in second_trajectories],
        )

    def test_default_action_transform_keeps_evaluation_values_unchanged(self):
        implicit_environment = PrimitiveEvaluationEnvironment()
        explicit_environment = PrimitiveEvaluationEnvironment()

        implicit = evaluate(
            agent=ChunkEvaluationAgent(),
            env=implicit_environment,
            num_eval_episodes=1,
            action_dim=1,
        )
        explicit = evaluate(
            agent=ChunkEvaluationAgent(),
            env=explicit_environment,
            num_eval_episodes=1,
            action_dim=1,
            action_transform_factory=None,
        )

        self.assertEqual(implicit[0], explicit[0])
        self.assertEqual(tuple(implicit[1][0]), tuple(explicit[1][0]))
        for name in implicit[1][0]:
            implicit_values = implicit[1][0][name]
            explicit_values = explicit[1][0][name]
            self.assertEqual(len(implicit_values), len(explicit_values))
            for implicit_value, explicit_value in zip(
                implicit_values, explicit_values
            ):
                np.testing.assert_array_equal(
                    implicit_value, explicit_value
                )
        for implicit_action, explicit_action in zip(
            implicit_environment.actions,
            explicit_environment.actions,
        ):
            np.testing.assert_array_equal(
                implicit_action, explicit_action
            )
        self.assertFalse(
            any(
                name.startswith("evaluation_bridge_")
                for name in implicit[0]
            )
        )

    def test_action_transform_receives_popped_primitive_before_env_step(self):
        environment = PrimitiveEvaluationEnvironment()
        transformed_primitives = []

        def factory():
            def transform(observation, proposed_action):
                transformed_primitives.append(
                    (
                        np.asarray(observation).copy(),
                        np.asarray(proposed_action).copy(),
                    )
                )
                return (
                    np.asarray(proposed_action) + np.float32(0.3),
                    {"evaluation_bridge_applied_fraction": 1.0},
                )

            return transform

        stats, trajectories, _ = evaluate(
            agent=ChunkEvaluationAgent(),
            env=environment,
            num_eval_episodes=1,
            action_dim=1,
            action_transform_factory=factory,
        )

        np.testing.assert_allclose(
            [item[1] for item in transformed_primitives],
            [[0.1], [0.2]],
        )
        np.testing.assert_allclose(
            environment.actions,
            [[0.4], [0.5]],
        )
        np.testing.assert_allclose(
            trajectories[0]["action"],
            [[0.4], [0.5]],
        )
        self.assertEqual(
            stats["evaluation_bridge_applied_fraction"], 1.0
        )

    def test_each_episode_gets_independent_stateful_action_transform(self):
        environment = PrimitiveEvaluationEnvironment()
        factory_calls = []

        def factory():
            factory_calls.append(len(factory_calls))
            local_applied_steps = 0

            def transform(observation, proposed_action):
                del observation
                nonlocal local_applied_steps
                local_applied_steps += 1
                return (
                    np.asarray(proposed_action)
                    + np.float32(0.1 * local_applied_steps),
                    {
                        "evaluation_bridge_ready_fraction": 1.0,
                        "evaluation_bridge_applied_fraction": 1.0,
                        "evaluation_bridge_executed_residual_l2": (
                            0.1 * local_applied_steps
                        ),
                    },
                )

            return transform

        stats, _, _ = evaluate(
            agent=ChunkEvaluationAgent(),
            env=environment,
            num_eval_episodes=2,
            action_dim=1,
            action_transform_factory=factory,
        )

        self.assertEqual(factory_calls, [0, 1])
        for episode_actions in environment.episode_actions:
            np.testing.assert_allclose(
                episode_actions,
                [[0.2], [0.4]],
            )
        self.assertEqual(
            stats["evaluation_bridge_ready_fraction"], 1.0
        )
        self.assertAlmostEqual(
            stats["evaluation_bridge_executed_residual_l2"],
            0.15,
        )

    def test_main_only_builds_transform_factory_for_correction_runtime(self):
        shadow_runtime = SimpleNamespace(
            config=SimpleNamespace(apply_correction=False)
        )
        snapshots = []

        class Snapshot:
            def evaluate_action(self, observation, action):
                return action, {}

        class CorrectionRuntime:
            config = SimpleNamespace(apply_correction=True)

            def make_evaluation_snapshot(self):
                snapshot = Snapshot()
                snapshots.append(snapshot)
                return snapshot

        self.assertIsNone(
            build_bridge_evaluation_action_transform_factory(None)
        )
        self.assertIsNone(
            build_bridge_evaluation_action_transform_factory(
                shadow_runtime
            )
        )
        factory = build_bridge_evaluation_action_transform_factory(
            CorrectionRuntime()
        )
        first = factory()
        second = factory()
        self.assertIsNot(first.__self__, second.__self__)
        self.assertEqual(snapshots, [first.__self__, second.__self__])

    def test_bridge_evaluation_schema_only_appears_for_correction(self):
        shadow_info = {"success": np.float32(0.5)}
        returned_shadow_info = complete_bridge_evaluation_diagnostics(
            shadow_info,
            apply_correction=False,
        )
        self.assertIs(returned_shadow_info, shadow_info)
        self.assertEqual(tuple(shadow_info), ("success",))

        correction_info = {"success": np.float32(0.5)}
        complete_bridge_evaluation_diagnostics(
            correction_info,
            apply_correction=True,
        )
        diagnostic_names = [
            name
            for name in correction_info
            if name.startswith("evaluation_bridge_")
        ]
        self.assertEqual(len(diagnostic_names), 6)
        for name in diagnostic_names:
            self.assertEqual(float(correction_info[name]), 0.0)

    def test_evaluation_without_episode_seeds_keeps_unseeded_reset_call(self):
        environment = SeededEvaluationEnvironment()
        environment.current_seed = 9

        evaluate(
            agent=DeterministicEvaluationAgent(),
            env=environment,
            num_eval_episodes=1,
            action_dim=1,
        )

        self.assertEqual(environment.reset_kwargs, [{}])

    def test_evaluation_episode_seed_count_must_match_all_episodes(self):
        with self.assertRaisesRegex(
            ValueError, "episode_seeds length"
        ):
            evaluate(
                agent=DeterministicEvaluationAgent(),
                env=SeededEvaluationEnvironment(),
                num_eval_episodes=2,
                num_video_episodes=1,
                action_dim=1,
                episode_seeds=[30001, 30002],
            )

    def test_online_environment_seed_schedule_is_paired_by_episode(self):
        first_env = ResetRecordingEnvironment()
        second_env = ResetRecordingEnvironment()

        for episode_index in range(4):
            reset_online_environment(
                first_env,
                seed_base=40001,
                episode_index=episode_index,
            )
            reset_online_environment(
                second_env,
                seed_base=40001,
                episode_index=episode_index,
            )

        expected = [40001, 40002, 40003, 40004]
        self.assertEqual(first_env.reset_seeds, expected)
        self.assertEqual(second_env.reset_seeds, expected)

    def test_online_environment_without_seed_base_keeps_unseeded_reset(self):
        environment = ResetRecordingEnvironment()

        _, _, episode_seed = reset_online_environment(
            environment,
            seed_base=None,
            episode_index=0,
        )

        self.assertIsNone(episode_seed)
        self.assertEqual(environment.reset_kwargs, [{}])

    def test_evaluation_seed_schedule_is_reused_from_fixed_base(self):
        first = build_evaluation_episode_seeds(
            30001,
            num_eval_episodes=3,
            num_video_episodes=2,
        )
        second = build_evaluation_episode_seeds(
            30001,
            num_eval_episodes=3,
            num_video_episodes=2,
        )

        self.assertEqual(first, (30001, 30002, 30003, 30004, 30005))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
