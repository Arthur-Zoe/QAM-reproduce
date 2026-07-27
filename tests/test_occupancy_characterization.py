import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest


class OccupancyCharacterizationMatrixTest(unittest.TestCase):
    def test_local_protocol_disables_balanced_sampling(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        runs = build_characterization_runs(
            mode="local",
            run_group="characterization-test",
        )

        self.assertTrue(
            all(run["kwargs"]["balanced_sampling"] is False for run in runs)
        )

    def test_pilot_protocol_disables_balanced_sampling(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        runs = build_characterization_runs(
            mode="pilot",
            run_group="characterization-test",
        )

        self.assertTrue(
            all(run["kwargs"]["balanced_sampling"] is False for run in runs)
        )

    def test_local_start_training_disables_online_updates(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        for run in build_characterization_runs(
            mode="local",
            run_group="characterization-test",
        ):
            self.assertEqual(
                run["kwargs"]["start_training"],
                run["kwargs"]["online_steps"] + 1,
            )

    def test_pilot_start_training_disables_online_updates(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        for run in build_characterization_runs(
            mode="pilot",
            run_group="characterization-test",
        ):
            self.assertEqual(
                run["kwargs"]["start_training"],
                run["kwargs"]["online_steps"] + 1,
            )

    def test_local_behavior_policy_is_fixed_offline_agent(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
            infer_behavior_policy,
        )

        runs = build_characterization_runs(
            mode="local",
            run_group="characterization-test",
        )

        for run in runs:
            self.assertEqual(
                infer_behavior_policy(run["kwargs"]),
                "fixed_offline_agent",
            )
            self.assertEqual(
                run["behavior_policy"],
                "fixed_offline_agent",
            )
            self.assertEqual(run["agent_online_updates"], "disabled")

    def test_pilot_behavior_policy_is_fixed_offline_agent(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
            infer_behavior_policy,
        )

        runs = build_characterization_runs(
            mode="pilot",
            run_group="characterization-test",
        )

        self.assertTrue(
            all(
                infer_behavior_policy(run["kwargs"])
                == run["behavior_policy"]
                == "fixed_offline_agent"
                and run["agent_online_updates"] == "disabled"
                for run in runs
            )
        )

    def test_local_matrix_has_exactly_two_runs(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        runs = build_characterization_runs(
            mode="local",
            run_group="characterization-test",
        )

        self.assertEqual(len(runs), 2)

    def test_pilot_matrix_has_exactly_twenty_one_runs(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        runs = build_characterization_runs(
            mode="pilot",
            run_group="characterization-test",
        )

        self.assertEqual(len(runs), 21)

    def test_local_records_fix_conditions_seeds_and_training_protocol(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        runs = build_characterization_runs(
            mode="local",
            run_group="characterization-test",
        )

        self.assertEqual(
            [run["condition"] for run in runs],
            ["nominal", "gain_0p7"],
        )
        self.assertEqual({run["seed"] for run in runs}, {10001})
        for run in runs:
            kwargs = run["kwargs"]
            self.assertEqual(run["method"], "QAM_FQL")
            self.assertEqual(run["domain"], "cube-triple-play")
            self.assertEqual(run["task"], 2)
            self.assertEqual(
                kwargs["start_training"],
                kwargs["online_steps"] + 1,
            )
            self.assertEqual(
                kwargs["train_action_gain"],
                kwargs["eval_action_gain"],
            )
            self.assertEqual(
                kwargs["train_action_delay"],
                kwargs["eval_action_delay"],
            )

    def test_pilot_conditions_and_seeds_are_complete(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        runs = build_characterization_runs(
            mode="pilot",
            run_group="characterization-test",
        )

        self.assertEqual(
            {run["condition"] for run in runs},
            {
                "nominal",
                "gain_0p9",
                "gain_0p7",
                "gain_0p5",
                "delay_1",
                "delay_2",
                "delay_3",
            },
        )
        self.assertEqual(
            {run["seed"] for run in runs},
            {10001, 20002, 30003},
        )

    def test_condition_semantics_and_severity_are_fixed(self):
        from experiments.occupancy_characterization_matrix import (
            PILOT_CONDITIONS,
        )

        conditions = {
            condition.name: condition for condition in PILOT_CONDITIONS
        }
        nominal = conditions["nominal"]
        self.assertEqual(
            (
                nominal.condition_family,
                nominal.action_gain,
                nominal.action_delay,
                nominal.severity,
            ),
            ("nominal", 1.0, 0, 0.0),
        )
        self.assertAlmostEqual(conditions["gain_0p9"].severity, 0.1)
        self.assertAlmostEqual(conditions["gain_0p7"].severity, 0.3)
        self.assertAlmostEqual(conditions["gain_0p5"].severity, 0.5)
        self.assertEqual(conditions["delay_1"].severity, 1.0)
        self.assertEqual(conditions["delay_2"].severity, 2.0)
        self.assertEqual(conditions["delay_3"].severity, 3.0)

    def test_qam_fql_parameters_are_read_from_formal_matrix(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )
        from experiments.qam_matrix import AGENT_PARAMS

        run = build_characterization_runs(
            mode="local",
            run_group="characterization-test",
        )[0]
        expected = AGENT_PARAMS["QAM_FQL"]["cube-triple-play"]

        for name, value in expected.items():
            self.assertEqual(run["kwargs"][f"agent.{name}"], value)

    def test_building_characterization_does_not_modify_formal_matrix(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )
        from experiments.qam_matrix import AGENT_PARAMS

        before = copy.deepcopy(AGENT_PARAMS)
        build_characterization_runs(
            mode="local",
            run_group="characterization-test",
        )
        build_characterization_runs(
            mode="pilot",
            run_group="characterization-test",
        )

        self.assertEqual(AGENT_PARAMS, before)

    def test_condition_validation_rejects_invalid_semantics(self):
        from experiments.occupancy_characterization_matrix import (
            DynamicsCondition,
            validate_condition_definitions,
        )

        nominal = DynamicsCondition("nominal", "nominal", 1.0, 0)
        invalid_cases = (
            (
                nominal,
                DynamicsCondition("nominal", "gain", 0.7, 0),
            ),
            (DynamicsCondition("nominal", "nominal", 0.9, 0),),
            (
                nominal,
                DynamicsCondition("gain_bad", "gain", 0.7, 1),
            ),
            (
                nominal,
                DynamicsCondition("delay_bad", "delay", 0.9, 1),
            ),
            (
                nominal,
                DynamicsCondition("delay_bad", "delay", 1.0, -1),
            ),
            (
                nominal,
                DynamicsCondition("gain_bad", "gain", math.nan, 0),
            ),
            (
                nominal,
                DynamicsCondition("gain_bad", "gain", 0.0, 0),
            ),
        )
        for conditions in invalid_cases:
            with self.subTest(conditions=conditions):
                with self.assertRaises(ValueError):
                    validate_condition_definitions(conditions)

    def test_run_validation_rejects_duplicate_identity_and_wrong_counts(self):
        from experiments.occupancy_characterization_matrix import (
            validate_characterization_matrix,
            validate_run_records,
        )

        duplicate = (
            {"condition": "nominal", "seed": 10001},
            {"condition": "nominal", "seed": 10001},
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_run_records(duplicate)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            validate_run_records(duplicate[:1], expected_count=2)
        with self.assertRaisesRegex(ValueError, "local.*2"):
            validate_characterization_matrix(local_seeds=())
        with self.assertRaisesRegex(ValueError, "pilot.*21"):
            validate_characterization_matrix(pilot_seeds=(10001,))

    def test_run_validation_rejects_random_only_behavior_policy(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
            validate_run_records,
        )

        run = copy.deepcopy(
            build_characterization_runs(
                mode="local",
                run_group="characterization-test",
            )[0]
        )
        run["kwargs"]["balanced_sampling"] = True

        with self.assertRaisesRegex(
            ValueError,
            "fixed_offline_agent",
        ):
            validate_run_records((run,))

    def test_run_validation_rejects_online_agent_updates(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
            validate_run_records,
        )

        run = copy.deepcopy(
            build_characterization_runs(
                mode="local",
                run_group="characterization-test",
            )[0]
        )
        run["kwargs"]["start_training"] = run["kwargs"]["online_steps"]

        with self.assertRaisesRegex(
            ValueError,
            "agent_online_updates=disabled",
        ):
            validate_run_records((run,))


class OccupancyCharacterizationCommandTest(unittest.TestCase):
    def test_manifest_records_behavior_policy_metadata(self):
        from experiments.occupancy_characterization import (
            create_manifest,
        )
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        runs = build_characterization_runs(
            mode="local",
            run_group="characterization-test",
        )
        manifest = create_manifest(
            mode="local",
            run_group="characterization-test",
            runs=runs,
            generated_at="2026-01-01T00:00:00+00:00",
        )

        for run in manifest["runs"]:
            self.assertEqual(
                run["behavior_policy"],
                "fixed_offline_agent",
            )
            self.assertEqual(run["agent_online_updates"], "disabled")

    def test_local_shell_and_manifest_are_complete(self):
        from experiments.occupancy_characterization import (
            write_characterization_output,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run.sh"
            manifest_path = write_characterization_output(
                mode="local",
                output_format="shell",
                run_group="characterization-test",
                output_path=output,
            )
            script = output.read_text()
            manifest = json.loads(manifest_path.read_text())

        self.assertTrue(script.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -euo pipefail", script)
        self.assertIn("export MUJOCO_GL=egl", script)
        self.assertIn("export WANDB_MODE=disabled", script)
        self.assertIn(
            "export WANDB_PROJECT=qam-occupancy-characterization",
            script,
        )
        self.assertEqual(script.count("python3 -u main.py"), 2)
        self.assertNotIn("eval ", script)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["mode"], "local")
        self.assertEqual(manifest["run_count"], 2)
        self.assertEqual(
            {run["condition"] for run in manifest["runs"]},
            {"nominal", "gain_0p7"},
        )

    def test_local_shell_uses_fixed_offline_agent_actions(self):
        from experiments.occupancy_characterization import (
            render_shell_script,
        )
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        script = render_shell_script(
            build_characterization_runs(
                mode="local",
                run_group="characterization-test",
            ),
            "characterization-test",
        )

        self.assertEqual(script.count("--balanced_sampling=False"), 2)
        self.assertNotIn("--balanced_sampling=True", script)

    def test_local_shell_enables_deterministic_gpu_operations(self):
        from experiments.occupancy_characterization import (
            render_shell_script,
        )
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        script = render_shell_script(
            build_characterization_runs(
                mode="local",
                run_group="characterization-test",
            ),
            "characterization-test",
        )

        self.assertIn("--xla_gpu_deterministic_ops=true", script)
        self.assertIn(
            "--xla_gpu_exclude_nondeterministic_ops=true",
            script,
        )

    def test_shell_commands_include_complete_characterization_flags(self):
        from experiments.occupancy_characterization import (
            render_run_command,
        )
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        command = render_run_command(
            build_characterization_runs(
                mode="local",
                run_group="characterization-test",
            )[0]
        )

        required_fragments = (
            "--offline_steps=5000",
            "--online_steps=5000",
            "--start_training=5001",
            "--recent_dynamics_capacity=1000",
            "--occupancy_detector=True",
            "--occupancy_batch_size=128",
            "--occupancy_start_size=500",
            "--occupancy_update_interval=500",
            "--occupancy_updates_per_interval=5",
            "--agent.inv_temp=10.0",
            "--agent.fql_alpha=300.0",
            "--agent.edit_scale=0.0",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, command)

    def test_shell_quoting_handles_metacharacters_without_eval(self):
        from experiments.occupancy_characterization import (
            render_shell_script,
        )
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        run_group = "group with spaces;echo unsafe"
        script = render_shell_script(
            build_characterization_runs("local", run_group),
            run_group,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run.sh"
            path.write_text(script)
            result = subprocess.run(
                ["bash", "-n", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("\neval ", script)
        self.assertIn("'logs/group with spaces;echo unsafe'", script)
        self.assertIn(
            "'--run_group=group with spaces;echo unsafe'",
            script,
        )

    def test_local_runs_only_differ_in_condition_fields(self):
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        nominal, shifted = build_characterization_runs(
            "local", "characterization-test"
        )
        excluded = {
            "tags",
            "train_action_gain",
            "train_action_delay",
            "eval_action_gain",
            "eval_action_delay",
        }
        nominal_common = {
            key: value
            for key, value in nominal["kwargs"].items()
            if key not in excluded
        }
        shifted_common = {
            key: value
            for key, value in shifted["kwargs"].items()
            if key not in excluded
        }
        self.assertEqual(nominal_common, shifted_common)

    def test_pilot_sbatch_contains_twenty_one_one_gpu_commands(self):
        from experiments.occupancy_characterization import (
            write_characterization_output,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "pilot.sbatch"
            manifest_path = write_characterization_output(
                mode="pilot",
                output_format="sbatch",
                run_group="characterization-pilot",
                output_path=output,
            )
            script = output.read_text()
            manifest = json.loads(manifest_path.read_text())

        self.assertIn("#SBATCH --gres=gpu:1", script)
        self.assertIn("#SBATCH --array=0-20%21", script)
        self.assertIn("export MUJOCO_GL=egl", script)
        self.assertEqual(script.count("python3 -u main.py"), 21)
        self.assertEqual(manifest["run_count"], 21)
        identities = {
            (run["condition"], run["seed"]) for run in manifest["runs"]
        }
        self.assertEqual(len(identities), 21)
        for run in manifest["runs"]:
            self.assertEqual(
                run["kwargs"]["online_save_interval"], 10000
            )
            self.assertTrue(run["kwargs"]["occupancy_detector"])

    def test_every_pilot_command_uses_fixed_offline_agent_actions(self):
        from experiments.occupancy_characterization import (
            render_sbatch_script,
        )
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        script = render_sbatch_script(
            build_characterization_runs(
                mode="pilot",
                run_group="characterization-pilot",
            ),
            "characterization-pilot",
        )

        self.assertEqual(script.count("--balanced_sampling=False"), 21)
        self.assertNotIn("--balanced_sampling=True", script)

    def test_pilot_sbatch_enables_deterministic_gpu_operations(self):
        from experiments.occupancy_characterization import (
            render_sbatch_script,
        )
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        script = render_sbatch_script(
            build_characterization_runs(
                mode="pilot",
                run_group="characterization-pilot",
            ),
            "characterization-pilot",
        )

        self.assertIn("--xla_gpu_deterministic_ops=true", script)
        self.assertIn(
            "--xla_gpu_exclude_nondeterministic_ops=true",
            script,
        )

    def test_json_output_is_the_complete_manifest(self):
        from experiments.occupancy_characterization import (
            write_characterization_output,
        )
        from experiments.occupancy_characterization_matrix import (
            BASE_COMMIT,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "manifest.json"
            returned = write_characterization_output(
                mode="local",
                output_format="json",
                run_group="characterization-test",
                output_path=output,
            )
            manifest = json.loads(output.read_text())

        self.assertEqual(returned, output)
        self.assertEqual(manifest["base_commit"], BASE_COMMIT)
        self.assertIn("generated_at", manifest)
        self.assertEqual(len(manifest["runs"][0]["kwargs"]), 43)


class OccupancyCharacterizationSummaryTest(unittest.TestCase):
    def write_valid_fixture(self, root, run_group="summary-test"):
        from experiments.occupancy_characterization import create_manifest
        from experiments.occupancy_characterization_matrix import (
            build_characterization_runs,
        )

        root = Path(root)
        exp_root = root / "exp"
        log_dir = root / "logs" / run_group
        output_dir = log_dir / "summary"
        log_dir.mkdir(parents=True)
        runs = build_characterization_runs("local", run_group)
        manifest = create_manifest(
            "local",
            run_group,
            runs,
            generated_at="2026-01-01T00:00:00+00:00",
        )
        manifest_path = log_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        for run in runs:
            run_dir = (
                exp_root
                / "dummy"
                / run_group
                / run["env_name"]
                / f"{run['condition']}-seed{run['seed']}"
            )
            run_dir.mkdir(parents=True)
            flags = {}
            agent_flags = {"agent_name": "qam"}
            for key, value in run["kwargs"].items():
                if key.startswith("agent."):
                    agent_flags[key.split(".", 1)[1]] = value
                elif key == "agent":
                    continue
                elif key == "save_dir":
                    flags[key] = str(run_dir)
                else:
                    flags[key] = value
            flags["agent"] = agent_flags
            (run_dir / "flags.json").write_text(json.dumps(flags))
            (run_dir / "token.tk").write_text("")

            with (run_dir / "offline_agent.csv").open(
                "w", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=("actor/loss", "critic/loss", "step"),
                )
                writer.writeheader()
                for index, step in enumerate(range(1000, 5001, 1000)):
                    writer.writerow(
                        {
                            "actor/loss": 1.0 + index,
                            "critic/loss": 2.0 + index,
                            "step": step,
                        }
                    )

            condition_offset = (
                0.0 if run["condition"] == "nominal" else 0.1
            )
            with (run_dir / "occupancy_detector.csv").open(
                "w", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=(
                        "eval/balanced_accuracy",
                        "eval/logit_gap",
                        "eval/offline_probability_mean",
                        "eval/online_probability_mean",
                        "step",
                    ),
                )
                writer.writeheader()
                for index, online_step in enumerate(
                    range(500, 5001, 500)
                ):
                    writer.writerow(
                        {
                            "eval/balanced_accuracy": (
                                0.5 + condition_offset + 0.01 * index
                            ),
                            "eval/logit_gap": (
                                0.1 + condition_offset + 0.02 * index
                            ),
                            "eval/offline_probability_mean": (
                                0.4 - condition_offset / 2
                            ),
                            "eval/online_probability_mean": (
                                0.6 + condition_offset / 2
                            ),
                            "step": 5000 + online_step,
                        }
                    )

            with (run_dir / "eval.csv").open(
                "w", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=("success", "episode.return", "step"),
                )
                writer.writeheader()
                for step in range(1000, 5001, 1000):
                    writer.writerow(
                        {
                            "success": 0.99,
                            "episode.return": 99.0,
                            "step": step,
                        }
                    )
                for index, online_step in enumerate(
                    range(1000, 5001, 1000)
                ):
                    writer.writerow(
                        {
                            "success": (
                                0.2 - condition_offset + 0.05 * index
                            ),
                            "episode.return": (
                                20.0 - 10 * condition_offset + index
                            ),
                            "step": 5000 + online_step,
                        }
                    )
        return exp_root, manifest_path, output_dir

    def read_csv_rows(self, path):
        with Path(path).open(newline="") as file:
            return list(csv.DictReader(file))

    def run_directory(self, exp_root, condition):
        matches = list(Path(exp_root).rglob(f"{condition}-seed10001"))
        self.assertEqual(len(matches), 1)
        return matches[0]

    def rewrite_online_performance(self, exp_root, **values):
        for eval_path in Path(exp_root).rglob("eval.csv"):
            rows = self.read_csv_rows(eval_path)
            for row in rows:
                if int(float(row["step"])) > 5000:
                    for key, value in values.items():
                        row[key] = str(value)
            with eval_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

    def test_valid_runs_match_flags_steps_and_write_all_outputs(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            validation = summarize_characterization(
                exp_root=exp_root,
                run_group="summary-test",
                output_dir=output_dir,
                manifest_path=manifest_path,
            )
            output_names = {
                path.name for path in output_dir.iterdir()
            }

        self.assertEqual(validation["run_count"], 2)
        self.assertEqual(validation["completed_run_count"], 2)
        self.assertEqual(validation["performance_key"], "success")
        self.assertEqual(
            output_names,
            {
                "run_summary.csv",
                "aggregate_summary.csv",
                "paired_deltas.csv",
                "report.md",
                "validation.json",
            },
        )

    def test_validation_records_fixed_behavior_policy(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )

        self.assertEqual(
            validation["behavior_policy"],
            "fixed_offline_agent",
        )
        self.assertEqual(
            validation["agent_online_updates"],
            "disabled",
        )
        self.assertTrue(
            validation["offline_training_trajectory_consistent"]
        )
        self.assertEqual(
            validation[
                "offline_training_trajectory_max_abs_difference"
            ],
            0.0,
        )

    def test_report_states_transitions_use_fixed_offline_agent(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )
            report = (output_dir / "report.md").read_text()

        self.assertIn(
            "Online transitions 由固定离线 Agent 采集",
            report,
        )
        self.assertIn("QAM online update 被禁用", report)
        self.assertIn(
            "same-seed offline training trajectory consistency check",
            report,
        )

    def test_offline_training_trajectory_mismatch_is_rejected(self):
        from experiments.summarize_occupancy_characterization import (
            CharacterizationError,
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            path = (
                self.run_directory(exp_root, "gain_0p7")
                / "offline_agent.csv"
            )
            rows = self.read_csv_rows(path)
            rows[0]["actor/loss"] = "99"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(
                CharacterizationError,
                "offline training trajectory mismatch.*step 1000.*actor/loss",
            ):
                summarize_characterization(
                    exp_root,
                    "summary-test",
                    output_dir,
                    manifest_path=manifest_path,
                )

    def test_global_steps_are_converted_to_expected_online_steps(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )
            rows = self.read_csv_rows(output_dir / "run_summary.csv")

        self.assertEqual(
            rows[0]["occupancy_online_steps"],
            "500;1000;1500;2000;2500;3000;3500;4000;4500;5000",
        )

    def test_duplicate_occupancy_step_is_rejected(self):
        from experiments.summarize_occupancy_characterization import (
            CharacterizationError,
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            path = (
                self.run_directory(exp_root, "nominal")
                / "occupancy_detector.csv"
            )
            rows = self.read_csv_rows(path)
            rows[-1]["step"] = rows[-2]["step"]
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(
                CharacterizationError, "duplicate occupancy step"
            ):
                summarize_characterization(
                    exp_root,
                    "summary-test",
                    output_dir,
                    manifest_path=manifest_path,
                )

    def test_non_finite_metric_is_rejected(self):
        from experiments.summarize_occupancy_characterization import (
            CharacterizationError,
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            path = (
                self.run_directory(exp_root, "gain_0p7")
                / "occupancy_detector.csv"
            )
            text = path.read_text()
            path.write_text(text.replace("0.6,", "nan,", 1))

            with self.assertRaisesRegex(
                CharacterizationError, "non-finite"
            ):
                summarize_characterization(
                    exp_root,
                    "summary-test",
                    output_dir,
                    manifest_path=manifest_path,
                )

    def test_missing_token_and_unfinished_progress_are_rejected(self):
        from experiments.summarize_occupancy_characterization import (
            CharacterizationError,
            summarize_characterization,
        )

        with self.subTest("missing token"):
            with tempfile.TemporaryDirectory() as temporary_directory:
                exp_root, manifest_path, output_dir = (
                    self.write_valid_fixture(temporary_directory)
                )
                (
                    self.run_directory(exp_root, "nominal") / "token.tk"
                ).unlink()
                with self.assertRaisesRegex(
                    CharacterizationError, "token.tk"
                ):
                    summarize_characterization(
                        exp_root,
                        "summary-test",
                        output_dir,
                        manifest_path=manifest_path,
                    )
        with self.subTest("unfinished progress"):
            with tempfile.TemporaryDirectory() as temporary_directory:
                exp_root, manifest_path, output_dir = (
                    self.write_valid_fixture(temporary_directory)
                )
                (
                    self.run_directory(exp_root, "nominal")
                    / "progress.tk"
                ).write_text("online,1000")
                with self.assertRaisesRegex(
                    CharacterizationError, "unfinished progress"
                ):
                    summarize_characterization(
                        exp_root,
                        "summary-test",
                        output_dir,
                        manifest_path=manifest_path,
                    )

    def test_last_five_detector_mean_uses_exact_window(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )
            rows = self.read_csv_rows(output_dir / "run_summary.csv")
            nominal = next(
                row for row in rows if row["condition"] == "nominal"
            )

        self.assertEqual(int(nominal["last_5_window_size"]), 5)
        self.assertAlmostEqual(
            float(nominal["mean_last_5_eval_balanced_accuracy"]),
            0.57,
        )
        self.assertAlmostEqual(
            float(nominal["mean_last_5_eval_logit_gap"]),
            0.24,
        )

    def test_requested_performance_key_overrides_priority(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                performance_key="episode.return",
                manifest_path=manifest_path,
            )

        self.assertEqual(validation["performance_key"], "episode.return")

    def test_automatic_performance_selection_prefers_informative_candidate(
        self,
    ):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            self.rewrite_online_performance(exp_root, success=0)

            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )

        self.assertEqual(validation["performance_key"], "episode.return")
        self.assertTrue(validation["performance_key_informative"])
        self.assertEqual(
            validation["performance_key_reason"],
            "informative",
        )

    def test_all_constant_candidates_fall_back_to_highest_priority(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            self.rewrite_online_performance(
                exp_root,
                success=0,
                **{"episode.return": -3000},
            )
            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )

        self.assertEqual(validation["performance_key"], "success")
        self.assertEqual(
            validation["performance_key_source"],
            "automatic",
        )

    def test_constant_selected_key_is_marked_uninformative(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            self.rewrite_online_performance(
                exp_root,
                success=0,
                **{"episode.return": -3000},
            )
            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )

        self.assertFalse(validation["performance_key_informative"])
        self.assertEqual(validation["performance_key_reason"], "constant")

    def test_explicit_constant_performance_key_is_allowed(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            self.rewrite_online_performance(
                exp_root,
                **{"episode.return": -3000},
            )
            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                performance_key="episode.return",
                manifest_path=manifest_path,
            )

        self.assertEqual(validation["performance_key"], "episode.return")
        self.assertEqual(validation["performance_key_source"], "explicit")
        self.assertFalse(validation["performance_key_informative"])
        self.assertEqual(validation["performance_key_reason"], "constant")

    def test_non_finite_candidate_is_not_informative(self):
        from experiments.summarize_occupancy_characterization import (
            performance_candidate_statistics,
        )

        candidate = performance_candidate_statistics(
            [{"success": 0.0}, {"success": math.nan}],
            fieldnames=("success", "step"),
        )["success"]

        self.assertEqual(candidate["count"], 2)
        self.assertEqual(candidate["finite_count"], 1)
        self.assertFalse(candidate["informative"])
        self.assertEqual(candidate["reason"], "non_finite")

    def test_single_finite_sample_is_insufficient(self):
        from experiments.summarize_occupancy_characterization import (
            performance_candidate_statistics,
        )

        candidate = performance_candidate_statistics(
            [{"success": 1.0}],
            fieldnames=("success",),
        )["success"]

        self.assertEqual(candidate["finite_count"], 1)
        self.assertFalse(candidate["informative"])
        self.assertEqual(candidate["reason"], "insufficient_samples")

    def test_step_and_index_columns_are_not_performance_candidates(self):
        from experiments.summarize_occupancy_characterization import (
            performance_candidate_statistics,
        )

        candidates = performance_candidate_statistics(
            [
                {
                    "success": 0.0,
                    "step": 6000,
                    "global_step": 6000,
                    "row_index": 0,
                },
                {
                    "success": 1.0,
                    "step": 7000,
                    "global_step": 7000,
                    "row_index": 1,
                },
            ]
        )

        self.assertEqual(set(candidates), {"success"})

    def test_validation_includes_all_candidate_statistics(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )
            persisted = json.loads(
                (output_dir / "validation.json").read_text()
            )

        self.assertEqual(
            set(validation["performance_candidates"]),
            {"success", "episode.return"},
        )
        self.assertEqual(
            persisted["performance_candidates"],
            validation["performance_candidates"],
        )
        for statistics_by_key in persisted[
            "performance_candidates"
        ].values():
            self.assertEqual(
                set(statistics_by_key),
                {
                    "count",
                    "finite_count",
                    "minimum",
                    "maximum",
                    "mean",
                    "std",
                    "unique_count",
                    "is_constant",
                    "informative",
                    "reason",
                },
            )

    def test_report_explicitly_discloses_constant_performance(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            self.rewrite_online_performance(
                exp_root,
                success=0,
                **{"episode.return": -3000},
            )
            summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )
            report = (output_dir / "report.md").read_text()

        self.assertIn("| `success` | 10 | 10 | 0", report)
        self.assertIn("| `episode.return` | 10 | 10 | -3000", report)
        self.assertIn("constant", report)
        self.assertIn(
            "不能用于 detector score 与策略性能关系分析",
            report,
        )

    def test_validation_reports_separate_readiness_fields(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )

        self.assertIn("detector_pilot_ready", validation)
        self.assertIn("performance_correlation_ready", validation)
        self.assertIsInstance(validation["detector_pilot_ready"], bool)
        self.assertIsInstance(
            validation["performance_correlation_ready"],
            bool,
        )

    def test_constant_fixture_is_detector_but_not_performance_ready(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            self.rewrite_online_performance(
                exp_root,
                success=0,
                **{"episode.return": -3000},
            )
            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )

        self.assertTrue(validation["detector_pilot_ready"])
        self.assertFalse(validation["performance_correlation_ready"])

    def test_nonconstant_fixture_is_performance_correlation_ready(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )

        self.assertTrue(validation["detector_pilot_ready"])
        self.assertTrue(validation["performance_correlation_ready"])

    def test_performance_auto_selection_failure_lists_numeric_columns(self):
        from experiments.summarize_occupancy_characterization import (
            CharacterizationError,
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            for eval_path in Path(exp_root).rglob("eval.csv"):
                rows = self.read_csv_rows(eval_path)
                with eval_path.open("w", newline="") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=("metric_a", "metric_b", "step"),
                    )
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(
                            {
                                "metric_a": row["success"],
                                "metric_b": row["episode.return"],
                                "step": row["step"],
                            }
                        )
            with self.assertRaisesRegex(
                CharacterizationError,
                "numeric columns: metric_a, metric_b",
            ):
                summarize_characterization(
                    exp_root,
                    "summary-test",
                    output_dir,
                    manifest_path=manifest_path,
                )

    def test_local_paired_deltas_are_shift_minus_nominal(self):
        from experiments.summarize_occupancy_characterization import (
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )
            rows = self.read_csv_rows(output_dir / "paired_deltas.csv")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["condition"], "gain_0p7")
        self.assertAlmostEqual(
            float(rows[0]["delta_final_eval_balanced_accuracy"]),
            0.1,
        )
        self.assertAlmostEqual(
            float(rows[0]["delta_mean_last_5_eval_logit_gap"]),
            0.1,
        )
        self.assertAlmostEqual(
            float(rows[0]["delta_final_online_performance"]),
            -0.1,
        )

    def test_aggregate_mean_standard_deviation_and_n(self):
        from experiments.summarize_occupancy_characterization import (
            aggregate_run_summaries,
        )

        rows = [
            {
                "condition": "gain_0p7",
                "condition_family": "gain",
                "severity": 0.3,
                **{metric: 1.0 for metric in (
                    "final_eval_balanced_accuracy",
                    "final_eval_logit_gap",
                    "final_eval_offline_probability_mean",
                    "final_eval_online_probability_mean",
                    "mean_last_5_eval_balanced_accuracy",
                    "mean_last_5_eval_logit_gap",
                    "mean_last_5_eval_offline_probability_mean",
                    "mean_last_5_eval_online_probability_mean",
                    "final_online_performance",
                    "mean_online_performance",
                    "minimum_online_performance",
                )},
            },
            {
                "condition": "gain_0p7",
                "condition_family": "gain",
                "severity": 0.3,
                **{metric: 3.0 for metric in (
                    "final_eval_balanced_accuracy",
                    "final_eval_logit_gap",
                    "final_eval_offline_probability_mean",
                    "final_eval_online_probability_mean",
                    "mean_last_5_eval_balanced_accuracy",
                    "mean_last_5_eval_logit_gap",
                    "mean_last_5_eval_offline_probability_mean",
                    "mean_last_5_eval_online_probability_mean",
                    "final_online_performance",
                    "mean_online_performance",
                    "minimum_online_performance",
                )},
            },
        ]

        aggregate = aggregate_run_summaries(rows)[0]

        self.assertEqual(aggregate["n"], 2)
        self.assertEqual(
            aggregate["final_eval_balanced_accuracy_mean"], 2.0
        )
        self.assertEqual(
            aggregate["final_eval_balanced_accuracy_std"], 1.0
        )

    def test_spearman_handles_order_and_ties(self):
        from experiments.summarize_occupancy_characterization import (
            average_ranks,
            spearman_rank_correlation,
        )

        self.assertEqual(average_ranks([1.0, 2.0, 2.0, 4.0]), [1, 2.5, 2.5, 4])
        self.assertAlmostEqual(
            spearman_rank_correlation(
                [0.1, 0.3, 0.5],
                [10.0, 20.0, 30.0],
            ),
            1.0,
        )
        self.assertAlmostEqual(
            spearman_rank_correlation(
                [0.1, 0.3, 0.5],
                [30.0, 20.0, 10.0],
            ),
            -1.0,
        )

    def test_spearman_is_unavailable_with_fewer_than_three_points(self):
        from experiments.summarize_occupancy_characterization import (
            spearman_rank_correlation,
            summarize_characterization,
        )

        self.assertIsNone(
            spearman_rank_correlation([0.1, 0.3], [1.0, 2.0])
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            validation = summarize_characterization(
                exp_root,
                "summary-test",
                output_dir,
                manifest_path=manifest_path,
            )
        self.assertEqual(
            validation["correlations"]["gain"][
                "final_eval_balanced_accuracy"
            ],
            "unavailable",
        )

    def test_online_agent_data_rows_are_rejected(self):
        from experiments.summarize_occupancy_characterization import (
            CharacterizationError,
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            path = (
                self.run_directory(exp_root, "nominal")
                / "online_agent.csv"
            )
            path.write_text("loss,step\n1.0,6000\n")
            with self.assertRaisesRegex(
                CharacterizationError, "online_agent.csv"
            ):
                summarize_characterization(
                    exp_root,
                    "summary-test",
                    output_dir,
                    manifest_path=manifest_path,
                )

    def test_old_random_action_run_is_rejected_explicitly(self):
        from experiments.summarize_occupancy_characterization import (
            CharacterizationError,
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            manifest = json.loads(manifest_path.read_text())
            for run in manifest["runs"]:
                run["kwargs"]["balanced_sampling"] = True
                run.pop("behavior_policy", None)
                run.pop("agent_online_updates", None)
            manifest_path.write_text(json.dumps(manifest))
            for flags_path in Path(exp_root).rglob("flags.json"):
                flags = json.loads(flags_path.read_text())
                flags["balanced_sampling"] = True
                flags_path.write_text(json.dumps(flags))

            with self.assertRaisesRegex(
                CharacterizationError,
                "balanced_sampling=True caused random actions",
            ):
                summarize_characterization(
                    exp_root,
                    "summary-test",
                    output_dir,
                    manifest_path=manifest_path,
                )

    def test_summarizer_requires_balanced_sampling_false_in_flags(self):
        from experiments.summarize_occupancy_characterization import (
            CharacterizationError,
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            flags_path = next(Path(exp_root).rglob("flags.json"))
            flags = json.loads(flags_path.read_text())
            flags["balanced_sampling"] = True
            flags_path.write_text(json.dumps(flags))

            with self.assertRaisesRegex(
                CharacterizationError,
                "balanced_sampling.*mismatch",
            ):
                summarize_characterization(
                    exp_root,
                    "summary-test",
                    output_dir,
                    manifest_path=manifest_path,
                )

    def test_non_target_flag_mismatch_between_conditions_is_rejected(self):
        from experiments.summarize_occupancy_characterization import (
            CharacterizationError,
            summarize_characterization,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            exp_root, manifest_path, output_dir = (
                self.write_valid_fixture(temporary_directory)
            )
            manifest = json.loads(manifest_path.read_text())
            shifted = next(
                run
                for run in manifest["runs"]
                if run["condition"] == "gain_0p7"
            )
            shifted["kwargs"]["log_interval"] = 999
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                CharacterizationError, "non-target flag mismatch"
            ):
                summarize_characterization(
                    exp_root,
                    "summary-test",
                    output_dir,
                    manifest_path=manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
