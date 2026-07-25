"""Generate QAM action-dynamics-shift baseline commands without submitting them."""

import argparse
from pathlib import Path

try:
    from .generate import SbatchGenerator
    from .qam_matrix import AGENT_PARAMS, base_kwargs
except ImportError:
    from generate import SbatchGenerator
    from qam_matrix import AGENT_PARAMS, base_kwargs


ENV_NAME = "cube-triple-play-singletask-task2-v0"
DOMAIN = "cube-triple-play"
SEEDS = (10001, 20002, 30003)
CONDITIONS = {
    "nominal": {
        "train_action_gain": 1.0,
        "train_action_delay": 0,
        "eval_action_gain": 1.0,
        "eval_action_delay": 0,
    },
    "gain-0.8": {
        "train_action_gain": 0.8,
        "train_action_delay": 0,
        "eval_action_gain": 0.8,
        "eval_action_delay": 0,
    },
    "gain-1.2": {
        "train_action_gain": 1.2,
        "train_action_delay": 0,
        "eval_action_gain": 1.2,
        "eval_action_delay": 0,
    },
    "delay-1": {
        "train_action_gain": 1.0,
        "train_action_delay": 1,
        "eval_action_gain": 1.0,
        "eval_action_delay": 1,
    },
}


def generate(debug=False):
    generator = SbatchGenerator(
        j=1,
        limit=100,
        prefix=("MUJOCO_GL=egl", "python main.py"),
        comment="dynamics-shift-baseline",
    )
    if debug:
        generator.add_common_prefix(
            {
                "offline_steps": 100,
                "online_steps": 100,
                "start_training": 50,
                "eval_episodes": 1,
                "eval_interval": 50,
                "log_interval": 25,
            }
        )
        seeds = SEEDS[:1]
    else:
        generator.add_common_prefix(
            {
                "offline_steps": 1000000,
                "online_steps": 500000,
                "save_interval": 50000,
                "eval_interval": 50000,
            }
        )
        seeds = SEEDS

    for condition, shift_kwargs in CONDITIONS.items():
        for seed in seeds:
            kwargs = {
                "run_group": f"dynamics-shift-baseline-{condition}",
                "agent": "agents/qam.py",
                "tags": f"QAM,dynamics-shift,{condition}",
                **base_kwargs(seed, ENV_NAME, DOMAIN, data_root=""),
                **shift_kwargs,
            }
            for key, value in AGENT_PARAMS["QAM"][DOMAIN].items():
                kwargs[f"agent.{key}"] = value
            generator.add_run(kwargs)

    expected_count = len(CONDITIONS) * len(seeds)
    actual_count = len(generator.commands)
    if actual_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} commands, generated {actual_count}."
        )
    return generator, expected_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate short smoke-test runs for the first seed only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not write Slurm scripts.",
    )
    parser.add_argument(
        "--output-dir",
        default="sbatch",
        help="Directory for generated Slurm scripts outside dry-run mode.",
    )
    args = parser.parse_args()

    generator, expected_count = generate(debug=args.debug)
    print(f"Expected command count: {expected_count}")
    print(f"Actual command count: {len(generator.commands)}")

    if args.dry_run:
        print("\n".join(generator.commands))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-debug" if args.debug else ""
    for index, sbatch_str in enumerate(generator.generate_str(), start=1):
        output_path = output_dir / f"dynamics-shift-baseline{suffix}-part{index}.sh"
        output_path.write_text(sbatch_str)
        print(f"Wrote {output_path}")
    print("No Slurm jobs were submitted.")


if __name__ == "__main__":
    main()
