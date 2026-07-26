"""Generate auditable local or pilot occupancy-characterization commands."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex

try:
    from .occupancy_characterization_matrix import (
        BASE_COMMIT,
        build_characterization_runs,
    )
except ImportError:
    from occupancy_characterization_matrix import (
        BASE_COMMIT,
        build_characterization_runs,
    )


SCHEMA_VERSION = 1
SBATCH_RUN_GROUP_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
DETERMINISTIC_XLA_FLAGS = (
    "--xla_gpu_deterministic_ops=true "
    "--xla_gpu_exclude_nondeterministic_ops=true"
)


def _json_run(run):
    return {
        "condition": run["condition"],
        "condition_family": run["condition_family"],
        "severity": run["severity"],
        "seed": run["seed"],
        "method": run["method"],
        "domain": run["domain"],
        "task": run["task"],
        "env_name": run["env_name"],
        "run_group": run["run_group"],
        "tags": run["tags"],
        "behavior_policy": run["behavior_policy"],
        "agent_online_updates": run["agent_online_updates"],
        "kwargs": dict(run["kwargs"]),
    }


def create_manifest(mode, run_group, runs, generated_at=None):
    """Return a JSON-serializable audit manifest."""
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "run_group": run_group,
        "base_commit": BASE_COMMIT,
        "generated_at": generated_at,
        "run_count": len(runs),
        "runs": [_json_run(run) for run in runs],
    }


def render_run_command(run):
    """Render one main.py command with one safely quoted argv token per flag."""
    argv = ["python3", "-u", "main.py"]
    for key, value in run["kwargs"].items():
        argv.append(f"--{key}={value}")
    return shlex.join(argv)


def render_shell_script(runs, run_group):
    """Render a fail-fast sequential local runner."""
    log_directory = f"logs/{run_group}"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "export MUJOCO_GL=egl",
        "export WANDB_MODE=disabled",
        "export WANDB_PROJECT=qam-occupancy-characterization",
        "export PYTHONUNBUFFERED=1",
        (
            'export XLA_FLAGS="${XLA_FLAGS:-} '
            f'{DETERMINISTIC_XLA_FLAGS}"'
        ),
        "",
        f"LOG_DIR={shlex.quote(log_directory)}",
        'mkdir -p "${LOG_DIR}"',
        'test -f "${LOG_DIR}/manifest.json"',
        "",
    ]
    for run in runs:
        identity = f"{run['condition']}-seed{run['seed']}"
        lines.extend(
            [
                (
                    "echo "
                    + shlex.quote(
                        "START "
                        f"condition={run['condition']} seed={run['seed']}"
                    )
                ),
                (
                    render_run_command(run)
                    + f' >"${{LOG_DIR}}/{identity}.log" 2>&1'
                ),
                (
                    "echo "
                    + shlex.quote(
                        "END "
                        f"condition={run['condition']} seed={run['seed']}"
                    )
                ),
                "",
            ]
        )
    return "\n".join(lines)


def render_sbatch_script(runs, run_group):
    """Render one-GPU-per-run Slurm array commands without submitting them."""
    if not SBATCH_RUN_GROUP_PATTERN.fullmatch(run_group):
        raise ValueError(
            "sbatch run_group must contain only letters, digits, '.', '_', "
            "or '-'."
        )
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={run_group}",
        f"#SBATCH --array=0-{len(runs) - 1}%{len(runs)}",
        "#SBATCH --gres=gpu:1",
        f"#SBATCH --output=logs/{run_group}/slurm-%A_%a.out",
        "set -euo pipefail",
        "",
        "export MUJOCO_GL=egl",
        "export WANDB_PROJECT=qam-occupancy-characterization",
        "export PYTHONUNBUFFERED=1",
        (
            'export XLA_FLAGS="${XLA_FLAGS:-} '
            f'{DETERMINISTIC_XLA_FLAGS}"'
        ),
        "",
        'case "${SLURM_ARRAY_TASK_ID}" in',
    ]
    for index, run in enumerate(runs):
        lines.extend(
            [
                f"  {index})",
                (
                    "    echo "
                    + shlex.quote(
                        "START "
                        f"condition={run['condition']} seed={run['seed']}"
                    )
                ),
                f"    {render_run_command(run)}",
                (
                    "    echo "
                    + shlex.quote(
                        "END "
                        f"condition={run['condition']} seed={run['seed']}"
                    )
                ),
                "    ;;",
            ]
        )
    lines.extend(
        [
            "  *)",
            '    echo "Unknown SLURM_ARRAY_TASK_ID" >&2',
            "    exit 2",
            "    ;;",
            "esac",
            "",
        ]
    )
    return "\n".join(lines)


def write_characterization_output(
    mode,
    output_format,
    run_group,
    output_path,
):
    """Write commands plus manifest and return the manifest path."""
    runs = build_characterization_runs(mode=mode, run_group=run_group)
    manifest = create_manifest(mode, run_group, runs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        output_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return output_path
    if output_format == "shell":
        content = render_shell_script(runs, run_group)
    elif output_format == "sbatch":
        content = render_sbatch_script(runs, run_group)
    else:
        raise ValueError(f"unsupported output format: {output_format!r}")
    output_path.write_text(content)
    if output_format == "shell":
        output_path.chmod(output_path.stat().st_mode | 0o111)

    manifest_path = output_path.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "pilot"), required=True)
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("shell", "sbatch", "json"),
        required=True,
    )
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    manifest_path = write_characterization_output(
        mode=args.mode,
        output_format=args.output_format,
        run_group=args.run_group,
        output_path=args.output,
    )
    manifest = json.loads(manifest_path.read_text())
    if args.print_summary:
        print(f"mode: {manifest['mode']}")
        print(f"run_group: {manifest['run_group']}")
        print(f"run_count: {manifest['run_count']}")
        for run in manifest["runs"]:
            print(
                f"- {run['condition']} seed={run['seed']} "
                f"gain={run['kwargs']['train_action_gain']} "
                f"delay={run['kwargs']['train_action_delay']}"
            )
    print(f"Wrote {args.output}")
    print(f"Wrote {manifest_path}")
    print("No Slurm jobs were submitted.")


if __name__ == "__main__":
    main()
