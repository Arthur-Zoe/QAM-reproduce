# QAM-Chunk Phase-1 Reproduction

本仓库当前用于 QAM-Chunk 后续方向的第一阶段复现。目标不是完整复刻 QAM 论文所有 baseline，而是先稳定复现 QAM 方法族。

主实验入口：

```text
experiments/reproduce.py
```

唯一实验矩阵来源：

```text
experiments/qam_matrix.py
```

当前主实验规模：

```text
3 methods x 4 domains x 4 tasks x 5 seeds = 240 runs
```

当前 debug 规模：

```text
3 methods x 4 domains x 1 task x 1 seed = 12 runs
```

## Current Matrix

methods:

```text
QAM
QAM_FQL
QAM_EDIT
```

domains:

```text
cube-triple-play
scene-play-sparse
puzzle-3x3-play-sparse
antmaze-large-navigate
```

tasks:

```text
1, 2, 3, 4
```

seeds:

```text
10001, 20002, 30003, 40004, 50005
```

debug task:

```text
task2
```

The current main matrix does not include old large-scale domains:

```text
humanoidmaze-large-navigate
humanoidmaze-medium-navigate
antmaze-giant-navigate
cube-quadruple-play
cube-double-play
puzzle-4x4-play-sparse
```

## Environment

Create and activate a Python environment, then install dependencies:

```bash
conda create -n qam python=3.10 -y
conda activate qam
pip install -r requirements.txt
```

Recommended runtime variables:

```bash
export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB_PROJECT=qam-reproduce
# Optional, only for team projects:
# export WANDB_ENTITY=your_wandb_entity
```

`QAM_DATA_ROOT` is optional for the current 240-run matrix.
It is only required when enabling external 100M domains such as `cube-quadruple-play` or `puzzle-4x4-play-sparse`.

If those external domains are added back later, set:

```bash
export QAM_DATA_ROOT=/path/to/ogbench_100m_data
```

The expected external directories are:

```text
cube-quadruple-play-100m-v0/
puzzle-4x4-play-100m-v0/
```

## Check Environment

On a Slurm server:

```bash
bash scripts/check_env.sh
```

This checks repository paths, Python, Slurm commands, JAX GPU, MuJoCo, W&B, optional external data requirements for the active matrix, and Python syntax.

For local machines without Slurm:

```bash
bash scripts/check_env_local.sh
```

## Generate Slurm Scripts

```bash
bash scripts/generate_main_sbatch.sh
```

Expected counts:

```text
Formal experiment count: 240
Debug experiment count: 12
Main sbatch generation passed.
```

The script deletes and regenerates `sbatch/`. Do not keep manual files there.

Expected generated files:

```text
sbatch/main-experiments-part1.sh
sbatch/main-experiments-part1_debug.sh
```

`main-experiments-part2.sh` and `main-experiments-part3.sh` are previous large-matrix leftovers and should not be regenerated for the current matrix.

## Short Debug

Before submitting formal jobs:

```bash
bash scripts/run_debug_one.sh
```

This runs short local/server smoke tests for:

```text
QAM
QAM_FQL
QAM_EDIT
```

## Dry-Run And Submit

Preview formal sbatch submission:

```bash
bash scripts/submit_main_sbatch.sh --dry-run
```

Submit formal jobs after reviewing the dry-run output:

```bash
bash scripts/submit_main_sbatch.sh
```

The submit script only submits formal `sbatch/main-experiments-part*.sh` files and does not submit debug scripts.

## One-Click Workflow

Interactive one-click flow:

```bash
bash scripts/bootstrap_qam_reproduce.sh
```

Top-level compatibility entry:

```bash
bash "One-click -script.sh"
```

The workflow is scoped to the current QAM-Chunk phase-1 matrix. It asks for local vs Slurm mode, optional `WANDB_ENTITY`, W&B login / smoke test, sbatch generation confirmation, optional short debug, and explicit formal submission confirmation.

It writes fixed defaults for `MUJOCO_GL=egl`, `WANDB_MODE=online`, and `WANDB_PROJECT=qam-reproduce`. The current 240-run matrix does not use external 100M domains, so `QAM_DATA_ROOT` is not configured by the one-click workflow. Environment check runs automatically as `QAM_REQUIRE_GPU=0 bash scripts/check_env.sh`, and dry-run runs automatically after generation.

## Recommended Manual Flow

```bash
conda activate qam

wandb login

export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB_PROJECT=qam-reproduce
# export WANDB_ENTITY=your_wandb_entity

bash scripts/check_env.sh
bash scripts/generate_main_sbatch.sh
bash scripts/run_debug_one.sh
bash scripts/submit_main_sbatch.sh --dry-run
bash scripts/submit_main_sbatch.sh
```
