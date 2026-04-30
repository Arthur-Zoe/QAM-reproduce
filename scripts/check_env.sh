#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "================ Repo ================"
pwd
test -f main.py
test -d agents
test -d envs
test -d experiments
test -d utils

echo "================ Python ================"
which python3
python3 --version

echo "================ Required commands ================"
command -v git
command -v python3

if command -v sbatch >/dev/null 2>&1; then
  echo "sbatch found: $(which sbatch)"
else
  echo "ERROR: sbatch not found. Slurm is required for full experiments."
  exit 1
fi

if command -v parallel >/dev/null 2>&1; then
  echo "GNU parallel found: $(which parallel)"
else
  echo "ERROR: GNU parallel not found. Generated sbatch scripts require GNU parallel."
  echo "Try: conda install -c conda-forge parallel"
  exit 1
fi

echo "================ Python imports and GPU check ================"
python3 - <<'PY'
import os
import sys

print("Python executable:", sys.executable)

import jax
print("JAX version:", jax.__version__)
devices = jax.devices()
print("JAX devices:", devices)

require_gpu = os.environ.get("QAM_REQUIRE_GPU", "1") != "0"
has_gpu = any(
    ("gpu" in getattr(d, "platform", "").lower()) or ("cuda" in str(d).lower())
    for d in devices
)

if require_gpu and not has_gpu:
    raise SystemExit(
        "ERROR: JAX did not detect GPU. Full reproduction should not run on CPU."
    )

import mujoco
print("MuJoCo import ok")

import wandb
print("wandb version:", wandb.__version__)
PY

echo "================ QAM_DATA_ROOT ================"
if [ -z "${QAM_DATA_ROOT:-}" ]; then
  echo "ERROR: QAM_DATA_ROOT is not set."
  echo "Please run:"
  echo "  export QAM_DATA_ROOT=/path/to/ogbench_100m_data"
  exit 1
fi

DATA_ROOT="${QAM_DATA_ROOT%/}"
echo "QAM_DATA_ROOT=${DATA_ROOT}"

if [ ! -d "${DATA_ROOT}" ]; then
  echo "ERROR: QAM_DATA_ROOT does not exist: ${DATA_ROOT}"
  exit 1
fi

for d in cube-quadruple-play-100m-v0 puzzle-4x4-play-100m-v0; do
  if [ ! -d "${DATA_ROOT}/${d}" ]; then
    echo "ERROR: Missing required 100M dataset directory:"
    echo "  ${DATA_ROOT}/${d}"
    exit 1
  fi
done

echo "================ Env Vars ================"
echo "MUJOCO_GL=${MUJOCO_GL:-not set}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-not set}"
echo "WANDB_MODE=${WANDB_MODE:-not set}"

if [ "${MUJOCO_GL:-}" != "egl" ]; then
  echo "WARNING: MUJOCO_GL is not set to egl. On headless GPU servers, use:"
  echo "  export MUJOCO_GL=egl"
fi

if [ -z "${WANDB_PROJECT:-}" ]; then
  echo "WARNING: WANDB_PROJECT is not set. Recommended:"
  echo "  export WANDB_PROJECT=qam-reproduce"
fi

echo "================ Syntax check ================"
python3 -m py_compile main.py evaluation.py log_utils.py
python3 -m py_compile agents/*.py
python3 -m py_compile envs/*.py
python3 -m py_compile utils/*.py
python3 -m py_compile experiments/*.py

echo "Server environment checks passed."
