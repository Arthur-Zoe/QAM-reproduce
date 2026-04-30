#!/usr/bin/env bash
set -euo pipefail

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

echo "================ Python imports ================"
python3 - <<'PY'
import sys
print("Python executable:", sys.executable)

try:
    import jax
    print("JAX version:", jax.__version__)
    print("JAX devices:", jax.devices())
except Exception as e:
    print("JAX import/check failed:", repr(e))
    raise

try:
    import mujoco
    print("MuJoCo import ok")
except Exception as e:
    print("MuJoCo import failed:", repr(e))
    raise

try:
    import wandb
    print("wandb version:", wandb.__version__)
except Exception as e:
    print("wandb import failed:", repr(e))
    raise
PY

echo "================ Optional commands ================"
if command -v sbatch >/dev/null 2>&1; then
  echo "sbatch found: $(which sbatch)"
else
  echo "WARNING: sbatch not found. This is okay for local debug, but required on Slurm server."
fi

if command -v parallel >/dev/null 2>&1; then
  echo "GNU parallel found: $(which parallel)"
else
  echo "WARNING: GNU parallel not found. This is okay for local single debug, but required for generated sbatch scripts."
fi

echo "================ Env Vars ================"
echo "MUJOCO_GL=${MUJOCO_GL:-not set}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-not set}"
echo "WANDB_MODE=${WANDB_MODE:-not set}"
echo "QAM_DATA_ROOT=${QAM_DATA_ROOT:-not set}"

echo "================ Syntax check ================"
python3 -m py_compile main.py evaluation.py log_utils.py
python3 -m py_compile agents/*.py
python3 -m py_compile envs/*.py
python3 -m py_compile utils/*.py
python3 -m py_compile experiments/*.py

echo "All local environment checks passed."
