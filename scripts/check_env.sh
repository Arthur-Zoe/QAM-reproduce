#!/usr/bin/env bash
set -Eeuo pipefail

# Server environment check for QAM reproduction.
# This script is intended for Slurm servers.

if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; RESET="$(tput sgr0)"
  RED="$(tput setaf 1)"; GREEN="$(tput setaf 2)"
  YELLOW="$(tput setaf 3)"; BLUE="$(tput setaf 4)"
else
  BOLD=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""
fi

section() {
  echo
  echo "${BOLD}${BLUE}================ $* ================${RESET}"
}

ok() {
  echo "${GREEN}[OK]${RESET} $*"
}

warn() {
  echo "${YELLOW}[WARN]${RESET} $*" >&2
}

fail() {
  echo "${RED}[ERROR]${RESET} $*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    ok "${cmd} found: $(command -v "$cmd")"
  else
    fail "${cmd} not found"
  fi
}

on_error() {
  echo
  echo "${RED}[FAILED]${RESET} Server environment check failed near line ${BASH_LINENO[0]}." >&2
  echo "Please read the message above and fix the first reported error." >&2
}
trap on_error ERR

cd "$(dirname "$0")/.."

section "Repo"
echo "Repository: $(pwd)"

for p in main.py agents envs experiments utils; do
  if [ -e "$p" ]; then
    ok "Found ${p}"
  else
    fail "Missing required path: ${p}"
  fi
done

section "Python"
require_cmd python3
python3 --version

section "Required commands"
require_cmd git
require_cmd python3
if command -v sbatch >/dev/null 2>&1; then
  ok "sbatch found: $(command -v sbatch)"
else
  warn "sbatch not found. This is OK on a local machine; required only on Slurm server for actual submission."
fi
if command -v parallel >/dev/null 2>&1; then
  ok "parallel found: $(command -v parallel)"
else
  warn "parallel not found. This is OK on a local machine for generation and dry-run; required on the Slurm server when running generated sbatch scripts."
fi

section "Python imports and JAX GPU check"
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
    raise SystemExit("ERROR: JAX did not detect GPU. Full reproduction should not run on CPU.")

import mujoco
print("MuJoCo import ok")

import wandb
print("wandb version:", wandb.__version__)
PY
ok "Python imports and GPU check passed"

section "QAM_DATA_ROOT"
mapfile -t REQUIRED_EXTERNAL_DATASET_DIRS < <(PYTHONPATH=experiments python3 - <<'PY'
from qam_matrix import required_external_dataset_dirs, validate_matrix

validate_matrix()
for path in required_external_dataset_dirs():
    print(path)
PY
)

if [ "${#REQUIRED_EXTERNAL_DATASET_DIRS[@]}" -eq 0 ]; then
  if [ -z "${QAM_DATA_ROOT:-}" ]; then
    warn "QAM_DATA_ROOT is not set. This is OK for the current 240-run matrix."
  elif [ -d "${QAM_DATA_ROOT%/}" ]; then
    ok "QAM_DATA_ROOT is set but optional for the current matrix: ${QAM_DATA_ROOT%/}"
  else
    warn "QAM_DATA_ROOT does not exist, but no current matrix domain requires it: ${QAM_DATA_ROOT%/}"
  fi
else
  if [ -z "${QAM_DATA_ROOT:-}" ]; then
    fail "QAM_DATA_ROOT is required because the current matrix includes external 100M domains: ${REQUIRED_EXTERNAL_DATASET_DIRS[*]}"
  fi

  DATA_ROOT="${QAM_DATA_ROOT%/}"
  echo "QAM_DATA_ROOT=${DATA_ROOT}"

  if [ ! -d "${DATA_ROOT}" ]; then
    fail "QAM_DATA_ROOT does not exist: ${DATA_ROOT}"
  fi

  for d in "${REQUIRED_EXTERNAL_DATASET_DIRS[@]}"; do
    if [ -d "${DATA_ROOT}/${d}" ]; then
      FILE_COUNT="$(find "${DATA_ROOT}/${d}" -type f 2>/dev/null | wc -l || true)"
      ok "Found dataset directory: ${DATA_ROOT}/${d}  files=${FILE_COUNT}"
    else
      fail "Missing required 100M dataset directory: ${DATA_ROOT}/${d}"
    fi
  done
fi

section "Environment variables"
echo "MUJOCO_GL=${MUJOCO_GL:-not set}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-not set}"
echo "WANDB_MODE=${WANDB_MODE:-not set}"
echo "WANDB_ENTITY=${WANDB_ENTITY:-not set}"

if [ "${MUJOCO_GL:-}" != "egl" ]; then
  warn "MUJOCO_GL is not set to egl. On headless GPU servers, use: export MUJOCO_GL=egl"
else
  ok "MUJOCO_GL=egl"
fi

if [ -z "${WANDB_PROJECT:-}" ]; then
  warn "WANDB_PROJECT is not set. Recommended: export WANDB_PROJECT=qam-reproduce"
else
  ok "WANDB_PROJECT=${WANDB_PROJECT}"
fi

if [ "${WANDB_MODE:-}" = "online" ]; then
  ok "W&B mode: online"
elif [ "${WANDB_MODE:-}" = "offline" ]; then
  warn "W&B mode: offline. This is acceptable only if the server cannot access W&B online."
elif [ -z "${WANDB_MODE:-}" ]; then
  warn "WANDB_MODE is not set. Recommended: export WANDB_MODE=online"
else
  warn "WANDB_MODE=${WANDB_MODE}"
fi

section "Syntax check"
python3 -m py_compile main.py evaluation.py log_utils.py
python3 -m py_compile agents/*.py
python3 -m py_compile envs/*.py
python3 -m py_compile utils/*.py
python3 -m py_compile experiments/*.py
ok "Python syntax check passed"

section "Result"
ok "Server environment checks passed."
