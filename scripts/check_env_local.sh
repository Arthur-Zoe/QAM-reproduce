#!/usr/bin/env bash
set -Eeuo pipefail

# Local environment check for QAM reproduction.
# This script is intended for local debug machines.
# It does not require Slurm / sbatch.

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
  echo "${YELLOW}[WARNING]${RESET} $*" >&2
}

fail() {
  echo "${RED}[ERROR]${RESET} $*" >&2
  exit 1
}

on_error() {
  echo
  echo "${RED}[FAILED]${RESET} Local environment check failed near line ${BASH_LINENO[0]}." >&2
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
PYTHON_BIN="$(command -v python3 || true)"
[ -n "$PYTHON_BIN" ] || fail "python3 not found"
echo "python3: ${PYTHON_BIN}"
python3 --version

section "Python imports"
python3 - <<'PY'
import sys
print("Python executable:", sys.executable)

import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())

import mujoco
print("MuJoCo import ok")

import wandb
print("wandb version:", wandb.__version__)
PY
ok "Python imports passed"

section "Optional commands"
if command -v sbatch >/dev/null 2>&1; then
  ok "sbatch found: $(command -v sbatch)"
else
  warn "sbatch not found. This is OK for local debug, but required on a Slurm server."
fi

if command -v parallel >/dev/null 2>&1; then
  ok "GNU parallel found: $(command -v parallel)"
else
  warn "GNU parallel not found. This is OK for local single debug, but required by generated sbatch scripts."
fi

section "Environment variables"
echo "MUJOCO_GL=${MUJOCO_GL:-not set}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-not set}"
echo "WANDB_MODE=${WANDB_MODE:-not set}"
echo "QAM_DATA_ROOT=${QAM_DATA_ROOT:-not set}"

if [ "${MUJOCO_GL:-}" != "egl" ]; then
  warn "MUJOCO_GL is not set to egl. For MuJoCo headless rendering, recommended: export MUJOCO_GL=egl"
else
  ok "MUJOCO_GL=egl"
fi

if [ "${WANDB_MODE:-}" = "online" ]; then
  ok "W&B mode: online"
elif [ "${WANDB_MODE:-}" = "offline" ]; then
  warn "W&B mode: offline. This is fine for local debug, but formal reproduction usually uses online."
elif [ -z "${WANDB_MODE:-}" ]; then
  warn "WANDB_MODE is not set."
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
ok "All local environment checks passed."
