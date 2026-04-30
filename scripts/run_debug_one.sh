#!/usr/bin/env bash
set -Eeuo pipefail

# Run one short local/server debug job.
# This is only for environment validation, not a formal experiment result.

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
  echo "${RED}[FAILED]${RESET} Debug run failed near line ${BASH_LINENO[0]}." >&2
  echo "Please check the log file printed above." >&2
}
trap on_error ERR

cd "$(dirname "$0")/.."

export MUJOCO_GL=${MUJOCO_GL:-egl}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-qam-reproduce-local-debug}

DEBUG_RUN_GROUP=${DEBUG_RUN_GROUP:-local_debug_$(date +%Y%m%d_%H%M%S)}

mkdir -p logs

LOG_FILE="logs/${DEBUG_RUN_GROUP}.log"

section "Short debug run"
echo "Repository: $(pwd)"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_ENTITY=${WANDB_ENTITY:-not set}"
echo "DEBUG_RUN_GROUP=${DEBUG_RUN_GROUP}"
echo "LOG_FILE=${LOG_FILE}"

if [ "${WANDB_MODE}" = "online" ]; then
  ok "W&B online mode enabled. This run should appear on the W&B website."
elif [ "${WANDB_MODE}" = "offline" ]; then
  warn "W&B offline mode enabled. This run will be saved locally and not uploaded immediately."
else
  warn "WANDB_MODE=${WANDB_MODE}"
fi

section "Command"
cat <<EOF
python3 main.py \\
  --run_group="${DEBUG_RUN_GROUP}" \\
  --agent=agents/qam.py \\
  --tags=QAM_FQL_LOCAL_DEBUG \\
  --seed=10001 \\
  --env_name=cube-triple-play-singletask-task2-v0 \\
  --offline_steps=100 \\
  --online_steps=100 \\
  --eval_episodes=1 \\
  --video_episodes=0
EOF

section "Running"
set +e
python3 main.py \
  --run_group="${DEBUG_RUN_GROUP}" \
  --agent=agents/qam.py \
  --tags=QAM_FQL_LOCAL_DEBUG \
  --seed=10001 \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --sparse=False \
  --horizon_length=5 \
  --offline_steps=100 \
  --online_steps=100 \
  --log_interval=20 \
  --eval_interval=50 \
  --save_interval=50 \
  --start_training=20 \
  --eval_episodes=1 \
  --video_episodes=0 \
  --agent.action_chunking=True \
  --agent.inv_temp=10.0 \
  --agent.fql_alpha=300.0 \
  --agent.edit_scale=0.0 \
  2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e

section "Debug result"
if [ "${STATUS}" -ne 0 ]; then
  echo "${RED}[FAILED]${RESET} python3 main.py exited with status ${STATUS}."
  echo "Log file: ${LOG_FILE}"
  echo "Useful checks:"
  echo "  grep -n \"Traceback\\|ERROR\\|CUDA_ERROR\\|out of memory\" ${LOG_FILE}"
  exit "${STATUS}"
fi

if grep -q "Traceback" "${LOG_FILE}"; then
  fail "Traceback found in log file: ${LOG_FILE}"
fi

if grep -q "CUDA_ERROR_OUT_OF_MEMORY\|out of memory" "${LOG_FILE}"; then
  warn "The log contains possible GPU OOM messages. Please inspect: ${LOG_FILE}"
fi

if grep -q "wandb: .*View run" "${LOG_FILE}"; then
  ok "W&B run URL was printed in the log."
elif [ "${WANDB_MODE}" = "online" ]; then
  warn "W&B online mode was requested, but no obvious W&B run URL was detected in the log."
fi

ok "Short debug run finished successfully."
echo "Log file: ${LOG_FILE}"
