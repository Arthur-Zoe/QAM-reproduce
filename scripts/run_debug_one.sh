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

METHODS=(QAM QAM_FQL QAM_EDIT)

section "Short debug runs"
echo "Repository: $(pwd)"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_ENTITY=${WANDB_ENTITY:-not set}"
echo "DEBUG_RUN_GROUP=${DEBUG_RUN_GROUP}"
echo "Methods: ${METHODS[*]}"

if [ "${WANDB_MODE}" = "online" ]; then
  ok "W&B online mode enabled. This run should appear on the W&B website."
elif [ "${WANDB_MODE}" = "offline" ]; then
  warn "W&B offline mode enabled. This run will be saved locally and not uploaded immediately."
else
  warn "WANDB_MODE=${WANDB_MODE}"
fi

for METHOD in "${METHODS[@]}"; do
  case "${METHOD}" in
    QAM)
      INV_TEMP=3.0
      FQL_ALPHA=0.0
      EDIT_SCALE=0.0
      ;;
    QAM_FQL)
      INV_TEMP=10.0
      FQL_ALPHA=300.0
      EDIT_SCALE=0.0
      ;;
    QAM_EDIT)
      INV_TEMP=3.0
      FQL_ALPHA=0.0
      EDIT_SCALE=0.1
      ;;
    *)
      fail "Unknown debug method: ${METHOD}"
      ;;
  esac

  METHOD_RUN_GROUP="${DEBUG_RUN_GROUP}-${METHOD}"
  LOG_FILE="logs/${METHOD_RUN_GROUP}.log"

  section "Command: ${METHOD}"
  cat <<EOF
python3 main.py \\
  --run_group="${METHOD_RUN_GROUP}" \\
  --agent=agents/qam.py \\
  --tags=${METHOD} \\
  --seed=10001 \\
  --env_name=cube-triple-play-singletask-task2-v0 \\
  --offline_steps=100 \\
  --online_steps=100 \\
  --eval_episodes=1 \\
  --video_episodes=0 \\
  --agent.inv_temp=${INV_TEMP} \\
  --agent.fql_alpha=${FQL_ALPHA} \\
  --agent.edit_scale=${EDIT_SCALE}
EOF

  section "Running: ${METHOD}"
  set +e
  python3 main.py \
    --run_group="${METHOD_RUN_GROUP}" \
    --agent=agents/qam.py \
    --tags="${METHOD}" \
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
    --agent.inv_temp="${INV_TEMP}" \
    --agent.fql_alpha="${FQL_ALPHA}" \
    --agent.edit_scale="${EDIT_SCALE}" \
    2>&1 | tee "${LOG_FILE}"
  STATUS=${PIPESTATUS[0]}
  set -e

  if [ "${STATUS}" -ne 0 ]; then
    echo "${RED}[FAILED]${RESET} ${METHOD} exited with status ${STATUS}."
    echo "Log file: ${LOG_FILE}"
    echo "Useful checks:"
    echo "  grep -n \"Traceback\\|ERROR\\|CUDA_ERROR\\|out of memory\" ${LOG_FILE}"
    exit "${STATUS}"
  fi

  if grep -q "Traceback" "${LOG_FILE}"; then
    fail "Traceback found in log file: ${LOG_FILE}"
  fi

  if grep -q "CUDA_ERROR_OUT_OF_MEMORY\|out of memory" "${LOG_FILE}"; then
    warn "${METHOD} log contains possible GPU OOM messages. Please inspect: ${LOG_FILE}"
  fi

  if grep -q "wandb: .*View run" "${LOG_FILE}"; then
    ok "${METHOD}: W&B run URL was printed in the log."
  elif [ "${WANDB_MODE}" = "online" ]; then
    warn "${METHOD}: W&B online mode was requested, but no obvious W&B run URL was detected in the log."
  fi

  ok "${METHOD} short debug run finished successfully."
  echo "Log file: ${LOG_FILE}"
done

section "Debug result"
ok "All QAM debug runs finished successfully."
