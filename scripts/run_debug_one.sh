#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MUJOCO_GL=${MUJOCO_GL:-egl}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-qam-reproduce-local-debug}

DEBUG_RUN_GROUP=${DEBUG_RUN_GROUP:-local_debug_$(date +%Y%m%d_%H%M%S)}

mkdir -p logs

LOG_FILE="logs/${DEBUG_RUN_GROUP}.log"

echo "================ Local debug run ================"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "DEBUG_RUN_GROUP=${DEBUG_RUN_GROUP}"
echo "LOG_FILE=${LOG_FILE}"

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

echo "Local debug run finished."
