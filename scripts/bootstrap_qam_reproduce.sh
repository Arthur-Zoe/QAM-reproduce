#!/usr/bin/env bash
set -Eeuo pipefail

# Interactive one-click workflow for the current QAM-Chunk phase-1 matrix.
# This file can run from scripts/ or be copied to the repository root.

if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; RESET="$(tput sgr0)"
  RED="$(tput setaf 1)"; GREEN="$(tput setaf 2)"
  YELLOW="$(tput setaf 3)"; BLUE="$(tput setaf 4)"
  MAGENTA="$(tput setaf 5)"; CYAN="$(tput setaf 6)"
else
  BOLD=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/scripts" ] && [ -d "${SCRIPT_DIR}/experiments" ]; then
  cd "${SCRIPT_DIR}"
else
  cd "${SCRIPT_DIR}/.."
fi

WANDB_PROJECT_DEFAULT="qam-reproduce"
SBATCH_CPUS_DEFAULT="4"
SBATCH_MEM_DEFAULT="16G"
SBATCH_GRES_DEFAULT="gpu:1"
SBATCH_TIME_CHECK_DEFAULT="00:20:00"
SBATCH_TIME_DEBUG_DEFAULT="01:00:00"

hr() {
  printf '%*s\n' "${COLUMNS:-88}" '' | tr ' ' '-'
}

banner() {
  echo
  hr
  echo "${BOLD}${CYAN}$*${RESET}"
  hr
}

step() {
  echo
  echo "${BOLD}${BLUE}▶ $*${RESET}"
}

ok() {
  echo "${GREEN}✓ $*${RESET}"
}

warn() {
  echo "${YELLOW}⚠ $*${RESET}" >&2
}

danger() {
  echo "${RED}!! $*${RESET}" >&2
}

die() {
  echo "${RED}ERROR: $*${RESET}" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

ask() {
  local prompt="$1"
  local default="${2:-}"
  local ans

  if [ -n "$default" ]; then
    read -r -p "$(echo -e "${BOLD}?${RESET} ${prompt} [默认: ${default}]: ")" ans
    echo "${ans:-$default}"
  else
    read -r -p "$(echo -e "${BOLD}?${RESET} ${prompt}: ")" ans
    echo "$ans"
  fi
}

confirm() {
  local prompt="$1"
  local default="${2:-N}"
  local ans

  while true; do
    read -r -p "$(echo -e "${BOLD}?${RESET} ${prompt} [Y/N，默认 ${default}]: ")" ans
    ans="${ans:-$default}"
    case "$ans" in
      Y|y|yes|YES) return 0 ;;
      N|n|no|NO) return 1 ;;
      *) echo "请输入 Y 或 N。" ;;
    esac
  done
}

load_matrix_info() {
  mapfile -t MATRIX_INFO < <(PYTHONPATH=experiments python3 - <<'PY'
from qam_matrix import (
    debug_experiment_count,
    formal_experiment_count,
    matrix_summary_lines,
    required_external_dataset_dirs,
    validate_matrix,
)

validate_matrix()
print(f"EXPECTED_FORMAL_COUNT={formal_experiment_count()}")
print(f"EXPECTED_DEBUG_COUNT={debug_experiment_count()}")
print(f"REQUIRED_EXTERNAL_DATASET_DIRS={' '.join(required_external_dataset_dirs())}")
for line in matrix_summary_lines():
    print(f"SUMMARY={line}")
PY
)

  EXPECTED_FORMAL_COUNT=""
  EXPECTED_DEBUG_COUNT=""
  REQUIRED_EXTERNAL_DATASET_DIRS=""
  MATRIX_SUMMARY_LINES=()

  for item in "${MATRIX_INFO[@]}"; do
    case "${item}" in
      EXPECTED_FORMAL_COUNT=*) EXPECTED_FORMAL_COUNT="${item#EXPECTED_FORMAL_COUNT=}" ;;
      EXPECTED_DEBUG_COUNT=*) EXPECTED_DEBUG_COUNT="${item#EXPECTED_DEBUG_COUNT=}" ;;
      REQUIRED_EXTERNAL_DATASET_DIRS=*) REQUIRED_EXTERNAL_DATASET_DIRS="${item#REQUIRED_EXTERNAL_DATASET_DIRS=}" ;;
      SUMMARY=*) MATRIX_SUMMARY_LINES+=("${item#SUMMARY=}") ;;
    esac
  done
}

dataset_dir_ok() {
  [ -z "${REQUIRED_EXTERNAL_DATASET_DIRS:-}" ]
}

show_dataset_status() {
  if [ -z "${REQUIRED_EXTERNAL_DATASET_DIRS:-}" ]; then
    echo "当前 ${EXPECTED_FORMAL_COUNT}-run 矩阵不包含 external 100M domains，QAM_DATA_ROOT will be ignored / not required."
    return 0
  fi

  die "当前 bootstrap 只支持 QAM-Chunk phase-1 240-run 矩阵；检测到 external 100M domains: ${REQUIRED_EXTERNAL_DATASET_DIRS}"
}

choose_mode() {
  echo
  echo "${BOLD}请选择运行模式：${RESET}"
  echo
  echo "  ${GREEN}1) 本地验证模式${RESET}"
  echo "     用于个人电脑 / 非 Slurm 环境。"
  echo "     默认允许完成环境检查、sbatch 生成、计数验证和 dry-run。"
  echo "     不会因为本地缺少 sbatch 而失败。"
  echo
  echo "  ${MAGENTA}2) Slurm 服务器复现模式${RESET}"
  echo "     用于学长服务器 / 集群环境。"
  echo "     可选择把环境检查或短 debug 提交到 Slurm 计算节点。"
  echo "     正式提交前会强制要求 sbatch。"
  echo

  local mode
  while true; do
    read -r -p "$(echo -e "${BOLD}?${RESET} 请选择模式 1 或 2 [默认: 1]: ")" mode
    mode="${mode:-1}"
    case "$mode" in
      1) RUN_MODE="local"; IS_SERVER=0; ok "已选择：本地验证模式"; break ;;
      2) RUN_MODE="slurm"; IS_SERVER=1; ok "已选择：Slurm 服务器复现模式"; break ;;
      *) echo "请输入 1 或 2。" ;;
    esac
  done
}

write_env_file() {
  mkdir -p scripts logs
  {
    echo '#!/usr/bin/env bash'
    echo '# 使用方法：source scripts/env_reproduce.sh'
    echo "export MUJOCO_GL=${MUJOCO_GL}"
    echo "export WANDB_MODE=${WANDB_MODE}"
    echo "export WANDB_PROJECT=${WANDB_PROJECT}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
      echo "export WANDB_ENTITY=${WANDB_ENTITY}"
    fi
  } > scripts/env_reproduce.sh
  chmod +x scripts/env_reproduce.sh
  ok "已写入环境变量文件：scripts/env_reproduce.sh"
}

configure_env_vars() {
  step "配置实验环境变量"

  MUJOCO_GL="egl"
  WANDB_MODE="online"
  WANDB_PROJECT="$WANDB_PROJECT_DEFAULT"
  WANDB_ENTITY_INPUT="$(ask 'WANDB_ENTITY / 团队名，可留空')"

  export MUJOCO_GL WANDB_MODE WANDB_PROJECT
  if [ -n "$WANDB_ENTITY_INPUT" ]; then
    export WANDB_ENTITY="$WANDB_ENTITY_INPUT"
  else
    unset WANDB_ENTITY
  fi

  show_dataset_status
  write_env_file

  echo
  echo "${BOLD}当前关键环境变量：${RESET}"
  echo "  MUJOCO_GL=${MUJOCO_GL}"
  echo "  WANDB_MODE=${WANDB_MODE}"
  echo "  WANDB_PROJECT=${WANDB_PROJECT}"
  echo "  WANDB_ENTITY=${WANDB_ENTITY:-not set}"
  echo "  QAM_DATA_ROOT=ignored / not required"
}

configure_slurm_options() {
  step "配置 Slurm 作业参数"

  SBATCH_PARTITION="$(ask 'Slurm partition，可留空' "${SBATCH_PARTITION:-}")"
  SBATCH_ACCOUNT="$(ask 'Slurm account，可留空' "${SBATCH_ACCOUNT:-}")"
  SBATCH_GRES="$(ask 'GPU 资源参数 --gres，例如 gpu:1' "${SBATCH_GRES:-$SBATCH_GRES_DEFAULT}")"
  SBATCH_CPUS="$(ask 'CPU 数量 --cpus-per-task' "${SBATCH_CPUS:-$SBATCH_CPUS_DEFAULT}")"
  SBATCH_MEM="$(ask '内存 --mem' "${SBATCH_MEM:-$SBATCH_MEM_DEFAULT}")"
  SBATCH_TIME_CHECK="$(ask 'check_env 作业时间 --time' "${SBATCH_TIME_CHECK:-$SBATCH_TIME_CHECK_DEFAULT}")"
  SBATCH_TIME_DEBUG="$(ask 'debug 作业时间 --time' "${SBATCH_TIME_DEBUG:-$SBATCH_TIME_DEBUG_DEFAULT}")"

  echo
  echo "${BOLD}Slurm 参数摘要：${RESET}"
  echo "  partition=${SBATCH_PARTITION:-not set}"
  echo "  account=${SBATCH_ACCOUNT:-not set}"
  echo "  gres=${SBATCH_GRES}"
  echo "  cpus=${SBATCH_CPUS}"
  echo "  mem=${SBATCH_MEM}"
  echo "  check time=${SBATCH_TIME_CHECK}"
  echo "  debug time=${SBATCH_TIME_DEBUG}"
}

write_sbatch_header() {
  local job_name="$1"
  local output_log="$2"
  local time_limit="$3"

  cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${output_log}
#SBATCH --error=${output_log}
#SBATCH --time=${time_limit}
#SBATCH --cpus-per-task=${SBATCH_CPUS}
#SBATCH --mem=${SBATCH_MEM}
#SBATCH --gres=${SBATCH_GRES}
EOF

  if [ -n "${SBATCH_PARTITION:-}" ]; then
    echo "#SBATCH --partition=${SBATCH_PARTITION}"
  fi

  if [ -n "${SBATCH_ACCOUNT:-}" ]; then
    echo "#SBATCH --account=${SBATCH_ACCOUNT}"
  fi
}

create_slurm_debug_job() {
  mkdir -p sbatch logs
  local job_file="sbatch/qam-debug-one.sbatch"
  local log_file="logs/slurm-debug-one-%j.out"

  {
    write_sbatch_header "qam-debug-one" "$log_file" "$SBATCH_TIME_DEBUG"
    cat <<'EOF'

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/env_reproduce.sh

echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "Date: $(date)"

bash scripts/run_debug_one.sh
EOF
  } > "$job_file"

  chmod +x "$job_file"
  echo "$job_file"
}

submit_slurm_job() {
  local job_file="$1"
  command_exists sbatch || die "sbatch not found. Actual submission must be run on a Slurm server."
  echo "提交 Slurm 作业：${job_file}"
  sbatch "$job_file"
}

run_env_check() {
  step "环境检查"
  echo "自动执行：QAM_REQUIRE_GPU=0 bash scripts/check_env.sh"
  QAM_REQUIRE_GPU=0 bash scripts/check_env.sh
}

generate_sbatch() {
  step "生成主实验 sbatch 脚本"
  show_dataset_status

  if ! dataset_dir_ok; then
    die "当前 bootstrap 只支持不需要 external 100M domains 的 240-run 矩阵。"
  fi

  bash scripts/generate_main_sbatch.sh
}

run_debug() {
  step "短 Debug"

  if [ "$IS_SERVER" -eq 1 ]; then
    echo "选择短 debug 运行方式："
    echo "  1) 提交 Slurm debug 作业（推荐）"
    echo "  2) 直接在当前节点运行 scripts/run_debug_one.sh"
    echo "  3) 跳过"
    local choice
    read -r -p "$(echo -e "${BOLD}?${RESET} 请选择 1/2/3 [默认: 3]: ")" choice
    choice="${choice:-3}"
    case "$choice" in
      1) submit_slurm_job "$(create_slurm_debug_job)" ;;
      2) source scripts/env_reproduce.sh; bash scripts/run_debug_one.sh ;;
      3) warn "已跳过短 debug。" ;;
      *) die "无效选择：${choice}" ;;
    esac
  else
    if confirm "是否运行一次 cube-triple-task2 短 debug？会依次测试 QAM/QAM_FQL/QAM_EDIT" "N"; then
      source scripts/env_reproduce.sh
      bash scripts/run_debug_one.sh
    else
      warn "已跳过短 debug。"
    fi
  fi
}

run_dry_run() {
  step "Submit dry-run"
  bash scripts/submit_main_sbatch.sh --dry-run
}

maybe_submit_formal() {
  step "正式提交"
  danger "正式提交会把 formal sbatch 文件提交到 Slurm。当前 formal count 应为 ${EXPECTED_FORMAL_COUNT}。"

  if [ "$IS_SERVER" -ne 1 ] && [ "${QAM_SUBMIT:-0}" != "1" ]; then
    warn "当前是本地验证模式，只完成验证；正式提交只允许在 Slurm 服务器模式或 QAM_SUBMIT=1 时执行。"
    return 0
  fi

  if confirm "是否现在正式提交？" "N"; then
    bash scripts/submit_main_sbatch.sh
  else
    ok "已跳过正式提交。之后可手动执行：bash scripts/submit_main_sbatch.sh"
  fi
}

maybe_wandb_login() {
  step "W&B 登录 / smoke test"
  echo "正式复现实验推荐使用 W&B online。API key 不要提交到 GitHub。"

  if confirm "是否现在执行 wandb login？" "N"; then
    wandb login
  fi

  if confirm "是否运行一个极小的 W&B online smoke test？" "N"; then
    python3 - <<'PY'
import os
import wandb

project = os.environ.get("WANDB_PROJECT", "qam-reproduce")
entity = os.environ.get("WANDB_ENTITY") or None

run = wandb.init(project=project, entity=entity, name="wandb-online-smoke-test")
wandb.log({"bootstrap_test_metric": 1})
print("W&B run url:", run.url)
run.finish()
PY
  fi
}

banner "QAM reproduction bootstrap"

load_matrix_info

echo "这个脚本保留交互式一键流程：模式选择、环境变量、环境检查、sbatch 生成、debug、dry-run 和正式提交确认。"
echo
echo "${BOLD}当前矩阵：${RESET}"
printf '  %s\n' "${MATRIX_SUMMARY_LINES[@]}"
danger "正式提交默认不会自动执行，必须人工确认。"

choose_mode

step "检查基础命令"
command_exists git || die "未找到 git，请先安装 git。"
command_exists python3 || die "未找到 python3。"
if [ "$IS_SERVER" -eq 1 ]; then
  command_exists sbatch || warn "当前未找到 sbatch。只有提交 Slurm 作业或正式提交时才会失败。"
  command_exists squeue || warn "未找到 squeue，后续无法自动查看队列。"
  command_exists sacct || warn "未找到 sacct，后续无法查看作业历史。"
fi
ok "基础命令检查完成。"

configure_env_vars

if [ "$IS_SERVER" -eq 1 ]; then
  configure_slurm_options
fi

maybe_wandb_login

run_env_check

if confirm "是否生成主实验 sbatch 脚本？会删除并重建 sbatch/ 目录" "Y"; then
  generate_sbatch
else
  warn "已跳过主实验 sbatch 生成。"
fi

run_debug

run_dry_run

maybe_submit_formal

banner "完成"

echo "当前仓库：$(pwd)"
echo "环境变量文件：$(pwd)/scripts/env_reproduce.sh"
echo
echo "${BOLD}常用后续命令：${RESET}"
echo "  source scripts/env_reproduce.sh"
echo "  bash scripts/check_env.sh"
echo "  bash scripts/generate_main_sbatch.sh"
echo "  bash scripts/submit_main_sbatch.sh --dry-run"
echo "  bash scripts/submit_main_sbatch.sh"
echo
echo "本地只完成验证；服务器上先提交 debug sbatch，再提交正式 ${EXPECTED_FORMAL_COUNT} runs。"
