#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# QAM reproduction bootstrap script - server-safe edition
# ==============================================================================
# 重点：
# - 清晰区分【本地验证模式】和【Slurm 服务器复现模式】
# - Slurm 模式下，不默认在登录节点直接跑训练
# - Slurm 模式下，check_env 和 debug 可以提交为 sbatch 作业
# - W&B online 作为默认正式复现模式
# - OGBench 100M 数据集可检查 / 可选下载
# - 正式提交 9600 runs 默认不执行，必须人工确认
# ==============================================================================

REPO_URL_DEFAULT="https://github.com/Arthur-Zoe/QAM-reproduce.git"
TARGET_DIR_DEFAULT="${HOME}/QAM-reproduce"
ENV_NAME_DEFAULT="qam"
PYTHON_VERSION_DEFAULT="3.10"
WANDB_PROJECT_DEFAULT="qam-reproduce"
DATA_ROOT_DEFAULT="${HOME}/ogbench_100m_data"
OGBENCH_BASE_URL="https://rail.eecs.berkeley.edu/datasets/ogbench"

REQUIRED_DATASETS=(
  "cube-quadruple-play-100m-v0"
  "puzzle-4x4-play-100m-v0"
)

# ------------------------------------------------------------------------------
# Colors
# ------------------------------------------------------------------------------

if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"
  RESET="$(tput sgr0)"
  RED="$(tput setaf 1)"
  GREEN="$(tput setaf 2)"
  YELLOW="$(tput setaf 3)"
  BLUE="$(tput setaf 4)"
  MAGENTA="$(tput setaf 5)"
  CYAN="$(tput setaf 6)"
else
  BOLD=""
  RESET=""
  RED=""
  GREEN=""
  YELLOW=""
  BLUE=""
  MAGENTA=""
  CYAN=""
fi

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

ask() {
  local prompt="$1"
  local default="${2:-}"
  local ans

  if [ -n "$default" ]; then
    read -r -p "$(echo -e "${BOLD}?${RESET} ${prompt} ${DIM:-}[默认: ${default}]${RESET}: ")" ans
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

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

normalize_dir_with_slash() {
  local p="${1:-}"
  p="${p%/}"
  if [ -z "$p" ]; then
    echo ""
  else
    echo "${p}/"
  fi
}

print_sudo_data_root_commands() {
  local data_root="${1%/}"

  echo
  echo "${BOLD}如果必须使用该共享目录，可以让有 sudo 权限的人手动执行：${RESET}"
  echo
  echo "  sudo mkdir -p ${data_root}"
  echo "  sudo chown -R \$USER:\$USER ${data_root}"
  echo
  echo "或者由管理员创建公共只读数据目录，然后你只设置 QAM_DATA_ROOT 指向它。"
  echo
}

handle_unwritable_data_root() {
  local data_root="${1%/}"

  echo
  danger "当前用户无法写入数据集目录：${data_root}"
  echo
  echo "请选择处理方式："
  echo "  1) 改用用户目录：\$HOME/ogbench_100m_data"
  echo "  2) 重新输入一个有写权限的数据集目录"
  echo "  3) 只打印 sudo 授权命令，由管理员/有权限的人手动执行"
  echo "  4) 退出脚本"
  echo

  local choice
  while true; do
    read -r -p "$(echo -e "${BOLD}?${RESET} 请选择 1/2/3/4 [默认: 1]: ")" choice
    choice="${choice:-1}"

    case "$choice" in
      1)
        QAM_DATA_ROOT="$(normalize_dir_with_slash "${HOME}/ogbench_100m_data")"
        export QAM_DATA_ROOT
        warn "已改用用户目录：${QAM_DATA_ROOT}"
        ensure_writable_data_root "$QAM_DATA_ROOT"
        return $?
        ;;
      2)
        local new_root
        new_root="$(ask '请重新输入 QAM_DATA_ROOT' "${HOME}/ogbench_100m_data")"
        QAM_DATA_ROOT="$(normalize_dir_with_slash "$new_root")"
        export QAM_DATA_ROOT
        ensure_writable_data_root "$QAM_DATA_ROOT"
        return $?
        ;;
      3)
        print_sudo_data_root_commands "$data_root"
        warn "已打印 sudo 命令。请管理员处理后重新运行脚本，或选择其他可写目录。"
        return 1
        ;;
      4)
        die "用户选择退出。"
        ;;
      *)
        echo "请输入 1、2、3 或 4。"
        ;;
    esac
  done
}

ensure_writable_data_root() {
  local data_root="${1:-}"

  if [ -z "$data_root" ]; then
    die "数据集路径为空，无法创建目录。"
  fi

  data_root="${data_root%/}"

  # 如果目录已存在，要求可写。
  if [ -d "$data_root" ]; then
    if [ -w "$data_root" ]; then
      ok "数据集根目录存在且当前用户可写：${data_root}"
      return 0
    else
      handle_unwritable_data_root "$data_root"
      return $?
    fi
  fi

  # 如果目录不存在，检查最近存在的父目录是否可写。
  local parent="$data_root"
  while [ ! -d "$parent" ] && [ "$parent" != "/" ]; do
    parent="$(dirname "$parent")"
  done

  if [ ! -d "$parent" ]; then
    danger "无法找到可用父目录：${data_root}"
    handle_unwritable_data_root "$data_root"
    return $?
  fi

  if [ ! -w "$parent" ]; then
    danger "目标目录不存在，且当前用户没有权限在父目录中创建：${parent}"
    echo "你填写的目标路径是：${data_root}"
    handle_unwritable_data_root "$data_root"
    return $?
  fi

  mkdir -p "$data_root"
  ok "已创建数据集根目录：${data_root}"
  return 0
}

dataset_dir_ok() {
  local data_root="${1:-}"

  if [ -z "$data_root" ]; then
    return 1
  fi

  data_root="${data_root%/}"

  for d in "${REQUIRED_DATASETS[@]}"; do
    if [ ! -d "${data_root}/${d}" ]; then
      return 1
    fi
    if ! find "${data_root}/${d}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
      return 1
    fi
  done

  return 0
}

show_dataset_status() {
  local data_root="${1:-}"

  if [ -z "$data_root" ]; then
    warn "QAM_DATA_ROOT 尚未设置。"
    return 0
  fi

  data_root="${data_root%/}"

  echo "QAM_DATA_ROOT=${data_root}/"

  for d in "${REQUIRED_DATASETS[@]}"; do
    if [ -d "${data_root}/${d}" ]; then
      local n
      n="$(find "${data_root}/${d}" -type f 2>/dev/null | wc -l || true)"
      echo "  ${GREEN}[FOUND]${RESET} ${d}  files=${n}"
    else
      echo "  ${RED}[MISS ]${RESET} ${d}"
    fi
  done
}

download_one_dataset() {
  local data_root="${1%/}"
  local dataset="$2"
  local url="${OGBENCH_BASE_URL}/${dataset}/"

  ensure_writable_data_root "${data_root}" || die "数据集目录不可写，无法下载：${data_root}"

  echo
  echo "${BOLD}准备下载数据集：${dataset}${RESET}"
  echo "来源：${url}"
  echo "目标：${data_root}/${dataset}"
  echo

  if [ -d "${data_root}/${dataset}" ] && find "${data_root}/${dataset}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    ok "数据集目录已存在且非空，跳过下载：${data_root}/${dataset}"
    return 0
  fi

  if command_exists wget; then
    (
      cd "${data_root}"
      wget -c -r -np -nH --cut-dirs=2 --reject "index.html*" "${url}"
    )
  else
    die "未找到 wget。请先安装 wget，或手动下载：${url}"
  fi
}

maybe_prepare_datasets() {
  local data_root="${1:-}"

  if [ -z "$data_root" ] || [ "$data_root" = "/path/to/ogbench_100m_data" ]; then
    data_root="$(ask 'OGBench 100M 数据集根目录 QAM_DATA_ROOT' "$DATA_ROOT_DEFAULT")"
  fi

  data_root="$(normalize_dir_with_slash "$data_root")"
  export QAM_DATA_ROOT="$data_root"

  step "检查 OGBench 100M 数据集"
  show_dataset_status "$QAM_DATA_ROOT"

  if dataset_dir_ok "$QAM_DATA_ROOT"; then
    ok "必要数据集已存在。"
    return 0
  fi

  echo
  danger "缺少必要 OGBench 100M 数据集。"
  echo "需要至少包含："
  for d in "${REQUIRED_DATASETS[@]}"; do
    echo "  - ${d}"
  done

  echo
  warn "100M 数据集体积较大，下载可能很慢，并且需要较大磁盘空间。"
  echo "如果当前只是本地 debug，可以选择 N。"
  echo "如果当前是服务器，建议管理员或学长提前准备数据，或在服务器上下载。"

  if confirm "是否现在从 OGBench 官方地址下载缺失数据集？" "N"; then
    ensure_writable_data_root "$QAM_DATA_ROOT" || die "QAM_DATA_ROOT 不可写，无法下载数据集。"

    for d in "${REQUIRED_DATASETS[@]}"; do
      if [ ! -d "${QAM_DATA_ROOT%/}/${d}" ] || ! find "${QAM_DATA_ROOT%/}/${d}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
        download_one_dataset "$QAM_DATA_ROOT" "$d"
      else
        ok "跳过已存在数据集：${d}"
      fi
    done

    echo
    echo "下载后重新检查："
    show_dataset_status "$QAM_DATA_ROOT"

    if ! dataset_dir_ok "$QAM_DATA_ROOT"; then
      die "数据集仍不完整，请检查下载日志或手动下载缺失目录。"
    fi

    ok "数据集准备完成。"
    return 0
  else
    return 1
  fi
}

choose_mode() {
  echo
  echo "${BOLD}请选择运行模式：${RESET}"
  echo
  echo "  ${GREEN}1) 本地验证模式${RESET}"
  echo "     用于个人电脑 / 非 Slurm 环境。"
  echo "     默认不下载 OGBench 100M 数据集，不生成正式 sbatch，不提交任务。"
  echo "     主要验证：JAX GPU、MuJoCo、W&B online、短 debug。"
  echo
  echo "  ${MAGENTA}2) Slurm 服务器复现模式${RESET}"
  echo "     用于学长的服务器 / 集群环境。"
  echo "     会检查 sbatch、GNU parallel、QAM_DATA_ROOT、数据集和 GPU。"
  echo "     check_env 和 debug 可提交到 Slurm 计算节点运行。"
  echo "     最后可生成并提交完整 9600 runs。"
  echo

  local mode
  while true; do
    read -r -p "$(echo -e "${BOLD}?${RESET} 请选择模式 1 或 2 [默认: 1]: ")" mode
    mode="${mode:-1}"
    case "$mode" in
      1)
        RUN_MODE="local"
        IS_SERVER=0
        ok "已选择：本地验证模式"
        break
        ;;
      2)
        RUN_MODE="slurm"
        IS_SERVER=1
        ok "已选择：Slurm 服务器复现模式"
        break
        ;;
      *)
        echo "请输入 1 或 2。"
        ;;
    esac
  done
}

write_env_file() {
  mkdir -p scripts logs

  cat > scripts/env_reproduce.sh <<EOF
#!/usr/bin/env bash
# 使用方法：
#   source scripts/env_reproduce.sh

export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB_PROJECT=${WANDB_PROJECT}
EOF

  if [ -n "${WANDB_ENTITY:-}" ]; then
    cat >> scripts/env_reproduce.sh <<EOF
export WANDB_ENTITY=${WANDB_ENTITY}
EOF
  fi

  if [ -n "${QAM_DATA_ROOT:-}" ]; then
    cat >> scripts/env_reproduce.sh <<EOF
export QAM_DATA_ROOT=${QAM_DATA_ROOT}
EOF
  fi

  chmod +x scripts/env_reproduce.sh
  ok "已写入环境变量文件：scripts/env_reproduce.sh"
  echo "以后可执行：source scripts/env_reproduce.sh"
}

configure_slurm_options() {
  step "配置 Slurm 作业参数"

  echo "如果服务器不需要 partition/account，可以直接留空。"

  SBATCH_PARTITION="$(ask 'Slurm partition，可留空' "${SBATCH_PARTITION:-}")"
  SBATCH_ACCOUNT="$(ask 'Slurm account，可留空' "${SBATCH_ACCOUNT:-}")"
  SBATCH_GRES="$(ask 'GPU 资源参数 --gres，例如 gpu:1' "${SBATCH_GRES:-gpu:1}")"
  SBATCH_CPUS="$(ask 'CPU 数量 --cpus-per-task' "${SBATCH_CPUS:-4}")"
  SBATCH_MEM="$(ask '内存 --mem' "${SBATCH_MEM:-16G}")"
  SBATCH_TIME_CHECK="$(ask 'check_env 作业时间 --time' "${SBATCH_TIME_CHECK:-00:20:00}")"
  SBATCH_TIME_DEBUG="$(ask 'debug 作业时间 --time' "${SBATCH_TIME_DEBUG:-01:00:00}")"

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

create_slurm_check_job() {
  mkdir -p sbatch logs
  local job_file="sbatch/qam-check-env-one.sbatch"
  local log_file="logs/slurm-check-env-%j.out"

  {
    write_sbatch_header "qam-check-env" "$log_file" "$SBATCH_TIME_CHECK"
    cat <<EOF

set -euo pipefail

cd "$TARGET_DIR"

eval "\$("$CONDA_BASE/bin/conda" shell.bash hook)"
conda activate "$ENV_NAME"

source scripts/env_reproduce.sh

echo "Running on host: \$(hostname)"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-not set}"
echo "Date: \$(date)"

bash scripts/check_env.sh
EOF
  } > "$job_file"

  chmod +x "$job_file"
  echo "$job_file"
}

create_slurm_debug_job() {
  mkdir -p sbatch logs
  local job_file="sbatch/qam-debug-one.sbatch"
  local log_file="logs/slurm-debug-one-%j.out"

  {
    write_sbatch_header "qam-debug-one" "$log_file" "$SBATCH_TIME_DEBUG"
    cat <<EOF

set -euo pipefail

cd "$TARGET_DIR"

eval "\$("$CONDA_BASE/bin/conda" shell.bash hook)"
conda activate "$ENV_NAME"

source scripts/env_reproduce.sh

echo "Running on host: \$(hostname)"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-not set}"
echo "Date: \$(date)"

bash scripts/run_debug_one.sh
EOF
  } > "$job_file"

  chmod +x "$job_file"
  echo "$job_file"
}

submit_slurm_job() {
  local job_file="$1"
  echo "提交 Slurm 作业：${job_file}"
  local output
  output="$(sbatch "$job_file")"
  echo "$output"

  local job_id
  job_id="$(echo "$output" | awk '{print $NF}')"

  if [ -z "$job_id" ]; then
    warn "未能解析 job id，请手动使用 squeue/sacct 检查。"
    return 0
  fi

  ok "已提交 job id: ${job_id}"

  if confirm "是否等待该作业结束并显示日志位置？" "Y"; then
    echo "等待作业结束：${job_id}"
    while squeue -j "$job_id" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$job_id" -h 2>/dev/null)" ]; do
      squeue -j "$job_id" || true
      sleep 10
    done

    echo
    echo "作业已离开队列。sacct 状态："
    sacct -j "$job_id" --format=JobID,JobName,State,ExitCode,Elapsed || true

    echo
    echo "相关日志文件："
    find logs -maxdepth 1 -type f \( -name "*${job_id}.out" -o -name "*${job_id}.log" \) -print || true
  fi
}

run_env_check() {
  step "环境检查"

  if [ "$IS_SERVER" -eq 1 ]; then
    echo "${MAGENTA}Slurm 服务器模式：推荐把 check_env.sh 提交到计算节点执行。${RESET}"
    echo "原因：很多集群登录节点没有 GPU，直接运行 check_env.sh 可能误判 JAX GPU 不可用。"
    echo

    echo "选择环境检查方式："
    echo "  1) 提交 Slurm check_env 作业（推荐）"
    echo "  2) 直接在当前节点运行 scripts/check_env.sh"
    echo "  3) 跳过"
    local choice
    read -r -p "$(echo -e "${BOLD}?${RESET} 请选择 1/2/3 [默认: 1]: ")" choice
    choice="${choice:-1}"

    case "$choice" in
      1)
        local job_file
        job_file="$(create_slurm_check_job)"
        submit_slurm_job "$job_file"
        ;;
      2)
        bash scripts/check_env.sh
        ;;
      3)
        warn "已跳过服务器环境检查。"
        ;;
      *)
        die "无效选择：${choice}"
        ;;
    esac
  else
    echo "${GREEN}本地验证模式：运行 scripts/check_env_local.sh${RESET}"
    if confirm "是否运行本地环境检查 scripts/check_env_local.sh？" "Y"; then
      bash scripts/check_env_local.sh
    fi
  fi
}

run_debug() {
  step "短 Debug"

  if [ "$IS_SERVER" -eq 1 ]; then
    echo "${MAGENTA}Slurm 服务器模式：推荐把短 debug 提交到计算节点执行。${RESET}"
    echo "原因：很多集群不允许在登录节点直接占用 GPU 跑训练。"
    echo

    echo "选择短 debug 运行方式："
    echo "  1) 提交 Slurm debug 作业（推荐）"
    echo "  2) 直接在当前节点运行 scripts/run_debug_one.sh"
    echo "  3) 跳过"
    local choice
    read -r -p "$(echo -e "${BOLD}?${RESET} 请选择 1/2/3 [默认: 1]: ")" choice
    choice="${choice:-1}"

    case "$choice" in
      1)
        local job_file
        job_file="$(create_slurm_debug_job)"
        submit_slurm_job "$job_file"
        ;;
      2)
        source scripts/env_reproduce.sh
        bash scripts/run_debug_one.sh
        ;;
      3)
        warn "已跳过短 debug。"
        ;;
      *)
        die "无效选择：${choice}"
        ;;
    esac
  else
    if confirm "是否运行一次短 debug？" "Y"; then
      source scripts/env_reproduce.sh
      bash scripts/run_debug_one.sh
    else
      warn "已跳过短 debug。"
    fi
  fi
}

# ==============================================================================
# Main
# ==============================================================================

banner "QAM reproduction bootstrap - server-safe edition"

echo "这个脚本会帮助你从 git clone 开始配置 QAM 复现实验。"
echo "涉及账号、数据路径、生成 sbatch、短 debug、正式提交等步骤时会询问 Y/N。"
danger "正式提交 9600 个实验任务默认不会自动执行。"

choose_mode

REPO_URL="$(ask 'Git 仓库地址' "$REPO_URL_DEFAULT")"
TARGET_DIR="$(ask '本地仓库目录' "$TARGET_DIR_DEFAULT")"
ENV_NAME="$(ask 'Conda 环境名' "$ENV_NAME_DEFAULT")"
PYTHON_VERSION="$(ask 'Python 版本' "$PYTHON_VERSION_DEFAULT")"

step "检查基础命令"
command_exists git || die "未找到 git，请先安装 git。"
command_exists conda || die "未找到 conda。请先安装 Miniconda/Anaconda，并确保 conda 在 PATH 中。"

if [ "$IS_SERVER" -eq 1 ]; then
  command_exists sbatch || die "Slurm 模式需要 sbatch，但当前未找到。"
  command_exists squeue || warn "未找到 squeue，后续无法自动等待作业状态。"
  command_exists sacct || warn "未找到 sacct，后续无法查看作业历史状态。"
fi
ok "基础命令检查通过。"

eval "$(conda shell.bash hook)"

step "获取仓库"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "发现已有仓库：$TARGET_DIR"
  cd "$TARGET_DIR"
  echo "当前分支：$(git branch --show-current || true)"
  if confirm "是否执行 git pull --rebase origin main？" "Y"; then
    git fetch origin
    git switch main || git switch -c main origin/main
    git pull --rebase origin main
  fi
elif [ -e "$TARGET_DIR" ]; then
  die "目标路径已存在但不是 Git 仓库：$TARGET_DIR"
else
  git clone "$REPO_URL" "$TARGET_DIR"
  cd "$TARGET_DIR"
fi

echo
echo "${BOLD}仓库状态：${RESET}"
git rev-parse HEAD
git status -sb

step "准备 Conda 环境"

CONDA_BASE="$(conda info --base)"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda 环境已存在：$ENV_NAME"
  if confirm "是否继续使用该环境？选择 N 会退出，避免误删环境" "Y"; then
    :
  else
    die "已取消。你可以手动删除环境后重新运行：conda env remove -n ${ENV_NAME}"
  fi
elif [ -d "$ENV_PREFIX" ]; then
  warn "发现环境目录已存在但 conda env list 中没有正常注册：${ENV_PREFIX}"
  if confirm "是否删除该目录并重新创建环境？" "N"; then
    rm -rf "$ENV_PREFIX"
    conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
  else
    die "已取消。请手动处理异常环境目录：${ENV_PREFIX}"
  fi
else
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

conda activate "$ENV_NAME"

echo "当前 Python：$(which python || true)"
echo "当前 python3：$(which python3 || true)"
python --version || python3 --version

step "安装 Python 依赖"

if confirm "是否安装 requirements.txt？" "Y"; then
  if confirm "是否使用清华 PyPI 镜像？国内网络建议选 Y" "Y"; then
    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  else
    python -m pip install -r requirements.txt
  fi
else
  warn "跳过 requirements.txt 安装。"
fi

step "安装 / 检查 JAX GPU"

if confirm "是否安装或更新 GPU 版 JAX？" "Y"; then
  JAX_CUDA="$(ask '选择 JAX CUDA wheel：cuda12 或 cuda13' 'cuda12')"
  case "$JAX_CUDA" in
    cuda12|cuda13) ;;
    *) die "JAX CUDA wheel 只能填写 cuda12 或 cuda13。" ;;
  esac

  if confirm "是否使用清华 PyPI 镜像安装 jax[${JAX_CUDA}]？" "Y"; then
    python -m pip install -U "jax[${JAX_CUDA}]" -i https://pypi.tuna.tsinghua.edu.cn/simple
  else
    python -m pip install -U "jax[${JAX_CUDA}]"
  fi
fi

python - <<'PY'
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
PY

step "配置实验环境变量"

export MUJOCO_GL="egl"

if [ "$IS_SERVER" -eq 1 ]; then
  echo "${MAGENTA}当前为 Slurm 服务器复现模式。${RESET}"
  QAM_DATA_ROOT="$(ask 'QAM_DATA_ROOT，即 OGBench 100M 数据集根目录' "${QAM_DATA_ROOT:-$DATA_ROOT_DEFAULT}")"
  export QAM_DATA_ROOT="$(normalize_dir_with_slash "$QAM_DATA_ROOT")"

  echo "提示：如果选择在脚本中下载数据集，请确认当前用户对 QAM_DATA_ROOT 有写权限。"
  echo "如果使用管理员提前准备好的只读共享数据集目录，只要目录存在且数据完整即可。"

  if confirm "是否现在检查 / 下载 OGBench 100M 数据集？" "Y"; then
    if ! maybe_prepare_datasets "$QAM_DATA_ROOT"; then
      warn "数据集尚未准备完整。后续 check_env.sh 或 generate_main_sbatch.sh 可能失败。"
    fi
  fi
else
  echo "${GREEN}当前为本地验证模式。${RESET}"
  echo "本地短 debug 可以不设置 QAM_DATA_ROOT；生成正式 sbatch 需要 QAM_DATA_ROOT。"

  if confirm "是否现在配置 / 下载 OGBench 100M 数据集？本地只做 debug 可选 N" "N"; then
    QAM_DATA_ROOT="$(ask 'QAM_DATA_ROOT，即 OGBench 100M 数据集根目录' "${QAM_DATA_ROOT:-$DATA_ROOT_DEFAULT}")"
    export QAM_DATA_ROOT="$(normalize_dir_with_slash "$QAM_DATA_ROOT")"
    if ! maybe_prepare_datasets "$QAM_DATA_ROOT"; then
      warn "数据集尚未准备完整。"
    fi
  else
    QAM_DATA_ROOT="${QAM_DATA_ROOT:-}"
  fi
fi

WANDB_PROJECT="$(ask 'W&B 项目名' "${WANDB_PROJECT:-$WANDB_PROJECT_DEFAULT}")"
export WANDB_MODE="online"
export WANDB_PROJECT

WANDB_ENTITY_INPUT="$(ask 'W&B Entity / 团队名，可留空' "${WANDB_ENTITY:-}")"
if [ -n "$WANDB_ENTITY_INPUT" ]; then
  export WANDB_ENTITY="$WANDB_ENTITY_INPUT"
fi

write_env_file

echo
echo "${BOLD}当前关键环境变量：${RESET}"
echo "  MUJOCO_GL=${MUJOCO_GL}"
echo "  WANDB_MODE=${WANDB_MODE}"
echo "  WANDB_PROJECT=${WANDB_PROJECT}"
echo "  WANDB_ENTITY=${WANDB_ENTITY:-not set}"
echo "  QAM_DATA_ROOT=${QAM_DATA_ROOT:-not set}"

if [ "$IS_SERVER" -eq 1 ]; then
  configure_slurm_options
fi

step "W&B 登录"

echo "正式复现实验推荐使用 W&B online。"
echo "如果未登录，wandb login 会提示打开网页并粘贴 API key。"
danger "API key 不要发给别人，也不要提交到 GitHub。"

if confirm "是否现在执行 wandb login？" "Y"; then
  wandb login
fi

if confirm "是否运行一个极小的 W&B online smoke test？" "Y"; then
  python - <<'PY'
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

step "检查仓库脚本权限"
chmod +x scripts/*.sh 2>/dev/null || true
ok "脚本权限检查完成。"

run_env_check

step "生成主实验 sbatch 脚本"

if [ "$IS_SERVER" -eq 1 ]; then
  GEN_DEFAULT="Y"
else
  GEN_DEFAULT="N"
fi

if confirm "是否生成主实验 sbatch 脚本？会删除并重建 sbatch/ 目录" "$GEN_DEFAULT"; then
  if ! dataset_dir_ok "${QAM_DATA_ROOT:-}"; then
    danger "生成正式 sbatch 需要完整 OGBench 100M 数据集。"
    if maybe_prepare_datasets "${QAM_DATA_ROOT:-}"; then
      write_env_file
    else
      die "未准备完整 QAM_DATA_ROOT，无法生成正式 sbatch。"
    fi
  fi

  export QAM_DATA_ROOT="$(normalize_dir_with_slash "$QAM_DATA_ROOT")"
  bash scripts/generate_main_sbatch.sh
else
  warn "已跳过主实验 sbatch 生成。"
fi

run_debug

if [ "$IS_SERVER" -eq 1 ]; then
  step "Slurm 正式提交"

  if confirm "是否执行 dry-run，查看将提交哪些 sbatch 文件？" "Y"; then
    bash scripts/submit_main_sbatch.sh --dry-run
  fi

  echo
  danger "下一步会正式提交完整主实验。"
  echo "完整实验规模是 9600 runs。请确认："
  echo "  1. check_env 已通过，且是在计算节点上通过"
  echo "  2. generate_main_sbatch.sh 显示 Formal experiment count: 9600"
  echo "  3. 短 debug 已在 Slurm 计算节点上成功"
  echo "  4. W&B online 已成功记录"
  echo "  5. QAM_DATA_ROOT 数据路径正确"
  echo

  if confirm "是否现在正式提交完整主实验？默认 N" "N"; then
    bash scripts/submit_main_sbatch.sh
  else
    ok "已跳过正式提交。之后可手动执行：bash scripts/submit_main_sbatch.sh"
  fi
else
  echo
  ok "当前不是 Slurm 服务器模式，跳过正式提交。"
fi

banner "完成"

echo "当前仓库：$TARGET_DIR"
echo "当前环境：$ENV_NAME"
echo "环境变量文件：$TARGET_DIR/scripts/env_reproduce.sh"
echo
echo "${BOLD}下次进入仓库后可执行：${RESET}"
echo "  conda activate $ENV_NAME"
echo "  cd $TARGET_DIR"
echo "  source scripts/env_reproduce.sh"
echo
if [ "$IS_SERVER" -eq 1 ]; then
  echo "${BOLD}服务器正式流程：${RESET}"
  echo "  bash scripts/check_env.sh                    # 或用本脚本生成的 Slurm check 作业"
  echo "  bash scripts/generate_main_sbatch.sh"
  echo "  bash scripts/run_debug_one.sh                # 推荐在 Slurm 作业里运行"
  echo "  bash scripts/submit_main_sbatch.sh --dry-run"
  echo "  bash scripts/submit_main_sbatch.sh"
else
  echo "${BOLD}本地验证建议：${RESET}"
  echo "  bash scripts/check_env_local.sh"
  echo "  bash scripts/run_debug_one.sh"
fi
