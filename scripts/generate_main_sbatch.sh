#!/usr/bin/env bash
set -Eeuo pipefail

# Generate formal and debug Slurm scripts for QAM reproduction.

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
  echo "${RED}[FAILED]${RESET} Main sbatch generation failed near line ${BASH_LINENO[0]}." >&2
  echo "Please read the message above. Common causes: Python errors in experiments/reproduce.py or missing external datasets when enabled." >&2
}
trap on_error ERR

cd "$(dirname "$0")/.."

section "Generate main sbatch scripts"
echo "Repository: $(pwd)"

section "Matrix summary"
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
for item in "${MATRIX_INFO[@]}"; do
  case "${item}" in
    EXPECTED_FORMAL_COUNT=*) EXPECTED_FORMAL_COUNT="${item#EXPECTED_FORMAL_COUNT=}" ;;
    EXPECTED_DEBUG_COUNT=*) EXPECTED_DEBUG_COUNT="${item#EXPECTED_DEBUG_COUNT=}" ;;
    REQUIRED_EXTERNAL_DATASET_DIRS=*) REQUIRED_EXTERNAL_DATASET_DIRS="${item#REQUIRED_EXTERNAL_DATASET_DIRS=}" ;;
    SUMMARY=*) echo "${item#SUMMARY=}" ;;
  esac
done

if [ -n "${REQUIRED_EXTERNAL_DATASET_DIRS}" ]; then
  if [ -z "${QAM_DATA_ROOT:-}" ]; then
    fail "QAM_DATA_ROOT is required because the current matrix includes external 100M domains: ${REQUIRED_EXTERNAL_DATASET_DIRS}"
  fi

  DATA_ROOT="${QAM_DATA_ROOT%/}"
  if [ ! -d "${DATA_ROOT}" ]; then
    fail "QAM_DATA_ROOT does not exist: ${DATA_ROOT}"
  fi

  for d in ${REQUIRED_EXTERNAL_DATASET_DIRS}; do
    if [ ! -d "${DATA_ROOT}/${d}" ]; then
      fail "Missing required dataset directory: ${DATA_ROOT}/${d}"
    fi
  done
  ok "QAM_DATA_ROOT check passed for required external domains: ${DATA_ROOT}"
else
  if [ -z "${QAM_DATA_ROOT:-}" ]; then
    warn "QAM_DATA_ROOT is not set. This is OK for the current 240-run matrix."
  elif [ ! -d "${QAM_DATA_ROOT%/}" ]; then
    warn "QAM_DATA_ROOT does not exist, but no current matrix domain requires it: ${QAM_DATA_ROOT%/}"
  else
    ok "QAM_DATA_ROOT is optional for the current matrix: ${QAM_DATA_ROOT%/}"
  fi
fi

warn "This script will delete and regenerate the sbatch/ directory."
rm -rf sbatch
mkdir -p sbatch
ok "Prepared clean sbatch/ directory"

section "Run experiments/reproduce.py"
python3 experiments/reproduce.py
ok "experiments/reproduce.py finished"

section "Count generated commands"
FORMAL_COUNT=$(find sbatch -name "main-experiments-part*.sh" ! -name "*_debug.sh" -print0 \
  | xargs -0 grep -h "python main.py" 2>/dev/null \
  | wc -l)

DEBUG_COUNT=$(find sbatch -name "main-experiments-part*_debug.sh" -print0 \
  | xargs -0 grep -h "python main.py" 2>/dev/null \
  | wc -l)

echo "Formal experiment count: ${FORMAL_COUNT}"
echo "Debug experiment count: ${DEBUG_COUNT}"
echo "Expected formal experiment count: ${EXPECTED_FORMAL_COUNT}"
echo "Expected debug experiment count: ${EXPECTED_DEBUG_COUNT}"

if [ "${FORMAL_COUNT}" -ne "${EXPECTED_FORMAL_COUNT}" ]; then
  fail "Expected ${EXPECTED_FORMAL_COUNT} formal experiments, got ${FORMAL_COUNT}. Do not submit formal experiments."
else
  ok "Formal experiment count is correct: ${EXPECTED_FORMAL_COUNT}"
fi

if [ "${DEBUG_COUNT}" -ne "${EXPECTED_DEBUG_COUNT}" ]; then
  fail "Expected ${EXPECTED_DEBUG_COUNT} debug experiments, got ${DEBUG_COUNT}. Do not submit formal experiments."
else
  ok "Debug experiment count is correct: ${EXPECTED_DEBUG_COUNT}"
fi

section "Generated files"
FORMAL_FILES=$(find sbatch -maxdepth 1 -name "main-experiments-part*.sh" ! -name "*_debug.sh" | sort -V | wc -l)
DEBUG_FILES=$(find sbatch -maxdepth 1 -name "main-experiments-part*_debug.sh" | sort -V | wc -l)
echo "Formal sbatch files: ${FORMAL_FILES}"
echo "Debug sbatch files: ${DEBUG_FILES}"

section "Result"
ok "Main sbatch generation passed."
echo "Next recommended command:"
echo "  bash scripts/submit_main_sbatch.sh --dry-run"
