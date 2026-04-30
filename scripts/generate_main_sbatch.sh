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
  echo "Please read the message above. Common causes: missing QAM_DATA_ROOT, missing datasets, or Python errors in experiments/reproduce.py." >&2
}
trap on_error ERR

cd "$(dirname "$0")/.."

section "Generate main sbatch scripts"
echo "Repository: $(pwd)"

if [ -z "${QAM_DATA_ROOT:-}" ]; then
  fail "QAM_DATA_ROOT is not set. Set it before generating formal sbatch scripts, e.g. export QAM_DATA_ROOT=/path/to/ogbench_100m_data/"
fi

DATA_ROOT="${QAM_DATA_ROOT%/}"
if [ ! -d "${DATA_ROOT}" ]; then
  fail "QAM_DATA_ROOT does not exist: ${DATA_ROOT}"
fi

for d in cube-quadruple-play-100m-v0 puzzle-4x4-play-100m-v0; do
  if [ ! -d "${DATA_ROOT}/${d}" ]; then
    fail "Missing required dataset directory: ${DATA_ROOT}/${d}"
  fi
done
ok "QAM_DATA_ROOT check passed: ${DATA_ROOT}"

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

if [ "${FORMAL_COUNT}" -ne 9600 ]; then
  fail "Expected 9600 formal experiments, got ${FORMAL_COUNT}. Do not submit formal experiments."
else
  ok "Formal experiment count is correct: 9600"
fi

if [ "${DEBUG_COUNT}" -ne 160 ]; then
  fail "Expected 160 debug experiments, got ${DEBUG_COUNT}. Do not submit formal experiments."
else
  ok "Debug experiment count is correct: 160"
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
