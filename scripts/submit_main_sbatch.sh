#!/usr/bin/env bash
set -Eeuo pipefail

# Submit formal QAM reproduction sbatch scripts.
# Use --dry-run first to preview commands without submitting jobs.

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

on_error() {
  echo
  echo "${RED}[FAILED]${RESET} Submit script failed near line ${BASH_LINENO[0]}." >&2
  echo "Please read the message above and fix the first reported error." >&2
}
trap on_error ERR

cd "$(dirname "$0")/.."

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

section "Submit main experiments"

if [ "${DRY_RUN}" -eq 1 ]; then
  warn "DRY-RUN mode: commands will be printed, but no Slurm jobs will be submitted."
else
  warn "LIVE SUBMIT mode: formal Slurm jobs will be submitted."
fi

if command -v sbatch >/dev/null 2>&1; then
  ok "sbatch found: $(command -v sbatch)"
elif [ "${DRY_RUN}" -eq 1 ]; then
  warn "sbatch not found. This is OK on a local machine; required only on Slurm server for actual submission."
else
  fail "sbatch not found. Actual submission must be run on a Slurm server."
fi

if [ ! -d sbatch ] || ! find sbatch -maxdepth 1 -name "main-experiments-part*.sh" ! -name "*_debug.sh" | grep -q .; then
  fail "Formal main sbatch scripts not found. Run: bash scripts/generate_main_sbatch.sh"
fi

mapfile -t FILES < <(find sbatch -maxdepth 1 -name "main-experiments-part*.sh" ! -name "*_debug.sh" | sort -V)

if [ "${#FILES[@]}" -eq 0 ]; then
  fail "No formal main experiment sbatch files found."
fi

mapfile -t MATRIX_INFO < <(PYTHONPATH=experiments python3 - <<'PY'
from qam_matrix import METHODS, formal_experiment_count, matrix_summary_lines, validate_matrix

validate_matrix()
print(f"EXPECTED_FORMAL_COUNT={formal_experiment_count()}")
print(f"ALLOWED_METHODS={' '.join(METHODS)}")
for line in matrix_summary_lines():
    print(f"SUMMARY={line}")
PY
)

EXPECTED_FORMAL_COUNT=""
ALLOWED_METHODS=""
for item in "${MATRIX_INFO[@]}"; do
  case "${item}" in
    EXPECTED_FORMAL_COUNT=*) EXPECTED_FORMAL_COUNT="${item#EXPECTED_FORMAL_COUNT=}" ;;
    ALLOWED_METHODS=*) ALLOWED_METHODS="${item#ALLOWED_METHODS=}" ;;
  esac
done

section "Files to submit"
echo "Formal sbatch file count: ${#FILES[@]}"
echo "Allowed methods: ${ALLOWED_METHODS}"
printf '  %s\n' "${FILES[@]}"

if find sbatch -maxdepth 1 -name "*_debug.sh" | grep -q .; then
  ok "Debug sbatch files exist, but they will NOT be submitted by this script."
fi

FORMAL_COUNT=$(grep -h "python main.py" "${FILES[@]}" 2>/dev/null | wc -l)
echo "Formal experiment count: ${FORMAL_COUNT}"
echo "Expected formal experiment count: ${EXPECTED_FORMAL_COUNT}"

if [ "${FORMAL_COUNT}" -ne "${EXPECTED_FORMAL_COUNT}" ]; then
  fail "Expected ${EXPECTED_FORMAL_COUNT} formal experiments, got ${FORMAL_COUNT}. Regenerate with scripts/generate_main_sbatch.sh before submitting."
fi

OLD_METHODS="FQL|FEDIT|BC|IQL|CRL|CGQL|QSM|DAC|FBRAC|IFQL|CGQL_MSE|CGQL_LINEX|DSRL|FAWAC|BAM|REBRAC|RLPD"
if grep -Eh -- "--tags=(${OLD_METHODS})( |$)" "${FILES[@]}" >/dev/null; then
  fail "Found old baseline tags in formal sbatch files. Regenerate with scripts/generate_main_sbatch.sh before submitting."
fi
if grep -Eh -- "--run_group=[^ ]*-(${OLD_METHODS})( |$)" "${FILES[@]}" >/dev/null; then
  fail "Found old baseline W&B run groups in formal sbatch files. Regenerate with scripts/generate_main_sbatch.sh before submitting."
fi
if grep -Eh -- "--agent=agents/(fql|fedit|cgql|dcgql|fbrac|ifql|dsrl|fawac|bam|rebrac|rlpd)\\.py( |$)" "${FILES[@]}" >/dev/null; then
  fail "Found old baseline agent files in formal sbatch files. Regenerate with scripts/generate_main_sbatch.sh before submitting."
fi
ok "Formal sbatch files only reference QAM, QAM_FQL, and QAM_EDIT tags."

section "Submission"
SUBMITTED=0

for f in "${FILES[@]}"; do
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY RUN] sbatch ${f}"
  else
    echo "Submitting ${f}"
    sbatch "${f}"
    SUBMITTED=$((SUBMITTED + 1))
  fi
done

section "Result"
if [ "${DRY_RUN}" -eq 1 ]; then
  ok "Dry-run finished. No jobs were submitted."
  echo "If the file list is correct, run:"
  echo "  bash scripts/submit_main_sbatch.sh"
else
  ok "Submitted ${SUBMITTED} formal sbatch file(s)."
  echo "Check jobs with:"
  echo "  squeue -u \$USER"
  echo "  sacct -u \$USER --format=JobID,JobName,State,ExitCode,Elapsed"
fi
