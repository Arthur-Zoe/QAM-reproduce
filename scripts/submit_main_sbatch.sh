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
  echo "${YELLOW}[WARNING]${RESET} $*" >&2
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

if ! command -v sbatch >/dev/null 2>&1; then
  fail "sbatch not found. This script must be run on a Slurm server."
fi
ok "sbatch found: $(command -v sbatch)"

if [ ! -d sbatch ] || ! find sbatch -maxdepth 1 -name "main-experiments-part*.sh" ! -name "*_debug.sh" | grep -q .; then
  warn "Formal main sbatch scripts not found. Generating them now..."
  bash scripts/generate_main_sbatch.sh
fi

mapfile -t FILES < <(find sbatch -maxdepth 1 -name "main-experiments-part*.sh" ! -name "*_debug.sh" | sort -V)

if [ "${#FILES[@]}" -eq 0 ]; then
  fail "No formal main experiment sbatch files found."
fi

section "Files to submit"
echo "Formal sbatch file count: ${#FILES[@]}"
printf '  %s\n' "${FILES[@]}"

if find sbatch -maxdepth 1 -name "*_debug.sh" | grep -q .; then
  ok "Debug sbatch files exist, but they will NOT be submitted by this script."
fi

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
