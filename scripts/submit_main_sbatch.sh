#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. This script must be run on a Slurm server."
  exit 1
fi

if [ ! -d sbatch ] || ! find sbatch -name "main-experiments-part*.sh" ! -name "*_debug.sh" | grep -q .; then
  echo "Main sbatch scripts not found. Generating..."
  bash scripts/generate_main_sbatch.sh
fi

echo "================ Submit main experiments ================"

mapfile -t FILES < <(find sbatch -maxdepth 1 -name "main-experiments-part*.sh" ! -name "*_debug.sh" | sort -V)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "ERROR: No formal main experiment sbatch files found."
  exit 1
fi

echo "Found ${#FILES[@]} formal sbatch file(s)."

for f in "${FILES[@]}"; do
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY RUN] sbatch ${f}"
  else
    echo "Submitting ${f}"
    sbatch "${f}"
  fi
done

echo "Done."
