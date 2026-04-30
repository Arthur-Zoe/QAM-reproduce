#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "================ Generate main sbatch scripts ================"

rm -rf sbatch
mkdir -p sbatch

python3 experiments/reproduce.py

echo "================ Count generated commands ================"

FORMAL_COUNT=$(find sbatch -name "main-experiments-part*.sh" ! -name "*_debug.sh" -print0 \
  | xargs -0 grep -h "python main.py" \
  | wc -l)

DEBUG_COUNT=$(find sbatch -name "main-experiments-part*_debug.sh" -print0 \
  | xargs -0 grep -h "python main.py" \
  | wc -l)

echo "Formal experiment count: ${FORMAL_COUNT}"
echo "Debug experiment count: ${DEBUG_COUNT}"

if [ "${FORMAL_COUNT}" -ne 9600 ]; then
  echo "ERROR: Expected 9600 formal experiments, got ${FORMAL_COUNT}"
  exit 1
fi

if [ "${DEBUG_COUNT}" -ne 160 ]; then
  echo "ERROR: Expected 160 debug experiments, got ${DEBUG_COUNT}"
  exit 1
fi

echo "Main sbatch generation passed."
